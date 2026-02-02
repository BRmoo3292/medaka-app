from fastapi import FastAPI, UploadFile, HTTPException, Request
from fastapi.responses import FileResponse, Response, StreamingResponse
import uvicorn
from fastapi.middleware.cors import CORSMiddleware
from openai import AsyncOpenAI
import time
import numpy as np
import asyncio
from collections import defaultdict, deque
import tempfile
import os
from datetime import datetime
import psycopg2
from psycopg2 import pool
from psycopg2.extras import RealDictCursor
import atexit

# ========================================
# 環境変数・API設定
# ========================================
DB_URL = os.getenv("DB_URL")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
openai_client = AsyncOpenAI(api_key=OPENAI_API_KEY)

print(f"[起動時] DB_URL設定: {'あり' if DB_URL else 'なし'}")
print(f"[起動時] OpenAI API: {'設定済み' if OPENAI_API_KEY else '未設定'}")

# ========================================
# データベース接続プールの作成
# ========================================
try:
    # SSL設定の追加
    db_url = DB_URL
    if "pooler.supabase.com" in db_url:
        print("[DB接続] Supabase Pooler接続を使用")
        if ":5432" in db_url:
            print("[DB接続] Session Pooler (ポート5432)")
        elif ":6543" in db_url:
            print("[DB接続] Transaction Pooler (ポート6543)")
        
        if "sslmode=" not in db_url:
            if "?" in db_url:
                db_url += "&sslmode=require"
            else:
                db_url += "?sslmode=require"
    
    # 🔥 接続プールの作成
    pg_pool = psycopg2.pool.SimpleConnectionPool(
        1,   # 最小接続数
        10,  # 最大接続数
        db_url,
        cursor_factory=RealDictCursor,
        keepalives=1,
        keepalives_idle=30,
        keepalives_interval=10,
        keepalives_count=5,
        connect_timeout=10
    )
    
    if pg_pool:
        print("✅ [DB接続プール] 作成成功")
        
        # 接続テスト
        test_conn = pg_pool.getconn()
        test_conn.autocommit = True
        
        with test_conn.cursor() as cur:
            cur.execute("SELECT 1")
            cur.execute("""
                SELECT table_name 
                FROM information_schema.tables 
                WHERE table_schema = 'public' 
                LIMIT 5;
            """)
            tables = cur.fetchall()
            print(f"[DB情報] 検出されたテーブル: {[t['table_name'] for t in tables]}")
        
        pg_pool.putconn(test_conn)
        
except Exception as e:
    print(f"❌ [DB接続プール] 作成失敗: {e}")
    exit(1)

# ========================================
# 接続プール管理関数
# ========================================
def get_db_connection():
    """プールから接続を取得"""
    try:
        conn = pg_pool.getconn()
        if conn:
            # 🔥 必ずautocommitを有効化
            conn.autocommit = True
            
            # 🔥 接続テスト
            try:
                with conn.cursor() as cur:
                    cur.execute("SELECT 1")
                return conn
            except psycopg2.OperationalError:
                # 接続が死んでいる場合
                print("⚠️ [DB接続] 死んだ接続を検出、破棄します")
                try:
                    pg_pool.putconn(conn, close=True)  # 接続を破棄
                except:
                    pass
                # 再取得
                conn = pg_pool.getconn()
                conn.autocommit = True
                return conn
                
    except Exception as e:
        print(f"❌ [DB接続] 取得失敗: {e}")
        return None

def release_db_connection(conn):
    """接続をプールに戻す"""
    if not conn:
        return
    
    try:
        # 未コミットのトランザクションをクリーンアップ
        if not conn.closed:
            try:
                if not conn.autocommit:
                    conn.rollback()
            except:
                pass
        
        # プールに戻す
        pg_pool.putconn(conn)
        
    except Exception as e:
        print(f"⚠️ [DB接続] 解放エラー: {e}")

# アプリケーション終了時にプールをクローズ
@atexit.register
def cleanup_pool():
    """アプリケーション終了時にプールをクローズ"""
    try:
        if pg_pool:
            pg_pool.closeall()
            print("✅ [DB接続プール] クローズ完了")
    except:
        pass

# ========================================
# グローバル変数
# ========================================
active_session = {}
conversation_history = defaultdict(lambda: deque(maxlen=10))
latest_health = "Normal"
proactive_message_counts = defaultdict(int)

class CONFIG:
    PROFILE_ID = 1  # デフォルト値

# ========================================
# FastAPIアプリ初期化
# ========================================
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["*"]  
)

async def get_profile_async(profile_id: int):
    """非同期プロファイル取得"""
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, get_profile_sync, profile_id)

def get_profile_sync(profile_id: int):
    """同期的にプロファイルを取得"""
    conn = None
    try:
        conn = get_db_connection()
        if conn is None:
            raise HTTPException(503, "Database connection not available")
        with conn.cursor() as cur:
            cur.execute("SELECT * FROM profiles WHERE id = %s;", (profile_id,))
            profile = cur.fetchone()
            if not profile:
                raise HTTPException(404, "Profile not found")
            return profile
    finally:
        if conn:
            release_db_connection(conn)

def save_conversation_to_db(
    profile_id: int,
    speaker: str,  # 'medaka' または 実際のアカウント名
    message: str,
    health_status: str = None,
    development_stage: str = None,
    similar_example_used: bool = False,
    similar_example_text: str = None,
    similarity_score: float = None
):
    """会話履歴をデータベースに保存"""
    conn = None
    try:
        conn = get_db_connection()
        if conn is None:
            print("[会話履歴DB] 保存エラー: DB接続なし")
            return None
            
        with conn.cursor() as cur:
            cur.execute("""
                INSERT INTO conversation_history (
                    profile_id, speaker, message, health_status, development_stage,
                    similar_example_used, similar_example_text, similarity_score
                ) VALUES (
                    %s, %s, %s, %s, %s, %s, %s, %s
                ) RETURNING id;
            """, (
                profile_id,
                speaker,
                message,
                health_status,
                development_stage,
                similar_example_used,
                similar_example_text,
                similarity_score
            ))
            
            history_id = cur.fetchone()['id']
            print(f"[会話履歴DB] 保存完了 ID: {history_id} ({speaker}: {message[:30]}...)")
            return history_id
            
    except Exception as e:
        print(f"[会話履歴DB] 保存エラー: {e}")
        return None
    finally:
        if conn:
            release_db_connection(conn)
    
@app.get("/best.onnx")
async def serve_onnx_model():
    """ブラウザ検出用のONNXモデルを配信"""
    model_path = "best.onnx"
    if not os.path.exists(model_path):
        raise HTTPException(404, f"Model file not found: {model_path}")
    
    # ファイルを読み込み
    with open(model_path, "rb") as f:
        content = f.read()
    
    # Responseで直接返す（CORSヘッダー完全制御）
    return Response(
        content=content,
        media_type="application/octet-stream",
        headers={
            "Access-Control-Allow-Origin": "*",
            "Access-Control-Allow-Methods": "GET, OPTIONS",
            "Access-Control-Allow-Headers": "*",
            "Access-Control-Expose-Headers": "*",
            "Cache-Control": "public, max-age=31536000",
            "Content-Type": "application/octet-stream",
            "Content-Length": str(len(content))
        }
    )
@app.options("/best.onnx")
async def options_onnx_model():
    return Response(
        headers={
            "Access-Control-Allow-Origin": "*",
            "Access-Control-Allow-Methods": "GET, OPTIONS",
            "Access-Control-Allow-Headers": "*"
        }
    )

@app.post("/transcribe_audio")
async def transcribe_audio(file: UploadFile):
    """GPT-4o-mini-transcribeで音声をテキストに変換（高速版）"""
    start = time.time()
    audio_content = await file.read()
    with tempfile.NamedTemporaryFile(delete=False, suffix=".webm") as temp_audio:
            temp_audio.write(audio_content)
            temp_audio_path = temp_audio.name
    with open(temp_audio_path, "rb") as audio_file:
            transcript = await openai_client.audio.transcriptions.create(
                model="gpt-4o-mini-transcribe",
                file=audio_file,
                language="ja",
                response_format="text"  # 🆕 textに変更（より高速）
            )
    os.unlink(temp_audio_path)
        # textの場合、transcriptは文字列で返ってくる
    return {
            "text": transcript,  # 直接文字列
            "duration": None,
            "language": "ja"
        }

# assess_child_expression_level 関数を追加
async def assess_child_expression_level(child_input: str, current_stage: str) -> dict:
    """
    児童の発話から自己表現レベルを判定（LLM使用）
    
    Returns:
        {
            'detected_stage': 'stage_1' | 'stage_2' | 'stage_3',
            'confidence': 0.0-1.0,
            'reasoning': '判定理由',
            'should_upgrade': True | False
        }
    """
    # 発達段階の定義
    stage_definitions = {
        'stage_1': """
【Stage 1: 単語・最小限の応答レベル】
＜発話の特徴＞
-応答が 単語やごく短い文のみ
-会話を 自発的に始めることができない
-相手に話しかけられても、返答できるのは限られた場面だけ
-言葉が出ないことやオウム返しが多い
-話を広げたり質問を返したりは難しい
""",
        'stage_2': """
【Stage 2: 短文・断片的な応答レベル】
-話題を広げる・興味を共有することが難しい
-応答が短い、曖昧な返事が多い
-やり取りのテンポが遅れる・ズレることがある
-「メダカ5匹いる」「速いね、新幹線みたい」など短文で返せる
-「かな」「たぶん」など曖昧な返事でやり取りが止まる
""",
        'stage_3': """
【Stage 3: 文章・一方的な説明レベル】
-会話自体は成立するが、一方的になりやすい
-相手の発言に応答できず、キャッチボールが途切れることがある
-会話の順番が守れない／相手の気持ちを汲めない
-友達関係を築くのが難しい
-論理的で長い説明をするが、相手の興味に合わない
-相手の返答を拾わず、自分の話を続けてしまう
-表面上は会話できているが、噛み合わないことが多い
"""
    }
    
    prompt = f"""
あなたは児童の言語発達の専門家です。以下の発話を分析し、自己表現レベルを判定してください。
【発達段階の定義】
{stage_definitions['stage_1']}
{stage_definitions['stage_2']}
{stage_definitions['stage_3']}
【現在の登録段階】
{current_stage}
【児童の発話】
「{child_input}」
【判定手順】
1. 発話の内容・意図を確認（何を伝えようとしているか）
2. 会話的な要素を確認（興味共有・質問・同意など）
3. 発話の長さを確認（単語数・文の数）
4. 自発性を確認

**重要**: 文法の正確さより、コミュニケーション意図を優先してください。
- 「今日の天気めっちゃいいね」→ 興味共有あり → stage_2
- 「うん」「そう」→ 最小限応答 → stage_1


以下のJSON形式のみを出力してください。

{{
  "detected_stage": "stage_1",
  "confidence": 0.85,
  "reasoning": "単語のみで文構造がないため",
  "word_count": 3
}}

**JSONのみを出力してください。**
"""
    
    try:
        # Gemini呼び出しを削除し、OpenAI APIに置き換え
        response = await openai_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "あなたは児童の言語発達の専門家です。指示に従ってJSON形式のみを出力してください。"},
                {"role": "user", "content": prompt}
            ],
            temperature=0.3,
            max_tokens=200
        )
        # JSONをパース
        import json
        response_text = response.choices[0].message.content.strip()
        response_text = response_text.replace('```json\n', '').replace('```\n', '').replace('```', '').strip()
        result = json.loads(response_text)
        
        # 🔥 1段階昇格のみ許可（飛び級なし）
        stage_order = {'stage_1': 1, 'stage_2': 2, 'stage_3': 3}
        current_level = stage_order.get(current_stage, 1)
        detected_level = stage_order.get(result['detected_stage'], 1)
        
        # 昇格判定（1段階のみ）
        if detected_level == current_level + 1:
            result['should_upgrade'] = True
            print(f"✅ [発話レベル判定] 1段階昇格を推奨: {current_stage} → {result['detected_stage']}")
        elif detected_level > current_level + 1:
            # 飛び級は許可しない
            result['should_upgrade'] = False
            print(f"⚠️ [発話レベル判定] 飛び級は不可: {current_stage} → {result['detected_stage']}")
        else:
            result['should_upgrade'] = False
            print(f"[発話レベル判定] 昇格なし: 検出={result['detected_stage']}, 現在={current_stage}")
        
        print(f"[発話レベル判定] 信頼度: {result['confidence']:.2f}")
        print(f"[発話レベル判定] 理由: {result['reasoning']}")
        return result
        
    except json.JSONDecodeError as e:
        print(f"⚠️ [発話レベル判定] JSON解析エラー: {e}")
        return {
            'detected_stage': current_stage,
            'confidence': 0.0,
            'reasoning': 'JSON解析エラー',
            'should_upgrade': False
        }
    except Exception as e:
        print(f"❌ [発話レベル判定] エラー: {e}")
        return {
            'detected_stage': current_stage,
            'confidence': 0.0,
            'reasoning': 'エラー発生',
            'should_upgrade': False
        }

@app.post("/talk_with_fish_text")
async def talk_with_fish_text(file: UploadFile):
    start_total = time.time()
    time_log = {}
    
    # ⏱️ 1. 音声認識（先に実行）
    t1 = time.time()
    transcription_result = await transcribe_audio(file)
    user_input = transcription_result["text"]
    t2 = time.time()
    time_log['01_音声認識'] = t2 - t1
    print(f"[⏱️ 音声認識] {time_log['01_音声認識']:.2f}秒")
    
    # ⏱️ 2. プロファイル取得
    t1 = time.time()
    profile = await get_profile_async(CONFIG.PROFILE_ID)
    current_stage = profile["development_stage"]
    child_name = profile["name"]
    t2 = time.time()
    time_log['02_プロファイル取得'] = t2 - t1
    print(f"[⏱️ プロファイル取得] {time_log['02_プロファイル取得']:.2f}秒")
    
    print(f"児童の発話:{user_input}")
    save_conversation_to_db(
        profile_id=CONFIG.PROFILE_ID,
        speaker=child_name,  # 🔥 'child'ではなく実際の名前
        message=user_input,
        health_status=latest_health,
        development_stage=current_stage
    )
    # ⏱️ 2. 会話履歴の初期化
    t1 = time.time()
    if CONFIG.PROFILE_ID not in conversation_history:
        conversation_history[CONFIG.PROFILE_ID] = []
    current_history = conversation_history[CONFIG.PROFILE_ID]
    session = active_session.get(CONFIG.PROFILE_ID)
    t2 = time.time()
    time_log['02_履歴初期化'] = t2 - t1
    print(f"[⏱️ 履歴初期化] {time_log['02_履歴初期化']:.2f}秒")
    
    assessment_result = None  
    similar_example = None
    expression_assessment = None
    use_similar_example = False 

    if session is None:
        print("[会話フロー] 1回目の会話 - 類似例を検索")
        
        # ⏱️ 3. ベクトル検索
        t1 = time.time()
        similar_example = await find_similar_conversation(user_input, current_stage)
        t2 = time.time()
        time_log['03_ベクトル検索'] = t2 - t1
        print(f"[⏱️ ベクトル検索] {time_log['03_ベクトル検索']:.2f}秒")
        
        # 🔥 類似度の閾値判定（統一された基準）
        SIMILARITY_THRESHOLD = 0.88  # この値より小さい = 類似度が高い
        
        if similar_example is None:
            print("[会話フロー] 類似例なし - 発話レベル判定を実行")
            use_similar_example = False
        else:
            print(f"[会話フロー] 類似度が高い ({similar_example['distance']:.4f} < {SIMILARITY_THRESHOLD}) - 類似例を使用")
            use_similar_example = True
        
        if not use_similar_example:
            print("[会話フロー] 発話レベル判定+応答生成を並列実行")
            t1 = time.time()
        
            expression_assessment, reply_text = await asyncio.gather(
                assess_child_expression_level(user_input, current_stage),
                get_medaka_reply(
                    user_input, 
                    latest_health, 
                    current_history, 
                    None,
                    profile
                )
            )
            
            t2 = time.time()
            time_log['03_発話レベル判定+応答生成'] = t2 - t1
            print(f"[⏱️ 発話レベル判定+応答生成（並列）] {time_log['03_発話レベル判定+応答生成']:.2f}秒")
            save_conversation_to_db(
                profile_id=CONFIG.PROFILE_ID,
                speaker='medaka',  # メダカは固定
                message=reply_text,
                health_status=latest_health,
                development_stage=current_stage,
                similar_example_used=False,
                similar_example_text=None,
                similarity_score=None
            )
            # 🔥 昇格判定（信頼度0.7以上 かつ 1段階昇格推奨）
            if expression_assessment['should_upgrade'] and expression_assessment['confidence'] >= 0.7:
                t3 = time.time()
                upgrade_result = await upgrade_by_expression_assessment_async(
                    CONFIG.PROFILE_ID,
                    current_stage,
                    expression_assessment['reasoning']
                )
                t4 = time.time()
                time_log['03_段階更新'] = t4 - t3
                print(f"[⏱️ 段階更新] {time_log['03_段階更新']:.2f}秒")
                
                if upgrade_result['success']:
                    profile['development_stage'] = upgrade_result['new_stage']
                    current_stage = upgrade_result['new_stage']
                    print(f"✅ [段階変更] {upgrade_result['old_stage']} → {upgrade_result['new_stage']}")
            else:
                if expression_assessment.get('confidence', 0) < 0.7:
                    print(f"[段階変更] スキップ - 信頼度不足 ({expression_assessment.get('confidence', 0):.2f})")
                else:
                    print(f"[段階変更] スキップ - 昇格条件を満たさない")
        
        else:
            # 🔥 類似例を使う場合（既存の処理）
            print("[会話フロー] 類似例を使用した応答生成")
            t1 = time.time()
            reply_text = await get_medaka_reply(
                user_input, 
                latest_health, 
                current_history, 
                similar_example,  # 類似例を渡す
                profile
            )
            t2 = time.time()
            time_log['04_応答生成'] = t2 - t1
            print(f"[⏱️ 応答生成] {time_log['04_応答生成']:.2f}秒")
            save_conversation_to_db(
                profile_id=CONFIG.PROFILE_ID,
                speaker='medaka',  # メダカは固定
                message=reply_text,
                health_status=latest_health,
                development_stage=current_stage,
                similar_example_used=True,
                similar_example_text=similar_example['text'],
                similarity_score=similar_example['distance']
            )
        # ⏱️ 5. セッション作成
        t1 = time.time()
        is_max_stage = current_stage == "stage_3"
        
        if is_max_stage:
            print(f"[セッション] 最高段階 {current_stage} - 判定スキップ")
        elif (use_similar_example and
            similar_example and 
            'child_reply_1_embedding' in similar_example and 
            'child_reply_2_embedding' in similar_example and 
            similar_example.get('child_reply_2_embedding') is not None):
            
            session = ConversationSession(
                profile_id=CONFIG.PROFILE_ID,
                child_name=child_name,  # 🔥 追加
                first_input=user_input,
                medaka_response=reply_text,
                similar_example=similar_example,
                current_stage=current_stage
            )
            active_session[CONFIG.PROFILE_ID] = session
            print(f"[セッション] セッション作成完了 - 次回判定実行予定（類似度: {similar_example['distance']:.4f}）")
        else:
            print(f"[セッション] 通常の会話として処理")
        
        t2 = time.time()
        time_log['05_セッション作成'] = t2 - t1
        print(f"[⏱️ セッション作成] {time_log['05_セッション作成']:.2f}秒")   
    else:
        # 2回目の会話（既存のコードと同じ）
        print("[会話フロー] 2回目の会話 - 発達段階判定を実行")
        
        # ⏱️ 3. 発達段階判定
        t1 = time.time()
        assessment = await classify_child_response(
            user_input,
            session.similar_example,
            openai_client,
        )
        t2 = time.time()
        time_log['03_発達段階判定'] = t2 - t1
        print(f"[⏱️ 発達段階判定] {time_log['03_発達段階判定']:.2f}秒")
        
        # ⏱️ 4. 判定結果処理
        t1 = time.time()
        assessment_result = {
            'result': assessment[0],
            'maintain_score': round(float(assessment[1]), 3),
            'upgrade_score': round(float(assessment[2]), 3),
            'confidence_score': round(float(abs(assessment[2] - assessment[1])), 5),
            'assessed_at': datetime.now(),
        }
        
        if assessment[0] == "昇格":
            new_stage = upgrade_development_stage(CONFIG.PROFILE_ID, current_stage)
            profile["development_stage"] = new_stage
            
            if new_stage != current_stage:
                assessment_result['stage_upgraded'] = True
                assessment_result['previous_stage'] = current_stage
                assessment_result['new_stage'] = new_stage
                print(f"[会話フロー] 🎉 発達段階が昇格しました！ {current_stage} → {new_stage}")
            else:
                assessment_result['stage_upgraded'] = False
                assessment_result['already_max'] = True
                print(f"[会話フロー] すでに最高段階 {current_stage} に到達しています")
        else:
            assessment_result['stage_upgraded'] = False
            print(f"[会話フロー] 現状維持 - {current_stage} のまま")
        t2 = time.time()
        time_log['04_判定結果処理'] = t2 - t1
        print(f"[⏱️ 判定結果処理] {time_log['04_判定結果処理']:.2f}秒")
        
        # ⏱️ 5. メダカ応答生成
        t1 = time.time()
        reply_text = await get_medaka_reply(user_input, latest_health, current_history, None, profile)
        t2 = time.time()
        time_log['05_応答生成'] = t2 - t1
        print(f"[⏱️ 応答生成] {time_log['05_応答生成']:.2f}秒")
        save_conversation_to_db(
            profile_id=CONFIG.PROFILE_ID,
            speaker='medaka',  # メダカは固定
            message=reply_text,
            health_status=latest_health,
            development_stage=current_stage,
            similar_example_used=False
        )
        # ⏱️ 6. セッション完了処理
        t1 = time.time()
        del active_session[CONFIG.PROFILE_ID]
        t2 = time.time()
        time_log['06_セッション完了'] = t2 - t1
        print(f"[⏱️ セッション完了] {time_log['06_セッション完了']:.2f}秒")

    # ⏱️ 7. 会話履歴保存
    t1 = time.time()
    conversation_entry = {
            "child": user_input,
            "medaka": reply_text,
            "timestamp": datetime.now(),
            "similar_example_used": similar_example['text'] if similar_example else None,
            "similarity_score": similar_example['distance'] if similar_example else None,
            "has_assessment": assessment_result is not None,
            "assessment_result": assessment_result,
            "session_status": "started" if session and CONFIG.PROFILE_ID in active_session else "completed"
    }
    conversation_history[CONFIG.PROFILE_ID].append(conversation_entry)
    if len(conversation_history[CONFIG.PROFILE_ID]) > 20:
        conversation_history[CONFIG.PROFILE_ID] = conversation_history[CONFIG.PROFILE_ID][-20:]

    print(f"[会話履歴] 現在の履歴件数: {len(conversation_history[CONFIG.PROFILE_ID])}")
    t2 = time.time()
    time_log['07_履歴保存'] = t2 - t1
    print(f"[⏱️ 履歴保存] {time_log['07_履歴保存']:.2f}秒")
    
    # ⏱️ 8. TTS準備（ストリーミング開始まで）
    t_stream_start = time.time()
    
    async def audio_stream():
        chunk_count = 0
        t_first_chunk = None
        
        async with openai_client.audio.speech.with_streaming_response.create(
            model="tts-1",
            voice="nova",
            instructions="Voice Affect:かわいらしい, Tone:高い, Pacing:ゆっくり",
            speed=1.0,
            input=reply_text,
            response_format="mp3",
        ) as response:
            async for chunk in response.iter_bytes():
                chunk_count += 1
                if chunk_count == 1:
                    t_first_chunk = time.time()
                    first_chunk_time = t_first_chunk - t_stream_start
                    print(f"[⏱️ TTS最初のチャンク] {first_chunk_time:.2f}秒")
                yield chunk
    
    # ⏱️ 総処理時間の計算と表示
    end_total = time.time()
    total_time = end_total - start_total
    
    print("\n" + "="*50)
    print("⏱️  処理時間の詳細")
    print("="*50)
    
    for key in sorted(time_log.keys()):
        duration = time_log[key]
        percentage = (duration / total_time) * 100
        bar_length = int(percentage / 2)
        bar = "█" * bar_length + "░" * (50 - bar_length)
        print(f"{key:20} │ {bar} │ {duration:6.2f}秒 ({percentage:5.1f}%)")
    
    print("-" * 50)
    print(f"{'合計（ストリーミング開始まで）':20} │ {total_time:6.2f}秒 (100.0%)")
    print("="*50 + "\n")
    
    return StreamingResponse(
            audio_stream(),
            media_type="audio/mpeg",
            headers={"Content-Disposition": "inline; filename=reply.mp3"}
        )

async def generate_tts(text: str) -> str:
    """TTS生成（非同期関数）"""
    async with openai_client.audio.speech.with_streaming_response.create(
        model="tts-1",
        voice="nova",
        instructions="Voice Affect:かわいらしい, Tone:高い, Pacing:ゆっくり",
        speed=1.0,
        input=text,
        response_format="mp3",
    ) as response:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as tts_file:
            async for chunk in response.iter_bytes():
                tts_file.write(chunk)
            return tts_file.name

async def find_similar_conversation(user_input: str, development_stage: str, similarity_threshold: float = 0.88):
    resp = await openai_client.embeddings.create(
        input=[user_input],
        model="text-embedding-3-small"
    )
    query_vector = resp.data[0].embedding
    
    conn = None
    try:
        conn = get_db_connection()
        if conn is None:
            print("[類似会話] DB接続なし")
            return None

        with conn.cursor() as cur:
            cur.execute("""
                SELECT text, fish_text, children_reply_1, children_reply_2,
                       child_reply_1_embedding, child_reply_2_embedding,
                       user_embedding <-> %s::vector as distance
                FROM conversations
                WHERE development_stage = %s
                ORDER BY distance
                LIMIT 1;
            """, (query_vector, development_stage))
            
            result = cur.fetchone()
            if not result:
                print("[類似会話] 類似例は見つかりませんでした")
                return None

            print(f"[類似検索] 見つかった例: '{result['text']}'")
            print(f"[類似検索] 類似度スコア: {result['distance']:.4f}")
            if result['distance'] < similarity_threshold:
                print("[類似会話] 類似例が見つかりました:", result['text'] )
                return result
            else:
                print("[類似会話] 類似例は見つかりませんでした")
                return None
    except Exception as e:
        print(f"❌ [類似検索] エラー: {e}")
        return None
    finally:
        if conn:
            release_db_connection(conn)
        
async def get_medaka_reply(user_input, health_status="不明", conversation_hist=None, similar_example=None, profile_info=None):
    start = time.time()
    
    if health_status == "Active":
        medaka_state = "元気"
    elif health_status == "Normal":
        medaka_state = "休憩中"
    elif health_status == "Lethargic":
        medaka_state = "元気ない"
    else:
        medaka_state = "休憩中"
    
    print("メダカの状態:", medaka_state)
    
    # プロファイル情報の取得
    if profile_info:
        profile_name = profile_info.get('name', 'Unknown')
        age_text = f"{profile_info['age']}歳" if profile_info.get('age') else "年齢不明"
        stage_text = profile_info.get('development_stage', '不明')
        profile_context = f"話し相手: {profile_name}さん ({age_text}, {stage_text})\n"
        
        # 会話履歴
        history_context = ""
        if conversation_hist and len(conversation_hist) > 0:
            recent_history = conversation_hist[-3:]  # 最新3件
            history_context = "最近の会話履歴:\n"
            for i, h in enumerate(recent_history, 1):
                # 🔥 child が None の場合（プロアクティブメッセージ）はスキップ
                if h['child'] is None:
                    history_context += f"{i}. メダカ「{h['medaka']}」\n"
                else:
                    history_context += f"{i}. 児童「{h['child']}」→ メダカ「{h['medaka']}」\n"
        history_context += "\n"
        
        # 🆕 自己表現レベルの取得
        stage = profile_info.get('development_stage', 'stage_1')
        
        # stage から数値を抽出
        if stage == 'stage_1':
            child_expression_level = 1
        elif stage == 'stage_2':
            child_expression_level = 2
        elif stage == 'stage_3':
            child_expression_level = 3
        else:
            child_expression_level = 1  # デフォルト
    else:
        profile_context = ""
        history_context = ""
        child_expression_level = 1  # デフォルト
    
    # 🆕 自己表現レベルに応じた応答戦略
    if child_expression_level == 1:
        response_strategy = """
【応答戦略】
児童の発話が「抽象的」か「具体的」かを判断し、使い分けてください。
**発話が抽象的な場合**: 必ず2択や「どっち？」で答えを引き出すか、児童の単語に追加の言葉をつけて誘導する。
**発話が具体的な場合**: 児童の単語を短文に直して返す。または、発話をそのまま肯定しつつ、感情表現や語彙を少し増やす（例：「きれい」→「きれいだね〜！ピカピカしててうれしいね」）。
"""
    elif child_expression_level == 2:
        response_strategy = """
【応答戦略】
児童の発話タイプに合わせて対応を変えてください。
- **単語や短いフレーズどまり**: 短い返答を繰り返しながら、「どうして？」「どんな？」「他には？」と質問を足す。または興味に沿って「もっと詳しく教えて」と掘り下げる。
- **話が単発的で順序がない**: 「まずは？」「次は？」など、理由づけや順序立てを促す。
- **語彙や文法が不自然で、文脈がズレている**: 少しズレた説明や一方的な話でも否定せずに聞き役になる。
"""
    else:
        # stage_3 またはデフォルト
        response_strategy = ""
    
    # プロンプトの構築
    if similar_example:
        prompt = f"""
あなたは水槽に住むかわいいメダカ「シロちゃん」です。
メダカの状態: {medaka_state}
{profile_context}
以下の例と全く同じ言葉で30字程度で応答してください。
【会話】
児童:「{similar_example['text']}」
メダカ:「{similar_example['fish_text']}」

{history_context}【現在の会話】
児童:「{user_input}」
メダカ:
"""
    else:
        # 類似例がない場合、戦略を組み込んだプロンプトを使用
        prompt = f"""
あなたは水槽に住むかわいいメダカ「シロちゃん」です。応答は「」や名前を含めず、セリフのみを出力してください。
{profile_context}

{response_strategy}

{history_context}児童:「{user_input}」

上記の【応答戦略】に基づき、30文字以内で、優しく小学生らしい口調で答えてください。
メダカの状態: {medaka_state}

キンちゃん:"""
    
    print(f"[応答生成] プロンプト作成完了\n{prompt}")

    
    response = await openai_client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "あなたは水槽に住むかわいいメダカ「シロちゃん」です。応答は「」や名前を含めず、セリフのみを出力してください。"},
            {"role": "user", "content": prompt}
        ],
        temperature=1.0,
        max_tokens=100
    )
    end = time.time()
    reply = response.choices[0].message.content.strip()
    
    print(f"[メダカ応答生成] 所要時間: {end - start:.2f}秒")
    print(f"[応答生成] 生成された応答: '{reply}'")
    
    return reply

class ConversationSession:
    def __init__(self, profile_id: int, child_name: str, first_input: str, medaka_response: str, similar_example: dict, current_stage: str):
        self.profile_id = profile_id
        self.child_name = child_name  # 🔥 追加: 児童の名前
        self.first_child_input = first_input
        self.medaka_response = medaka_response
        self.similar_example = similar_example
        self.current_stage = current_stage
        self.started_at = datetime.now()

    def complete_session(self, second_input: str, assessment_result: tuple):
        """セッションを完了"""
        self.second_child_input = second_input
        self.assessment_result = assessment_result[0]
        self.maintain_score = round(float(assessment_result[1]), 3)
        self.upgrade_score = round(float(assessment_result[2]), 3)
        self.confidence_score = round(float(abs(self.upgrade_score - self.maintain_score)), 5)
        
        # 🔥 conversation_historyには保存しない（既に保存済み）
        # セッション情報だけログ出力
        print(f"[セッション完了] プロファイルID: {self.profile_id}")
        print(f"[セッション完了] 判定結果: {self.assessment_result} (信頼度: {self.confidence_score:.3f})")
        print(f"[セッション完了] 現状維持スコア: {self.maintain_score}, 昇格スコア: {self.upgrade_score}")
        
        return None  # DBには保存しない

STAGE_PROGRESSION = {
    "stage_1": "stage_2",
    "stage_2": "stage_3",
    "stage_3": "stage_3"
}

# 会話分類
async def classify_child_response(
        child_response: str,
        similar_conversation: dict,
        openai_client,
        threshold: float = 0.88
) -> tuple[str, float, float]:
    print(f"[発達段階判定] 児童の応答: '{child_response}'")
    
    resp = await openai_client.embeddings.create(
        input=[child_response],
        model="text-embedding-3-small"
    )
    response_vector = np.array(resp.data[0].embedding)
    
    def convert_to_vector(embedding_data):
        """データベースからの埋め込みデータを数値ベクトルに変換"""
        if isinstance(embedding_data, str):
            import json
            return np.array(json.loads(embedding_data), dtype=float)
            
    maintain_vector = convert_to_vector(similar_conversation['child_reply_1_embedding'])
    upgrade_vector = convert_to_vector(similar_conversation['child_reply_2_embedding'])
    
    def cosine_similarity(v1, v2):
        """コサイン類似度を計算"""
        if len(v1) != len(v2):
            raise ValueError(f"ベクトル次元が一致しません: {len(v1)} vs {len(v2)}")
        
        norm1 = np.linalg.norm(v1)
        norm2 = np.linalg.norm(v2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        return np.dot(v1, v2) / (norm1 * norm2)
    
    maintain_similarity = cosine_similarity(response_vector, maintain_vector)
    upgrade_similarity = cosine_similarity(response_vector, upgrade_vector)
    
    print(f"[発達段階判定] 現状維持との類似度: {maintain_similarity:.4f}")
    print(f"[発達段階判定] 昇格との類似度: {upgrade_similarity:.4f}")
    
    if upgrade_similarity > maintain_similarity and upgrade_similarity > threshold:
        result = "昇格"
    else:
        result = "現状維持"
    
    confidence = abs(upgrade_similarity - maintain_similarity)
    print(f"[発達段階判定] 結果: {result} (信頼度: {confidence:.4f})")
    
    return result, maintain_similarity, upgrade_similarity

async def upgrade_by_expression_assessment_async(profile_id: int, current_stage: str, reasoning: str = "") -> dict:
    """
    発話レベル判定による段階昇格（非同期版）
    """
    # 次の段階を取得
    next_stage = STAGE_PROGRESSION.get(current_stage, current_stage)
    
    # すでに最高段階
    if next_stage == current_stage:
        print(f"[発話昇格] すでに最高段階: {current_stage}")
        return {
            'success': False,
            'old_stage': current_stage,
            'new_stage': current_stage,
            'reasoning': '既に最高段階'
        }
    
    # 🔥 同期処理を非同期で実行
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, _upgrade_stage_sync, profile_id, current_stage, next_stage, reasoning)

def _upgrade_stage_sync(profile_id: int, current_stage: str, next_stage: str, reasoning: str) -> dict:
    """同期的なDB更新処理"""
    conn = None
    try:
        conn = get_db_connection()
        if conn is None:
            print(f"❌ [発話昇格] エラー: DB接続なし")
            return {
                'success': False, 'old_stage': current_stage, 'new_stage': current_stage,
                'reasoning': 'DB接続エラー'
            }

        with conn.cursor() as cur:
            cur.execute("""
                UPDATE profiles 
                SET development_stage = %s,
                    updated_at = NOW()
                WHERE id = %s
                RETURNING development_stage;
            """, (next_stage, profile_id))
            
            result = cur.fetchone()
            
            if result:
                print(f"🎉 [発話昇格] 成功: {current_stage} → {next_stage}")
                if reasoning:
                    print(f"   理由: {reasoning}")
                
                return {
                    'success': True,
                    'old_stage': current_stage,
                    'new_stage': next_stage,
                    'reasoning': reasoning
                }
            else:
                print(f"⚠️ [発話昇格] プロファイルが見つかりません")
                return {
                    'success': False,
                    'old_stage': current_stage,
                    'new_stage': current_stage,
                    'reasoning': 'プロファイル未検出'
                }
                
    except Exception as e:
        print(f"❌ [発話昇格] エラー: {e}")
        return {
            'success': False,
            'old_stage': current_stage,
            'new_stage': current_stage,
            'reasoning': f'エラー: {str(e)}'
        }
    finally:
        if conn:
            release_db_connection(conn)

#"""発達段階を1つ上げる"""
def upgrade_development_stage(profile_id: int, current_stage: str) -> str:
    next_stage = STAGE_PROGRESSION.get(current_stage, current_stage)
    
    if next_stage == current_stage:
        print(f"[発達段階] すでに最高段階: {current_stage}")
        return current_stage
    
    conn = None
    try:
        conn = get_db_connection()
        if conn is None:
            print(f"[発達段階] 更新エラー: DB接続なし")
            return current_stage

        with conn.cursor() as cur:
            cur.execute("""
                UPDATE profiles 
                SET development_stage = %s,
                    updated_at = NOW()
                WHERE id = %s
                RETURNING development_stage;
            """, (next_stage, profile_id))
            
            result = cur.fetchone()
            
            if result:
                print(f"[発達段階] 昇格成功: {current_stage} → {next_stage} (Profile ID: {profile_id})")
                return next_stage
            else:
                print(f"[発達段階] プロファイルが見つかりません: Profile ID {profile_id}")
                return current_stage
                
    except Exception as e:
        print(f"[発達段階] 更新エラー: {e}")
        return current_stage
    finally:
        if conn:
            release_db_connection(conn)

# ✅ ブラウザから元気度を受信するエンドポイント
@app.post("/update_health")
async def update_health(request: Request):
    """ブラウザから送信された元気度を更新"""
    global latest_health
    
    data = await request.json()
    status = data.get("status", "Unknown")
    avg_speed = data.get("avg_speed", 0)
    score = data.get("score", 0)
    
    latest_health = status
    
    print(f"[元気度更新] {status} (速度: {avg_speed:.2f}px/s, スコア: {score})")
    
    return {
        "status": "success",
        "current_health": latest_health
    }

@app.get("/")
async def read_index():
    return FileResponse('index.html', media_type='text/html')

@app.post("/set_current_profile")
async def set_current_profile(request: Request):
    """現在のプロファイルIDを設定"""
    data = await request.json()
    profile_id = data.get("profile_id")
    
    if not profile_id:
        raise HTTPException(400, "profile_id is required")
    
    # 🔥 変更前の値をログ出力
    old_id = CONFIG.PROFILE_ID
    CONFIG.PROFILE_ID = profile_id
    
    print(f"[プロファイル変更] {old_id} → {profile_id}")
    
    # 🔥 確認のため取得してログ出力
    conn = None
    try:
        conn = get_db_connection()
        if conn:
            with conn.cursor() as cur:
                cur.execute("SELECT name, age FROM profiles WHERE id = %s;", (profile_id,))
                profile = cur.fetchone()
                if profile:
                    print(f"[プロファイル変更] 選択: {profile['name']}さん ({profile['age']}歳)")
        else:
            print("[プロファイル変更] DB接続がなく、プロファイル名を確認できませんでした。")
    except Exception as e:
        print(f"[/set_current_profile] プロファイル名の取得エラー: {e}")
    finally:
        if conn:
            release_db_connection(conn)

    return {"success": True, "current_profile_id": CONFIG.PROFILE_ID}

#プロファイルの取得
@app.get("/profiles")
async def get_profiles():
    """全プロファイル取得"""
    conn = None
    try:
        conn = get_db_connection()
        if conn is None:
            raise HTTPException(status_code=503, detail="データベースに接続できません。")
        
        with conn.cursor() as cur:
            cur.execute("SELECT id, name, age, development_stage FROM profiles ORDER BY id;")
            profiles = cur.fetchall()
            return profiles
            
    except Exception as e:
        print(f"[/profiles] エラー: {e}")
        raise HTTPException(status_code=500, detail="プロファイルの取得中にエラーが発生しました。")
        
    finally:
        if conn:
            release_db_connection(conn)

@app.post("/profiles")
async def create_profile(request: Request):
    """新規プロファイル作成"""
    data = await request.json()
    name = data.get("name")
    age = data.get("age")
    
    conn = None
    try:
        conn = get_db_connection()
        if conn is None:
            raise HTTPException(status_code=503, detail="データベースに接続できません。")

        with conn.cursor() as cur:
            cur.execute("""
                INSERT INTO profiles (name, age, development_stage, created_at, updated_at)
                VALUES (%s, %s, 'stage_1', NOW(), NOW())
                RETURNING id, name, age, development_stage;
            """, (name, age))
            new_profile = cur.fetchone()

        raise HTTPException(status_code=500, detail="プロファイルの作成中にエラーが発生しました。")
    finally:
        if conn:
            release_db_connection(conn)
    
def get_proactive_medaka_message(profile):
    """この関数が実行された回数に応じてメダカからのプロアクティブメッセージを生成"""
    profile_id = profile['id']
    call_count = proactive_message_counts[profile_id]
    messages = {
            0: [  # 初対面の児童に対する言葉
                "ぼくね、君とお話するの大好きなの〜質問してみて？",
                "ぼくね、君とお話するの大好きなの〜質問してみて？",
                "ぼくね、君とお話するの大好きなの〜質問してみて？",
                "ぼくね、君とお話するの大好きなの〜質問してみて？",
                "ぼくね、君とお話するの大好きなの〜質問してみて？"
            ],
            1: [  # 自分の感情＋相手の感情を聞く
                "ぼくね、君とお話するの大好きなの〜質問してみて？",
                "ぼくね、君とお話するの大好きなの〜質問してみて？",
                "ぼくね、君とお話するの大好きなの〜質問してみて？",
                "ぼくね、君とお話するの大好きなの〜質問してみて？",
                "ぼくね、君とお話するの大好きなの〜質問してみて？"
            ],
            2: [  # 相手の日常生活に関すること（朝に偏らない）
                "ぼくね、君とお話するの大好きなの〜質問してみて？",
                "ぼくね、君とお話するの大好きなの〜質問してみて？",
                "ぼくね、君とお話するの大好きなの〜質問してみて？",
                "ぼくね、君とお話するの大好きなの〜質問してみて？",
                "ぼくね、君とお話するの大好きなの〜質問してみて？"
            ],
            3: [  # メダカ自身が困っていることを相談
                "ぼくね、君とお話するの大好きなの〜質問してみて？",
                "ぼくね、君とお話するの大好きなの〜質問してみて？",
                "ぼくね、君とお話するの大好きなの〜質問してみて？",
                "ぼくね、君とお話するの大好きなの〜質問してみて？",
                "ぼくね、君とお話するの大好きなの〜質問してみて？"
            ],
            4: [  # 児童からの質問を受け付ける・対話を引き出す
                "ぼくね、君とお話するの大好きなの〜質問してみて？",
                "ぼくね、君とお話するの大好きなの〜質問してみて？",
                "ぼくね、君とお話するの大好きなの〜質問してみて？",
                "ぼくね、君とお話するの大好きなの〜質問してみて？",
                "ぼくね、君とお話するの大好きなの〜質問してみて？"
            ]
        }
    stage_key = min(call_count, 4)
    
    import random
    message = random.choice(messages[stage_key])

    # 次回の呼び出しのためにカウントを増やす
    proactive_message_counts[profile_id] += 1
    
    return message

@app.post("/check_session_status")
async def check_session_status(request: Request):
    data = await request.json()
    profile_id = data.get("profile_id")
    
    if not profile_id:
        raise HTTPException(400, "profile_id is required")
    
    has_active_session = profile_id in active_session
    medaka_proactive_enabled = os.getenv("MEDAKA_PROACTIVE_ENABLED", "true").lower() == "true"
    
    return {
        "has_active_session": has_active_session,
        "conversation_count": len(conversation_history.get(profile_id, [])),
        "proactive_enabled": medaka_proactive_enabled
    }

@app.post("/get_proactive_message")
async def get_proactive_message(request: Request):
    data = await request.json()
    profile_id = data.get("profile_id")
    
    if not profile_id:
        raise HTTPException(400, "profile_id is required")
    
    conn = None
    try:
        conn = get_db_connection()
        if conn is None:
            raise HTTPException(status_code=503, detail="データベースに接続できません。")
        with conn.cursor() as cur:
            cur.execute("SELECT * FROM profiles WHERE id = %s;", (profile_id,))
            profile = cur.fetchone()
            if not profile:
                raise HTTPException(404, "Profile not found")
    except Exception as e:
        print(f"[/get_proactive_message] プロファイル取得エラー: {e}")
        raise HTTPException(status_code=500, detail="プロファイルの取得中にエラーが発生しました。")
    finally:
        if conn:
            release_db_connection(conn)

    # conversation_count はメッセージ生成に不要になった
    message = get_proactive_medaka_message(profile)
    
    # 🔥 会話履歴に追加（メモリ内）
    if profile_id not in conversation_history:
        conversation_history[profile_id] = []
    
    conversation_entry = {
        "child": None,  # プロアクティブメッセージなので児童発言なし
        "medaka": message,
        "timestamp": datetime.now(),
        "similar_example_used": None,
        "similarity_score": None,
        "has_assessment": False,
        "assessment_result": None,
        "session_status": None
    }
    conversation_history[profile_id].append(conversation_entry)
    
    # DB保存
    save_conversation_to_db(
        profile_id=profile_id,
        speaker='medaka',
        message=message,
        health_status=latest_health,
        development_stage=profile['development_stage'],
        similar_example_used=False
    )
    
    # TTS生成（以下同じ）
    async with openai_client.audio.speech.with_streaming_response.create(
        model="tts-1",
        voice="nova",
        instructions="Voice Affect:かわいらしい, Tone:高い, Pacing:ゆっくり",
        speed=1.0,
        input=message,
        response_format="mp3",
    ) as response:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as tts_file:
            async for chunk in response.iter_bytes():
                tts_file.write(chunk)
            tts_path = tts_file.name
    
    return FileResponse(tts_path, media_type="audio/mpeg", filename="proactive_reply.mp3")

# デバッグ用エンドポイント
@app.get("/conversation_history")
async def get_conversation_history():
    """現在のプロファイルの会話履歴を取得"""
    if CONFIG.PROFILE_ID in conversation_history:
        return {
            "profile_id": CONFIG.PROFILE_ID,
            "history": list(conversation_history[CONFIG.PROFILE_ID])
        }
    else:
        return {"profile_id": CONFIG.PROFILE_ID, "history": []}

@app.delete("/conversation_history")
async def clear_conversation_history():
    """現在のプロファイルの会話履歴をクリア"""
    if CONFIG.PROFILE_ID in conversation_history:
        del conversation_history[CONFIG.PROFILE_ID]
    return {"message": f"History cleared for profile {CONFIG.PROFILE_ID}"}

@app.post("/test_vector_search")
async def test_vector_search(request: Request):
    """ベクトル検索テスト用エンドポイント"""
    data = await request.json()
    user_input = data.get("user_input", "")
    stage = data.get("stage", "stage_1")
    
    result = await find_similar_conversation(user_input, stage)
    return {"query": user_input, "stage": stage, "result": result}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)