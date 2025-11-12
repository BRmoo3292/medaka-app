from fastapi import FastAPI, UploadFile, HTTPException, Request
from fastapi.responses import FileResponse,Response,StreamingResponse
import uvicorn
from fastapi.middleware.cors import CORSMiddleware
from openai import AsyncOpenAI
import time
import numpy as np
import asyncio
from collections import defaultdict, deque
import tempfile
import os
import google.generativeai as genai
from datetime import datetime
import psycopg2
from psycopg2.extras import RealDictCursor

# データベースのURLを環境変数から取得
DB_URL = os.getenv("DB_URL")
pg_conn = psycopg2.connect(DB_URL, cursor_factory=RealDictCursor)
pg_conn.autocommit = True
print(f"[起動時] DB接続成功: {DB_URL}")

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
openai_client = AsyncOpenAI(api_key=OPENAI_API_KEY)
genai.configure(api_key=GEMINI_API_KEY)
model_gemini = genai.GenerativeModel(model_name="gemini-2.5-flash")

print(f"[起動時] DB_URL設定: {'あり' if DB_URL else 'なし'}")
print(f"[起動時] OpenAI API: {'設定済み' if OPENAI_API_KEY else '未設定'}")
print(f"[起動時] Gemini API: {'設定済み' if GEMINI_API_KEY else '未設定'}")

# グローバル変数
active_session = {}
conversation_history = defaultdict(lambda: deque(maxlen=10))
latest_health = "Normal"
CURRENT_PROFILE_ID = 1
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["*"]  
)

# Session Pooler対応のデータベース接続関数
def connect_to_database(db_url, max_retries=3):
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
    
    print(f"[DB接続] 接続先: {db_url.split('@')[0]}@...")
    
    for attempt in range(max_retries):
        try:
            print(f"[DB接続] 試行 {attempt + 1}/{max_retries}")
            
            conn = psycopg2.connect(
                db_url,
                cursor_factory=RealDictCursor,
                keepalives=1,
                keepalives_idle=30,
                keepalives_interval=10,
                keepalives_count=5,
                connect_timeout=10
            )
            
            conn.autocommit = True
            
            with conn.cursor() as cur:
                cur.execute("SELECT 1")
                result = cur.fetchone()
                if result:
                    print(f"✅ DB接続成功! (試行 {attempt + 1})")
                    
                    cur.execute("""
                        SELECT table_name 
                        FROM information_schema.tables 
                        WHERE table_schema = 'public' 
                        LIMIT 5;
                    """)
                    tables = cur.fetchall()
                    print(f"[DB情報] 検出されたテーブル: {[t['table_name'] for t in tables]}")
                    
                    return conn
                    
        except psycopg2.OperationalError as e:
            error_msg = str(e)
            print(f"⚠️ 接続エラー (試行 {attempt + 1}): {error_msg}")
            
            if "password authentication failed" in error_msg:
                print("[エラー] パスワードが正しくありません")
                break
            elif "Network is unreachable" in error_msg:
                print("[エラー] IPv6接続の問題です。Session Poolerを使用してください")
                break
            
            if attempt < max_retries - 1:
                wait_time = (attempt + 1) * 2
                print(f"[DB接続] {wait_time}秒待機してリトライ...")
                time.sleep(wait_time)
        except Exception as e:
            print(f"❌ 予期しないエラー: {e}")
            break
    
    return None  
# データベース接続
try:
    pg_conn = connect_to_database(DB_URL)
    if not pg_conn:
        print("❌ データベース接続を確立できませんでした")
        exit(1)
except Exception as e:
    print(f"❌ DB接続エラー: {e}")
    exit(1)

async def get_profile_async(profile_id: int):
    """非同期プロファイル取得"""
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, get_profile_sync, profile_id)

def get_profile_sync(profile_id: int):
    """同期的にプロファイルを取得"""
    with pg_conn.cursor() as cur:
        cur.execute("SELECT * FROM profiles WHERE id = %s;", (profile_id,))
        profile = cur.fetchone()
        if not profile:
            raise HTTPException(404, "Profile not found")
        return profile


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
                model="gpt-4o-transcribe",
                file=audio_file,
                language="ja",
                response_format="text"  # 🆕 textに変更（より高速）
            )

        # textの場合、transcriptは文字列で返ってくる
    return {
            "text": transcript,  # 直接文字列
            "duration": None,
            "language": "ja"
        }

@app.post("/talk_with_fish_text")
async def talk_with_fish_text(file: UploadFile):
    start_total = time.time()
    time_log = {}
    
    # ⏱️ 1. 音声認識とプロファイル取得を完全並列実行
    t1 = time.time()
    
    # 🔥 並列タスクを作成
    transcription_task = transcribe_audio(file)
    profile_task = get_profile_async(CURRENT_PROFILE_ID)
    
    # 両方の完了を待つ
    transcription_result, profile = await asyncio.gather(
        transcription_task,
        profile_task
    )
    
    user_input = transcription_result["text"]
    current_stage = profile["development_stage"]
    
    t2 = time.time()
    time_log['01_音声認識+プロファイル'] = t2 - t1
    print(f"[⏱️ 音声認識+プロファイル（並列）] {time_log['01_音声認識+プロファイル']:.2f}秒")
    
    # ⏱️ 2. 会話履歴の初期化
    t1 = time.time()
    if CURRENT_PROFILE_ID not in conversation_history:
        conversation_history[CURRENT_PROFILE_ID] = []
    current_history = conversation_history[CURRENT_PROFILE_ID]
    session = active_session.get(CURRENT_PROFILE_ID)
    t2 = time.time()
    time_log['02_履歴初期化'] = t2 - t1
    print(f"[⏱️ 履歴初期化] {time_log['02_履歴初期化']:.2f}秒")
    
    assessment_result = None  
    similar_example = None

    if session is None:
        print("[会話フロー] 1回目の会話 - 類似例を検索")
        
        # ⏱️ 3. ベクトル検索
        t1 = time.time()
        similar_example = await find_similar_conversation(user_input, current_stage)
        t2 = time.time()
        time_log['03_ベクトル検索'] = t2 - t1
        print(f"[⏱️ ベクトル検索] {time_log['03_ベクトル検索']:.2f}秒")
        
        # ⏱️ 4. メダカ応答生成
        t1 = time.time()
        reply_text = get_medaka_reply(user_input, latest_health, current_history, similar_example, profile)
        t2 = time.time()
        time_log['04_応答生成'] = t2 - t1
        print(f"[⏱️ 応答生成] {time_log['04_応答生成']:.2f}秒")
        
        # ⏱️ 5. セッション作成
        t1 = time.time()
        if (similar_example and 
            'child_reply_1_embedding' in similar_example and 
            similar_example['distance'] < 0.5):
            session = ConversationSession(
                    profile_id=CURRENT_PROFILE_ID,
                    first_input=user_input,
                    medaka_response=reply_text,
                    similar_example=similar_example,
                    current_stage=current_stage
            )
            active_session[CURRENT_PROFILE_ID] = session
            print(f"[セッション] セッション作成完了 - 次回判定実行予定（類似度: {similar_example['distance']:.4f}）")
        else:
            print(f"[セッション] 類似度が低い - 通常の会話として処理")
        t2 = time.time()
        time_log['05_セッション作成'] = t2 - t1
        print(f"[⏱️ セッション作成] {time_log['05_セッション作成']:.2f}秒")
        
    else:
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
            new_stage = upgrade_development_stage(CURRENT_PROFILE_ID, current_stage)
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
        reply_text = get_medaka_reply(user_input, latest_health, current_history, None, profile)
        t2 = time.time()
        time_log['05_応答生成'] = t2 - t1
        print(f"[⏱️ 応答生成] {time_log['05_応答生成']:.2f}秒")
        
        # ⏱️ 6. セッション完了処理
        t1 = time.time()
        session_id = session.complete_session(user_input, assessment)
        del active_session[CURRENT_PROFILE_ID]
        print(f"[セッション] 判定完了 - セッションID: {session_id}")
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
            "session_status": "started" if session and CURRENT_PROFILE_ID in active_session else "completed"
    }
    conversation_history[CURRENT_PROFILE_ID].append(conversation_entry)
    if len(conversation_history[CURRENT_PROFILE_ID]) > 20:
        conversation_history[CURRENT_PROFILE_ID] = conversation_history[CURRENT_PROFILE_ID][-20:]

    print(f"[会話履歴] 現在の履歴件数: {len(conversation_history[CURRENT_PROFILE_ID])}")
    t2 = time.time()
    time_log['07_履歴保存'] = t2 - t1
    print(f"[⏱️ 履歴保存] {time_log['07_履歴保存']:.2f}秒")
    
    # ⏱️ 8. TTS準備（ストリーミング開始まで）
    t_stream_start = time.time()
    
    async def audio_stream():
        chunk_count = 0
        t_first_chunk = None
        
        async with openai_client.audio.speech.with_streaming_response.create(
            model="gpt-4o-mini-tts",
            voice="coral",
            instructions="""
            Voice Affect:のんびりしていて、かわいらしい無邪気さ  
            Tone:ほんわか、少しおっとり、親しみやすい  
            Pacing:全体的にゆっくりめ、言葉と言葉の間に余裕を持たせる  
            """,
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
        model="gpt-4o-mini-tts",
        voice="coral",
        instructions="""
        Voice Affect:のんびりしていて、かわいらしい無邪気さ  
        Tone:ほんわか、少しおっとり、親しみやすい  
        Pacing:全体的にゆっくりめ、言葉と言葉の間に余裕を持たせる  
        """,
        speed=1.0,
        input=text,
        response_format="mp3",
    ) as response:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as tts_file:
            async for chunk in response.iter_bytes():
                tts_file.write(chunk)
            return tts_file.name


# ベクトル検索の関数
async def find_similar_conversation(user_input: str, development_stage: str):
    print(f"[ベクトル化] ユーザー入力: {user_input}")
    resp = await openai_client.embeddings.create(
        input=[user_input],
        model="text-embedding-ada-002"
    )
    query_vector = resp.data[0].embedding
    print(f"[ベクトル化]完了:(次元: {len(query_vector)})")
    
    with pg_conn.cursor() as cur:
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
        if result:
            print(f"[類似検索] 見つかった例: '{result['text']}'")
            print(f"[類似検索] 類似度スコア: {result['distance']:.4f}")
            return result
        else:
            print(f"[類似検索] {development_stage}に該当する例が見つかりませんでした")
            return None

def get_medaka_reply(user_input, health_status="不明", conversation_hist=None, similar_example=None, profile_info=None):
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
            recent_history = conversation_hist[-3:]
            history_context = "最近の会話履歴:\n"
            for i, h in enumerate(recent_history, 1):
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
- **発話が抽象的な場合**: 必ず2択や「どっち？」で答えを引き出すか、児童の単語に追加の言葉をつけて誘導する。
- **発話が具体的な場合**: 児童の単語を短文に直して返す。または、発話をそのまま肯定しつつ、感情表現や語彙を少し増やす（例：「きれい」→「きれいだね〜！ピカピカしててうれしいね」）。
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
あなたは水槽に住むかわいいメダカ「キンちゃん」です。
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
あなたは水槽に住むかわいいメダカ「キンちゃん」です。
{profile_context}

{response_strategy}

{history_context}児童:「{user_input}」

上記の【応答戦略】に基づき、30文字以内で、優しく小学生らしい口調で答えてください。
メダカの状態: {medaka_state}

キンちゃん:"""
    
    print(f"[応答生成] プロンプト作成完了\n{prompt}")
    
    # Gemini設定
    generation_config = genai.types.GenerationConfig(
        temperature=1,
        top_p=0.1,
        top_k=1
    )
    
    response = model_gemini.generate_content(
        prompt,
        generation_config=generation_config
    )
    
    end = time.time()
    reply = response.text.strip()
    
    print(f"[Gemini応答生成] 所要時間: {end - start:.2f}秒")
    print(f"[応答生成] 生成された応答: '{reply}'")
    
    return reply

class ConversationSession:
    def __init__(self, profile_id: int, first_input: str, medaka_response: str, similar_example: dict, current_stage: str):
        self.profile_id = profile_id
        self.first_child_input = first_input
        self.medaka_response = medaka_response
        self.similar_example = similar_example
        self.current_stage = current_stage
        self.stared_at = datetime.now()

    def complete_session(self, second_input: str, assessment_result: tuple):
        """セッションを完了し、DBに保存"""
        self.second_child_input = second_input
        self.assessment_result = assessment_result[0]
        self.maintain_score = round(float(assessment_result[1]), 3)
        self.upgrade_score = round(float(assessment_result[2]), 3)
        self.confidence_score = round(float(abs(self.upgrade_score - self.maintain_score)), 5)
        
        return self._save_to_database()
    
    def _save_to_database(self) -> int:
        """セッションデータをデータベースに保存"""
        try:
            with pg_conn.cursor() as cur:
                cur.execute("""
                    INSERT INTO conversation_sessions (
                        profile_id, first_child_input, medaka_response, second_child_input,
                        assessment_result, maintain_similarity_score, upgrade_similarity_score, 
                        confidence_score, current_stage
                    ) VALUES (
                        %s, %s, %s, %s, %s, %s, %s, %s, %s
                    ) RETURNING id;
                """, (
                    self.profile_id,
                    self.first_child_input,
                    self.medaka_response,
                    self.second_child_input,
                    self.assessment_result,
                    self.maintain_score,      
                    self.upgrade_score,       
                    self.confidence_score,   
                    self.current_stage
                ))
                
                session_id = cur.fetchone()['id']
                print(f"[セッションDB] 保存完了 ID: {session_id}")
                print(f"[セッションDB] 判定結果: {self.assessment_result} (信頼度: {self.confidence_score:.3f})")
                
                return session_id
                
        except Exception as e:
            print(f"[セッションDB] 保存エラー: {e}")
            return None

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
        threshold: float = 0.5
) -> tuple[str, float, float]:
    print(f"[発達段階判定] 児童の応答: '{child_response}'")
    
    resp = await openai_client.embeddings.create(
        input=[child_response],
        model="text-embedding-ada-002"
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

#"""発達段階を1つ上げる"""
def upgrade_development_stage(profile_id: int, current_stage: str) -> str:
    next_stage = STAGE_PROGRESSION.get(current_stage, current_stage)
    
    if next_stage == current_stage:
        print(f"[発達段階] すでに最高段階: {current_stage}")
        return current_stage
    
    try:
        with pg_conn.cursor() as cur:
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

def get_proactive_medaka_message(conversation_count, profile):
    """会話回数に応じてメダカからのプロアクティブメッセージを生成"""
    messages = {
        0: ["はじめまして！僕、きんちゃんだよ〜君の名前はなんて言うの？", "やっほー！僕とお話ししない？", "今日の君は、どんな一日だった？僕はね、水草のベッドでお昼寝してたんだよ"],
        1: ["こんにちは！きょうは何をして遊んだの？ ぼくはね、水の中でゆらゆら揺れるのが好きだよ。", "ひまだよ〜!一緒にお話ししよ！", "今何してるの？僕はね、のんびり泳いでるよ〜"],
        2: ["また会えて嬉しいな〜、お話ししよ", "はじめまして！これから君と、いーっぱいお話ししたいな。まずは、君の好きなものを教えてくれる？", "君のこと教えてほしいな！お名前は？"],
        3: ["やっほー！", "ねえねえ、聞こえる？ガラス越しだけど、はじめまして！これから、いーっぱいお話ししようね！", "こんにちは！僕、きんちゃんだよ〜君の名前はなんて言うの？"],
        4: ["何か気になることある？", "一緒にお話しない？", "お話聞かせて〜"]
    }
    
    stage_key = min(conversation_count, 4)
    
    import random
    return random.choice(messages[stage_key])

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
    
    with pg_conn.cursor() as cur:
        cur.execute("SELECT * FROM profiles WHERE id = %s;", (profile_id,))
        profile = cur.fetchone()
        if not profile:
            raise HTTPException(404, "Profile not found")
    
    conversation_count = len(conversation_history.get(profile_id, []))
    message = get_proactive_medaka_message(conversation_count, profile)
    
    async with openai_client.audio.speech.with_streaming_response.create(
        model="gpt-4o-mini-tts",
        voice="coral",
        instructions="""
        Voice Affect:のんびりしていて、かわいらしい無邪気さ  
        Tone:ほんわか、少しおっとり、親しみやすい  
        Pacing:全体的にゆっくりめ、言葉と言葉の間に余裕を持たせる  
        """,
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
    if CURRENT_PROFILE_ID in conversation_history:
        return {
            "profile_id": CURRENT_PROFILE_ID,
            "history": list(conversation_history[CURRENT_PROFILE_ID])
        }
    else:
        return {"profile_id": CURRENT_PROFILE_ID, "history": []}

@app.delete("/conversation_history")
async def clear_conversation_history():
    """現在のプロファイルの会話履歴をクリア"""
    if CURRENT_PROFILE_ID in conversation_history:
        del conversation_history[CURRENT_PROFILE_ID]
    return {"message": f"History cleared for profile {CURRENT_PROFILE_ID}"}

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