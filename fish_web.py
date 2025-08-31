from fastapi import FastAPI, UploadFile, HTTPException,Request,Body
from fastapi.responses import StreamingResponse,JSONResponse,FileResponse
import uvicorn
from fastapi.middleware.cors import CORSMiddleware
from openai import AsyncOpenAI
import time
import numpy as np
from collections import defaultdict, deque
import httpx
import io
import tempfile
import io
import os
import google.generativeai as genai
import csv
from datetime import datetime
import psycopg2# psycopg2はPostgreSQLデータベースに接続するためのライブラリ
from psycopg2.extras import RealDictCursor

# データベースのURLを環境変数から取得、デフォルトはローカルのPostgreSQL
DB_URL = os.getenv("DB_URL")
pg_conn = psycopg2.connect(DB_URL, cursor_factory=RealDictCursor)
pg_conn.autocommit = True #データの変更を即座にデータベースに反映させるために自動コミットを有効にする
print(f"[起動時] DB接続成功: {DB_URL}")
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
openai_client = AsyncOpenAI(api_key=OPENAI_API_KEY)
genai.configure(api_key=GEMINI_API_KEY)
INFERENCE_SERVER_URL = os.getenv("INFERENCE_SERVER_URL")
model_gemini = genai.GenerativeModel(model_name="gemini-2.0-flash")
print(f"[起動時] DB_URL設定: {'あり' if DB_URL else 'なし'}")
print(f"[起動時] OpenAI API: {'設定済み' if OPENAI_API_KEY else '未設定'}")
print(f"[起動時] Gemini API: {'設定済み' if GEMINI_API_KEY else '未設定'}")

# グローバル変数
active_session ={}  # セッション管理用
conversation_history = defaultdict(lambda:deque(maxlen=10))  
speed_history = defaultdict(lambda: deque(maxlen=75))
fps = 15
latest_health = "Normal"
track_history = defaultdict(lambda: (0, 0))  
CURRENT_PROFILE_ID = 1  #プロファイルID
last_similar_example = defaultdict(lambda: None)  # 2回目の会話待ちの情報を保持

# Session Pooler対応のデータベース接続関数
def connect_to_database(db_url, max_retries=3):
    #Supabase Session Pooler経由でデータベースに接続
    if "pooler.supabase.com" in db_url:
        print("[DB接続] Supabase Pooler接続を使用")
        # ポート番号の確認
        if ":5432" in db_url:
            print("[DB接続] Session Pooler (ポート5432)")
        elif ":6543" in db_url:
            print("[DB接続] Transaction Pooler (ポート6543)")
        
        # SSLモードの追加（必須）
        if "sslmode=" not in db_url:
            if "?" in db_url:
                db_url += "&sslmode=require"
            else:
                db_url += "?sslmode=require"
    
    print(f"[DB接続] 接続先: {db_url.split('@')[0]}@...")
    
    for attempt in range(max_retries):
        try:
            print(f"[DB接続] 試行 {attempt + 1}/{max_retries}")
            
            # Session Pooler用の接続設定
            conn = psycopg2.connect(
                db_url,
                cursor_factory=RealDictCursor,
                keepalives=1,
                keepalives_idle=30,
                keepalives_interval=10,
                keepalives_count=5,
                connect_timeout=10
            )
            
            # autocommitは重要（Session Poolerでトランザクションの問題を避ける）
            conn.autocommit = True
            
            # 接続テスト
            with conn.cursor() as cur:
                cur.execute("SELECT 1")
                result = cur.fetchone()
                if result:
                    print(f"✅ DB接続成功! (試行 {attempt + 1})")
                    
                    # テーブル確認
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
                print("Supabaseダッシュボードから正しい接続文字列を取得してください")
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



#ベクトル検索の関数
async def find_similar_conversation(user_input: str,development_stage: str):
        # ユーザー入力をベクトル化
         print(f"[ベクトル化] ユーザー入力: {user_input}")
         resp = await openai_client.embeddings.create(
            input=[user_input],
            model="text-embedding-ada-002"
        )
         query_vector = resp.data[0].embedding
         print(f"「ベクトル化]完了:(次元: {len(query_vector)})")
         #類似検索
         with pg_conn.cursor() as cur:
             cur.execute("""
                    SELECT text,fish_text,children_reply_1,children_reply_2,
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

def get_medaka_reply(user_input, healt_status="不明",conversation_hist=None,similar_example=None,profile_info=None):
    start=time.time()
    if healt_status == "Active":
        medaka_state = "元気"
    elif healt_status == "Normal":
        medaka_state = "休憩中"
    elif healt_status == "Lethargic":
        medaka_state = "元気ない"
    else:
        medaka_state = "休憩中"
    print("メダカの状態:",medaka_state)
    if profile_info:
        profile_name = profile_info.get('name', 'Unknown')
        age_text = f"{profile_info['age']}歳" if profile_info.get('age') else "年齢不明"
        stage_text = profile_info.get('development_stage', '不明')
        profile_context = f"話し相手: {profile_name}さん ({age_text}, {stage_text})\n"
        history_context=""
        if conversation_hist and len(conversation_hist) > 0:
            recent_history = conversation_hist[-3:]
            history_context = "最近の会話履歴:\n"
            for i,h in enumerate(recent_history,1):
                history_context += f"{i}. 児童「{h['child']}」→ メダカ「{h['medaka']}」\n"
        history_context += "\n"
       
    if similar_example:
                # Few-shot形式でプロンプト作成
        prompt = f"""
                あなたは水槽に住むかわいいメダカ「キンちゃん」です。
                メダカの状態: {medaka_state}
                {profile_context}
                以下の例を参考に、全く同じ言葉で応答してください。
                【会話】
                児童:「{similar_example['text']}」
                メダカ:「{similar_example['fish_text']}」

                {history_context}【現在の会話】
                児童:「{user_input}」
                メダカ:
                """

    else:
                # 類似例がない場合の基本プロンプト
                prompt = f"""
        あなたは水槽に住むかわいいメダカ「キンちゃん」です。
        以下の応答戦略を参考にして
        ・相手の短い返答を繰り返しながら、「どうして？」「どんな？」「他には？」と質問を足す。
        ・理由づけや順序立てを促す。（「まずは？次は？」など）
        ・相手の興味に沿って「もっと詳しく教えて」と掘り下げる。
        ・少しズレた説明や一方的な話でも、否定せずに聞き役になる。
        一往復ごとに「続きを話せるきっかけ」を与える
        {profile_context}
        30文字以内で、優しく小学生らしい口調で答えてください。
        児童:「{user_input}」
        メダカの状態: {medaka_state}
        キンちゃん:
        """

    print(f"[応答生成] プロンプト作成完了",prompt)
    generation_config = genai.types.GenerationConfig(
        temperature=0.5,  # 創造性を下げて一貫性を重視
        top_p=0.1,        # より決定的な応答
        top_k=1           # 最も可能性の高い選択肢のみ
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
    def __init__(self,profile_id:int ,first_input:str,medaka_response:str,similar_example: dict, current_stage: str):
        self.profile_id = profile_id
        self.first_child_input = first_input
        self.medaka_response = medaka_response
        self.similar_example = similar_example
        self.current_stage = current_stage
        self.stared_at = datetime.now()

    def complete_session(self, second_input: str, assessment_result: tuple):
        """セッションを完了し、DBに保存"""
        self.second_child_input = second_input
        self.assessment_result = assessment_result[0]  # "昇格" or "現状維持"
        self.maintain_score = round(float(assessment_result[1]), 3)      # 小数第二位まで
        self.upgrade_score = round(float(assessment_result[2]), 3)       # 小数第二位まで
        self.confidence_score = round(float(abs(self.upgrade_score - self.maintain_score)), 5)  # 小数第二位まで
        
        # データベースに保存
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

#会話分類
async def classify_child_response(
        child_response: str,
        similar_conversation: dict,
        openai_client,
        threshold: float = 0.5
) -> tuple[str, float, float]:
    print(f"[発達段階判定] 児童の応答: '{child_response}'")
    
    # 児童の応答をベクトル化
    resp = await openai_client.embeddings.create(
        input=[child_response],
        model="text-embedding-ada-002"
    )
    response_vector = np.array(resp.data[0].embedding)
    
    # データベースからのベクトルデータを適切に変換
    def convert_to_vector(embedding_data):
        """データベースからの埋め込みデータを数値ベクトルに変換"""
        if isinstance(embedding_data, str):
            import json
            return np.array(json.loads(embedding_data), dtype=float)
            
    maintain_vector = convert_to_vector(similar_conversation['child_reply_1_embedding'])
    upgrade_vector = convert_to_vector(similar_conversation['child_reply_2_embedding'])
    
    # 類似度計算
    def cosine_similarity(v1, v2):
        """コサイン類似度を計算"""
        # ベクトルの次元数チェック
        if len(v1) != len(v2):
            raise ValueError(f"ベクトル次元が一致しません: {len(v1)} vs {len(v2)}")
        
        norm1 = np.linalg.norm(v1)
        norm2 = np.linalg.norm(v2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0  # ゼロベクトルの場合は0を返す
        
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

@app.post("/talk_with_fish_text_only")  # テキストのみを返すエンドポイント
async def talk_with_fish_text_only(request: Request):
    """メダカとの会話（テキストレスポンスのみ）"""
    start_total = time.time()
    data = await request.json()
    user_input = data.get("user_input", "")
    session_id = data.get("session_id", "")  # 今回は使用しないが、互換性のため受け取る
    
    if not user_input.strip():
        raise HTTPException(400, "user_input is required")
    
    # 1) プロファイルからステージ取得
    with pg_conn.cursor() as cur:
        cur.execute("SELECT * FROM profiles WHERE id = %s;", (CURRENT_PROFILE_ID,))
        profile = cur.fetchone()
        if not profile:
            raise HTTPException(404, "Profile not found")
        current_stage = profile["development_stage"]
    
    # 会話履歴の取得/初期化
    if CURRENT_PROFILE_ID not in conversation_history:
        conversation_history[CURRENT_PROFILE_ID] = []
    current_history = conversation_history[CURRENT_PROFILE_ID]
    
    session = active_session.get(CURRENT_PROFILE_ID)
    assessment_result = None
    similar_example = None

    if session is None:
        # 1回目の会話：類似例を検索
        print("[会話フロー] 1回目の会話 - 類似例を検索")
        similar_example = await find_similar_conversation(user_input, current_stage)
        # 類似例を保存（次回の判定用）
        if (similar_example and 
            'child_reply_1_embedding' in similar_example and 
            similar_example['distance'] < 0.5):
            reply_text = get_medaka_reply(user_input, latest_health, current_history, similar_example, profile)
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
            reply_text = get_medaka_reply(user_input, latest_health, current_history, None, profile)
            print(f"[セッション] 類似度が低い - 通常の会話として処理")
    else:
        # 2回目の会話の場合、発達段階判定を実行
        print("[会話フロー] 2回目の会話 - 発達段階判定を実行")
        
        # 児童の応答分類
        assessment = await classify_child_response(
            user_input,
            session.similar_example,
            openai_client,
        )
        
        assessment_result = {
            'result': assessment[0],
            'maintain_score': round(float(assessment[1]), 3),
            'upgrade_score': round(float(assessment[2]), 3),
            'confidence_score': round(float(abs(assessment[2] - assessment[1])), 5),
            'assessed_at': datetime.now().isoformat(),
        }
        
        reply_text = get_medaka_reply(user_input, latest_health, current_history, None, profile)
        session_id = session.complete_session(user_input, assessment)
        del active_session[CURRENT_PROFILE_ID]  # セッション完了後は削除
        print(f"[セッション] 判定完了 - セッションID: {session_id}")

    # 会話履歴に追加
    conversation_entry = {
        "child": user_input,
        "medaka": reply_text,
        "timestamp": datetime.now().isoformat(),
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
    
    end_total = time.time()
    print(f"[総処理時間] {end_total - start_total:.2f}秒")
    
    # JSONレスポンスを返す
    return {
        "reply": reply_text,
        "assessment_result": assessment_result,
        "session_status": "started" if CURRENT_PROFILE_ID in active_session else "completed",
        "processing_time": round(end_total - start_total, 2)
    }

@app.post("/predict")
async def predict(file: UploadFile):
   global latest_health
   # 全体の開始時間
   total_start = time.time()
   timestamps = {}
   try:
       # 1. ファイル読み込み時間測定
       file_read_start = time.time()
       file_content = await file.read()
       timestamps['file_read'] = (time.time() - file_read_start) * 1000
       
       # ファイルサイズ情報
       file_size_kb = len(file_content) / 1024
       print(f"📁 File size: {file_size_kb:.1f}KB")
       
       # 2. HTTPクライアント作成時間
       client_create_start = time.time()
       client = httpx.AsyncClient(timeout=30.0)
       timestamps['client_create'] = (time.time() - client_create_start) * 1000
       
       # 3. リクエスト準備時間
       request_prep_start = time.time()
       files = {"file": (file.filename, io.BytesIO(file_content), file.content_type)}
       timestamps['request_prep'] = (time.time() - request_prep_start) * 1000
       
       # 4. HTTP リクエスト送信時間（最重要）
       http_request_start = time.time()
       response = await client.post(f"{INFERENCE_SERVER_URL}/predict", files=files)
       timestamps['http_request'] = (time.time() - http_request_start) * 1000
       
       # 5. レスポンス検証時間
       validation_start = time.time()
       if response.status_code != 200:
           await client.aclose()
           raise HTTPException(status_code=response.status_code, 
                             detail=f"Inference server error: {response.text}")
       timestamps['validation'] = (time.time() - validation_start) * 1000
       
       # 6. ヘッダー処理時間
       header_start = time.time()
       health_status = response.headers.get("X-Health-Status", "Unknown")
       latest_health = health_status
       timestamps['header_process'] = (time.time() - header_start) * 1000
       
       # 7. レスポンス処理時間
       response_prep_start = time.time()
       response_content = response.content
       response_size_kb = len(response_content) / 1024
       result = StreamingResponse(
           io.BytesIO(response_content),
           media_type="image/jpeg"
       )
       timestamps['response_prep'] = (time.time() - response_prep_start) * 1000
       
       # 8. クリーンアップ時間
       cleanup_start = time.time()
       await client.aclose()
       timestamps['cleanup'] = (time.time() - cleanup_start) * 1000
       
       # 合計時間計算
       total_time = (time.time() - total_start) * 1000
       
       # 🔍 詳細ログ出力
       print("\n" + "="*50)
       print("🚀 PERFORMANCE ANALYSIS")
       print("="*50)
       print(f"📤 Upload size: {file_size_kb:.1f}KB")
       print(f"📥 Response size: {response_size_kb:.1f}KB")
       print(f"🌐 Total data: {(file_size_kb + response_size_kb):.1f}KB")
       print("-" * 30)
       
       for step, duration in timestamps.items():
           percentage = (duration / total_time) * 100
           bar_length = int(percentage / 5)  # 5%につき1文字
           bar = "█" * bar_length + "░" * (20 - bar_length)
           print(f"{step:15} │ {bar} │ {duration:6.1f}ms ({percentage:4.1f}%)")
       
       print("-" * 30)
       print(f"⏱️  TOTAL TIME: {total_time:.1f}ms")
       
       # 🚨 ボトルネック特定
       max_step = max(timestamps.items(), key=lambda x: x[1])
       if max_step[1] > total_time * 0.5:  # 50%以上を占める処理
           print(f"🚨 BOTTLENECK: {max_step[0]} ({max_step[1]:.1f}ms)")
       
       # 📊 通信速度計算
       if timestamps['http_request'] > 0:
           total_data_mb = (file_size_kb + response_size_kb) / 1024
           speed_mbps = (total_data_mb * 8) / (timestamps['http_request'] / 1000)
           print(f"🌐 Effective speed: {speed_mbps:.2f} Mbps")
           
           # 速度判定
           if speed_mbps < 1:
               print("🐌 Very slow - likely bandwidth limited")
           elif speed_mbps < 10:
               print("⚠️  Slow - network/tunnel overhead")
           else:
               print("✅ Good speed - latency is the issue")
       
       print("="*50 + "\n")
       
       return result
           
   except httpx.TimeoutException:
       print(f"⏰ Timeout after {(time.time() - total_start)*1000:.1f}ms")
       raise HTTPException(status_code=504, detail="Inference server timeout")
   except httpx.ConnectError:
       print(f"🔌 Connection error after {(time.time() - total_start)*1000:.1f}ms")
       raise HTTPException(status_code=503, detail="Cannot connect to inference server")
   except Exception as e:
       print(f"💥 Error after {(time.time() - total_start)*1000:.1f}ms: {str(e)}")
       raise HTTPException(status_code=500, detail=f"Server error: {str(e)}")
   
@app.get("/")
async def read_index():
    return FileResponse('index.html', media_type='text/html')

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
def get_proactive_medaka_message(conversation_count, profile):
    """会話回数に応じてメダカからのプロアクティブメッセージを生成"""
    
    # 全ステージ共通のメッセージパターン
    messages = {
        0: ["はじめまして！僕、きんちゃんだよ〜君の名前はなんて言うの？", "やっほー！僕とお話ししない？", "今日の君は、どんな一日だった？僕はね、水草のベッドでお昼寝してたんだよ"],
        1: ["こんにちは！きょうは何をして遊んだの？ ぼくはね、水の中でゆらゆら揺れるのが好きだよ。", "ひまだよ〜!一緒にお話ししよ！","今何してるの？僕はね、のんびり泳いでるよ〜"],
        2: ["また会えて嬉しいな〜、お話ししよ", "はじめまして！これから君と、いーっぱいお話ししたいな。まずは、君の好きなものを教えてくれる？","君のこと教えてほしいな！お名前は？"],
        3: ["やっほー！", "ねえねえ、聞こえる？ガラス越しだけど、はじめまして！これから、いーっぱいお話ししようね！", "こんにちは！僕、きんちゃんだよ〜君の名前はなんて言うの？"],
        4: ["何か気になることある？", "一緒にお話しない？", "お話聞かせて〜"]
    }
    
    # 会話回数に応じてメッセージを選択（最大4まで）
    stage_key = min(conversation_count, 4)
    
    import random
    return random.choice(messages[stage_key])    
# セッション状態確認エンドポイント
@app.post("/check_session_status")
async def check_session_status(request: Request):
    data = await request.json()
    profile_id = data.get("profile_id")
    
    if not profile_id:
        raise HTTPException(400, "profile_id is required")
    
    # アクティブセッションがあるかチェック
    has_active_session = profile_id in active_session
    
    # 環境変数からプロアクティブモード設定を取得
    medaka_proactive_enabled = os.getenv("MEDAKA_PROACTIVE_ENABLED", "true").lower() == "true"
    
    return {
        "has_active_session": has_active_session,
        "conversation_count": len(conversation_history.get(profile_id, [])),
        "proactive_enabled": medaka_proactive_enabled
    }
# プロアクティブメッセージ生成エンドポイント
@app.post("/get_proactive_message")
async def get_proactive_message(request: Request):
    data = await request.json()
    profile_id = data.get("profile_id")
    
    if not profile_id:
        raise HTTPException(400, "profile_id is required")
    
    # プロファイル取得
    with pg_conn.cursor() as cur:
        cur.execute("SELECT * FROM profiles WHERE id = %s;", (profile_id,))
        profile = cur.fetchone()
        if not profile:
            raise HTTPException(404, "Profile not found")
    
    # 会話回数を取得
    conversation_count = len(conversation_history.get(profile_id, []))
    
    # プロアクティブメッセージを生成
    message = get_proactive_medaka_message(conversation_count, profile)
    
    # ★ テキストレスポンスを返す（音声合成なし）
    return JSONResponse(content={
        "message": message,
        "conversation_count": conversation_count,
        "profile_name": profile.get('name', 'Unknown'),
        "development_stage": profile.get('development_stage', '不明')
    })


#--------デバック用エンドポイント--------
@app.get("/conversation_history")
async def get_conversation_history():
    """現在のプロファイルの会話履歴を取得"""
    if CURRENT_PROFILE_ID in conversation_history:
        return {
            "profile_id": CURRENT_PROFILE_ID,
            "history": conversation_history[CURRENT_PROFILE_ID]
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
#-----------------------------------
