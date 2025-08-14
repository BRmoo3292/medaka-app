from fastapi import FastAPI, UploadFile, HTTPException,Request,Body
from fastapi.responses import StreamingResponse,FileResponse
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
from dotenv import load_dotenv
# 最初に環境変数を読み込み
load_dotenv()
# データベースのURLを環境変数から取得、デフォルトはローカルのPostgreSQL
DB_URL = os.getenv("DB_URL")
pg_conn = psycopg2.connect(DB_URL, cursor_factory=RealDictCursor)
pg_conn.autocommit = True
print(f"[起動時] DB接続成功: {DB_URL}")

pg_conn.autocommit = True #データの変更を即座にデータベースに反映させるために自動コミットを有効にする
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

openai_client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
model_gemini = genai.GenerativeModel(model_name="gemini-2.0-flash")

conversation_history = {}
INFERENCE_SERVER_URL = os.getenv("INFERENCE_SERVER_URL", "https://5f600f70dd86.ngrok-free.app")
# グローバル変数
CONVERSATION_LOG_FILE = "conversation_log.csv"
conversation_history = defaultdict(lambda:deque(maxlen=10))  
speed_history = defaultdict(lambda: deque(maxlen=75))
fps = 15
latest_health = "Normal"
track_history = defaultdict(lambda: (0, 0))  
CURRENT_PROFILE_ID = 1  #プロファイルID
last_similar_example = defaultdict(lambda: None)  # 2回目の会話待ちの情報を保持
# Session Pooler対応のデータベース接続関数
def connect_to_database(db_url, max_retries=3):
    """Supabase Session Pooler経由でデータベースに接続"""
    
    # Session Poolerの確認
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
        {profile_context}{history_context}
        児童:「{user_input}」

        30文字以内で、優しく小学生らしい口調で答えてください。
        メダカの状態: {medaka_state}

        キンちゃん:"""

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

# """会話履歴をCSVファイルに保存する"""
def log_conversation(user_input, kinchan_reply):
    """会話履歴をCSVファイルに保存する"""
    try:
        file_exists = os.path.isfile(CONVERSATION_LOG_FILE)
        
        # UTF-8 BOM付きで保存
        with open(CONVERSATION_LOG_FILE, "a", newline="", encoding="utf-8-sig") as f:
            writer = csv.writer(f, quoting=csv.QUOTE_ALL)  # 全てクォート
            if not file_exists:
                writer.writerow(["timestamp", "user_input", "kinchan_reply"])
            writer.writerow([datetime.now().isoformat(), user_input, kinchan_reply])
            
        print(f"[CSVログ] 保存成功: {user_input[:10]}...")
        
    except Exception as e:
        print(f"[CSVログ] 保存エラー: {e}")

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

@app.post("/talk_with_fish_text")#メダカとの会話
async def talk_with_fish_text(request: Request):
    start_total = time.time()
    data = await request.json()
    user_input = data.get("user_input", "")
    if not user_input.strip():
        raise HTTPException(400, "user_input is required")
    # 1) プロファイルからステージ取得
    with pg_conn.cursor() as cur:
        cur.execute("SELECT * FROM profiles WHERE id = %s;", (CURRENT_PROFILE_ID,))
        profile = cur.fetchone()
        if not profile:
            raise HTTPException(404, "Profile not found")
        current_stage = profile["development_stage"]
    #会話履歴の取得/初期化
    if CURRENT_PROFILE_ID not in conversation_history:
        conversation_history[CURRENT_PROFILE_ID] = []
    current_history = conversation_history[CURRENT_PROFILE_ID]
    #前回の類似例があるか
    similar_example = last_similar_example[CURRENT_PROFILE_ID]
    assessment_result = None  # 初期化
    if similar_example is None:
      # 1回目の会話：類似例を検索
        print("[会話フロー] 1回目の会話 - 類似例を検索")
        similar_example = await find_similar_conversation(user_input, current_stage)
        first_child_input = user_input
        # 類似例を保存（次回の判定用）
        if similar_example and 'child_reply_1_embedding' in similar_example:
            last_similar_example[CURRENT_PROFILE_ID] = similar_example
            print("[会話フロー] 類似例を保存 - 次回発達段階判定予定")
    else:
        #2回目の会話の場合、発達段階判定を実行
        print("[会話フロー] 2回目の会話 - 発達段階判定を実行")
        second_child_input = user_input
        #児童の応答分類
        assessment = await classify_child_response(
            user_input,
            similar_example,
            openai_client,
        )

        assessment_result = {
            'result': assessment[0],
            'maintain_score': float(assessment[1]),
            'upgrade_score': float(assessment[2]),
            'assessed_at': datetime.now(),
        }
        #類似例をクリア
        last_similar_example[CURRENT_PROFILE_ID] = None  
        similar_example = None

    #応答生成
    reply_text = get_medaka_reply(
        user_input,
        latest_health, 
        current_history, 
        similar_example,
        profile
    )   
    first_reply = reply_text  # 初回の応答を保存
    # 会話履歴に追加
    conversation_entry = {
        "child": user_input,
        "medaka": reply_text,
        "timestamp": datetime.now(),
        "similar_example_used": similar_example['text'] if similar_example else None,
        "similarity_score": similar_example['distance'] if similar_example else None,
        "has_assessment": 'assessment_result' in locals(),
        "assessment_result": assessment_result if 'assessment_result' in locals() else None
    }
    conversation_history[CURRENT_PROFILE_ID].append(conversation_entry)
    if len(conversation_history[CURRENT_PROFILE_ID]) > 20:
        conversation_history[CURRENT_PROFILE_ID] = conversation_history[CURRENT_PROFILE_ID][-20:]
    # CSVログ保存
    log_conversation(user_input, reply_text)
    print(f"[会話履歴] 現在の履歴件数: {len(conversation_history[CURRENT_PROFILE_ID])}")
    t2 = time.time()
    
    async with openai_client.audio.speech.with_streaming_response.create(
        model="gpt-4o-mini-tts",  # OpenAIの利用可能なTTSモデル
        voice="coral",
        instructions="""
        Voice Affect:のんびりしていて、かわいらしい無邪気さ  
        Tone:ほんわか、少しおっとり、親しみやすい  
        Pacing:全体的にゆっくりめ、言葉と言葉の間に余裕を持たせる  
        Pronunciation:語尾はやわらかく、やや伸ばし気味に（例：「ねぇ〜」「だよぉ〜」）  
        Pauses:語尾や会話の区切りで軽く間をとる  
        Dialect:標準語だが、子どもっぽいやさしい言い回し  
        Delivery:おっとりしていて、聞いていてほっとするような声  
        Phrasing:たまにちょっとズレた発言も混ぜる。例：「お空って、水のうえにあるの？」
        """,
        speed=1.0,
        input=reply_text,
        response_format="mp3",
    ) as response:
        t3 = time.time()    
        #ストリーミング再生
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as tts_file:
            async for chunk in response.iter_bytes():
                tts_file.write(chunk)
            tts_path = tts_file.name
    end_total = time.time()
    print(f"[TTS生成] {t3 - t2:.2f}秒")
    print(f"[総処理時間] {end_total - start_total:.2f}秒")
    return FileResponse(tts_path, media_type="audio/mpeg", filename="reply.mp3")

@app.post("/predict")
async def predict(file: UploadFile):
    global latest_health
    try:
        # ファイルを読み込み
        file_content = await file.read()
        
        # 推論サーバーにリクエスト送信
        async with httpx.AsyncClient(timeout=30.0) as client:
            files = {"file": (file.filename, io.BytesIO(file_content), file.content_type)}
            response = await client.post(f"{INFERENCE_SERVER_URL}/predict", files=files)
            
            if response.status_code != 200:
                raise HTTPException(status_code=response.status_code, 
                                  detail=f"Inference server error: {response.text}")
            
            # ヘルスステータスをヘッダーから取得
            health_status = response.headers.get("X-Health-Status", "Unknown")
            latest_health = health_status
            
            # 画像データを返す
            return StreamingResponse(
                io.BytesIO(response.content),
                media_type="image/jpeg"
            )
            
    except httpx.TimeoutException:
        raise HTTPException(status_code=504, detail="Inference server timeout")
    except httpx.ConnectError:
        raise HTTPException(status_code=503, detail="Cannot connect to inference server")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Server error: {str(e)}")

@app.get("/")
async def read_index():
    return FileResponse('index.html', media_type='text/html')

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)

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
