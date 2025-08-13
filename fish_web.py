import os
from dotenv import load_dotenv

# 最初に環境変数を読み込み
load_dotenv()

from fastapi import FastAPI, UploadFile, HTTPException, Request
from fastapi.responses import StreamingResponse, FileResponse
import uvicorn
from fastapi.middleware.cors import CORSMiddleware
import time
import numpy as np
from collections import defaultdict, deque
import httpx
import io
import tempfile
import csv
from datetime import datetime
import psycopg2
from psycopg2.extras import RealDictCursor

# 環境変数取得
DB_URL = os.getenv("DB_URL")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
INFERENCE_SERVER_URL = os.getenv("INFERENCE_SERVER_URL", "http://localhost:8001")

print(f"[起動時] DB_URL設定: {'あり' if DB_URL else 'なし'}")
print(f"[起動時] OpenAI API: {'設定済み' if OPENAI_API_KEY else '未設定'}")
print(f"[起動時] Gemini API: {'設定済み' if GEMINI_API_KEY else '未設定'}")

if not DB_URL:
    print("❌ 致命的エラー: DB_URLが設定されていません")
    exit(1)

# データベース接続
try:
    pg_conn = psycopg2.connect(DB_URL, cursor_factory=RealDictCursor)
    pg_conn.autocommit = True
    print(f"✅ DB接続成功!")
except Exception as e:
    print(f"❌ DB接続エラー: {e}")
    exit(1)

# OpenAI設定（条件付き）
openai_client = None
if OPENAI_API_KEY and OPENAI_API_KEY.startswith("sk-"):
    try:
        from openai import AsyncOpenAI
        openai_client = AsyncOpenAI(api_key=OPENAI_API_KEY)
        print("✅ OpenAI API設定完了")
    except Exception as e:
        print(f"⚠️ OpenAI設定エラー: {e}")
        openai_client = None
else:
    print("⚠️ OpenAI API未設定")

# Gemini設定（条件付き）
model_gemini = None
if GEMINI_API_KEY and GEMINI_API_KEY.startswith("AIza"):
    try:
        import google.generativeai as genai
        genai.configure(api_key=GEMINI_API_KEY)
        model_gemini = genai.GenerativeModel(model_name="gemini-2.0-flash")
        print("✅ Gemini API設定完了")
    except Exception as e:
        print(f"⚠️ Gemini設定エラー: {e}")
        model_gemini = None
else:
    print("⚠️ Gemini API未設定")

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# グローバル変数
CONVERSATION_LOG_FILE = "conversation_log.csv"
conversation_history = defaultdict(lambda: deque(maxlen=10))
speed_history = defaultdict(lambda: deque(maxlen=75))
fps = 15
latest_health = "Normal"
track_history = defaultdict(lambda: (0, 0))
CURRENT_PROFILE_ID = 1
last_similar_example = defaultdict(lambda: None)

# ベクトル検索の関数
async def find_similar_conversation(user_input: str, development_stage: str):
    if not openai_client:
        print("[ベクトル検索] OpenAI未設定のため、テキスト検索")
        try:
            with pg_conn.cursor() as cur:
                cur.execute("""
                    SELECT text, fish_text, children_reply_1, children_reply_2,
                           children_reply_1_embedding, children_reply_2_embedding
                    FROM conversations
                    WHERE development_stage = %s
                    AND (text ILIKE %s OR text ILIKE %s)
                    ORDER BY created_at DESC
                    LIMIT 1;
                """, (development_stage, f"%{user_input}%", f"%{user_input[:5]}%"))
                result = cur.fetchone()
                if result:
                    print(f"[テキスト検索] 見つかった例: '{result['text']}'")
                    return result
                return None
        except Exception as e:
            print(f"[検索エラー] {e}")
            return None
    
    # OpenAI利用可能時のベクトル検索
    print(f"[ベクトル化] ユーザー入力: {user_input}")
    try:
        resp = await openai_client.embeddings.create(
            input=[user_input],
            model="text-embedding-ada-002"
        )
        query_vector = resp.data[0].embedding
        print(f"[ベクトル化] 完了: (次元: {len(query_vector)})")
        
        with pg_conn.cursor() as cur:
            cur.execute("""
                SELECT text, fish_text, children_reply_1, children_reply_2,
                       children_reply_1_embedding, children_reply_2_embedding,
                       user_embedding <-> %s::vector as distance
                FROM conversations
                WHERE development_stage = %s
                AND user_embedding IS NOT NULL
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
    except Exception as e:
        print(f"[ベクトル検索エラー] {e}")
        return None

def get_medaka_reply(user_input, health_status="不明", conversation_hist=None, similar_example=None, profile_info=None):
    start = time.time()
    
    # 健康状態の変換
    if health_status == "Active":
        medaka_state = "元気"
    elif health_status == "Normal":
        medaka_state = "休憩中"
    elif health_status == "Lethargic":
        medaka_state = "元気ない"
    else:
        medaka_state = "休憩中"
    
    print("メダカの状態:", medaka_state)
    
    # プロファイル情報の処理
    profile_context = ""
    if profile_info:
        profile_name = profile_info.get('name', 'Unknown')
        age_text = f"{profile_info['age']}歳" if profile_info.get('age') else "年齢不明"
        stage_text = profile_info.get('development_stage', '不明')
        profile_context = f"話し相手: {profile_name}さん ({age_text}, {stage_text})\n"
    
    # 会話履歴の処理
    history_context = ""
    if conversation_hist and len(conversation_hist) > 0:
        recent_history = conversation_hist[-3:]
        history_context = "最近の会話履歴:\n"
        for i, h in enumerate(recent_history, 1):
            history_context += f"{i}. 児童「{h['child']}」→ メダカ「{h['medaka']}」\n"
        history_context += "\n"
    
    # プロンプト作成
    if similar_example:
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
        prompt = f"""
        あなたは水槽に住むかわいいメダカ「キンちゃん」です。
        {profile_context}{history_context}
        児童:「{user_input}」

        30文字以内で、優しく小学生らしい口調で答えてください。
        メダカの状態: {medaka_state}

        キンちゃん:"""

    print(f"[応答生成] プロンプト作成完了")
    
    # 応答生成（Gemini利用可能時）
    if model_gemini:
        try:
            generation_config = {
                'temperature': 0.5,
                'top_p': 0.1,
                'top_k': 1
            }
            response = model_gemini.generate_content(prompt, generation_config=generation_config)
            reply = response.text.strip()
            end = time.time()
            print(f"[Gemini応答生成] 所要時間: {end - start:.2f}秒")
            print(f"[応答生成] 生成された応答: '{reply}'")
            return reply
        except Exception as e:
            print(f"[Gemini応答エラー] {e}")
            # フォールバックルール
    
    # Gemini未設定時のルールベース応答
    if any(word in user_input for word in ['こんにちは', 'おはよう', 'こんばんは']):
        return f"こんにちは！今日は{medaka_state}だよ〜"
    elif any(word in user_input for word in ['元気', 'げんき']):
        return f"うん、{medaka_state}にしてるよ！"
    elif any(word in user_input for word in ['泳ぐ', 'およぐ', '泳い']):
        return "一緒に泳ごうね〜スイスイ〜"
    elif any(word in user_input for word in ['ありがとう', 'ありがと']):
        return "どういたしまして〜また遊ぼうね！"
    elif any(word in user_input for word in ['好き', 'すき']):
        return "わたしも大好きだよ〜♪"
    else:
        return f"うんうん、そうなんだ〜{medaka_state}に聞いてるよ！"

def log_conversation(user_input, kinchan_reply):
    try:
        file_exists = os.path.isfile(CONVERSATION_LOG_FILE)
        with open(CONVERSATION_LOG_FILE, "a", newline="", encoding="utf-8-sig") as f:
            writer = csv.writer(f, quoting=csv.QUOTE_ALL)
            if not file_exists:
                writer.writerow(["timestamp", "user_input", "kinchan_reply"])
            writer.writerow([datetime.now().isoformat(), user_input, kinchan_reply])
        print(f"[CSVログ] 保存成功: {user_input[:10]}...")
    except Exception as e:
        print(f"[CSVログ] 保存エラー: {e}")

# 会話分類（条件付き実行）
async def classify_child_response(child_response: str, similar_conversation: dict, openai_client, threshold: float = 0.5):
    if not openai_client:
        print("[発達段階判定] OpenAI未設定のためスキップ")
        return "現状維持", 0.0, 0.0
    
    print(f"[発達段階判定] 児童の応答: '{child_response}'")
    try:
        resp = await openai_client.embeddings.create(
            input=[child_response],
            model="text-embedding-ada-002"
        )
        response_vector = np.array(resp.data[0].embedding)
        
        # ベクトル変換関数
        def convert_to_vector(embedding_data):
            if isinstance(embedding_data, str):
                import json
                return np.array(json.loads(embedding_data), dtype=float)
            return np.array(embedding_data, dtype=float)
        
        maintain_vector = convert_to_vector(similar_conversation['children_reply_1_embedding'])
        upgrade_vector = convert_to_vector(similar_conversation['children_reply_2_embedding'])
        
        def cosine_similarity(v1, v2):
            if len(v1) != len(v2):
                return 0.0
            norm1 = np.linalg.norm(v1)
            norm2 = np.linalg.norm(v2)
            if norm1 == 0 or norm2 == 0:
                return 0.0
            return np.dot(v1, v2) / (norm1 * norm2)
        
        maintain_similarity = cosine_similarity(response_vector, maintain_vector)
        upgrade_similarity = cosine_similarity(response_vector, upgrade_vector)
        
        result = "昇格" if upgrade_similarity > maintain_similarity and upgrade_similarity > threshold else "現状維持"
        confidence = abs(upgrade_similarity - maintain_similarity)
        
        print(f"[発達段階判定] 結果: {result} (信頼度: {confidence:.4f})")
        return result, maintain_similarity, upgrade_similarity
        
    except Exception as e:
        print(f"[発達段階判定エラー] {e}")
        return "現状維持", 0.0, 0.0

@app.post("/talk_with_fish_text")
async def talk_with_fish_text(request: Request):
    start_total = time.time()
    data = await request.json()
    user_input = data.get("user_input", "")
    
    if not user_input.strip():
        raise HTTPException(400, "user_input is required")
    
    try:
        # プロファイル取得
        with pg_conn.cursor() as cur:
            cur.execute("SELECT * FROM profiles WHERE id = %s;", (CURRENT_PROFILE_ID,))
            profile = cur.fetchone()
            if not profile:
                # デフォルトプロファイル
                profile = {"id": 1, "name": "テストユーザー", "development_stage": "stage_1"}
        
        current_stage = profile["development_stage"]
        
        # 会話履歴の取得/初期化
        if CURRENT_PROFILE_ID not in conversation_history:
            conversation_history[CURRENT_PROFILE_ID] = []
        current_history = conversation_history[CURRENT_PROFILE_ID]
        
        # 前回の類似例があるか
        similar_example = last_similar_example[CURRENT_PROFILE_ID]
        assessment_result = None
        
        if similar_example is None:
            # 1回目の会話：類似例を検索
            print("[会話フロー] 1回目の会話 - 類似例を検索")
            similar_example = await find_similar_conversation(user_input, current_stage)
            
            # 類似例を保存（次回の判定用）
            if similar_example and 'children_reply_1_embedding' in similar_example:
                last_similar_example[CURRENT_PROFILE_ID] = similar_example
                print("[会話フロー] 類似例を保存 - 次回発達段階判定予定")
        else:
            # 2回目の会話の場合、発達段階判定を実行
            print("[会話フロー] 2回目の会話 - 発達段階判定を実行")
            
            # 児童の応答分類
            assessment = await classify_child_response(user_input, similar_example, openai_client)
            
            assessment_result = {
                'result': assessment[0],
                'maintain_score': float(assessment[1]),
                'upgrade_score': float(assessment[2]),
                'assessed_at': datetime.now(),
            }
            
            # 類似例をクリア
            last_similar_example[CURRENT_PROFILE_ID] = None
            similar_example = None
        
        # 応答生成
        reply_text = get_medaka_reply(user_input, latest_health, current_history, similar_example, profile)
        
        # 会話履歴に追加
        conversation_entry = {
            "child": user_input,
            "medaka": reply_text,
            "timestamp": datetime.now(),
            "similar_example_used": similar_example['text'] if similar_example else None,
            "has_assessment": assessment_result is not None,
            "assessment_result": assessment_result
        }
        conversation_history[CURRENT_PROFILE_ID].append(conversation_entry)
        
        if len(conversation_history[CURRENT_PROFILE_ID]) > 20:
            conversation_history[CURRENT_PROFILE_ID] = conversation_history[CURRENT_PROFILE_ID][-20:]
        
        # CSVログ保存
        log_conversation(user_input, reply_text)
        
        # 音声合成（OpenAI利用可能時）
        if openai_client:
            try:
                async with openai_client.audio.speech.with_streaming_response.create(
                    model="tts-1",
                    voice="nova",
                    input=reply_text,
                    response_format="mp3",
                ) as response:
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as tts_file:
                        async for chunk in response.iter_bytes():
                            tts_file.write(chunk)
                        tts_path = tts_file.name
                
                end_total = time.time()
                print(f"[総処理時間] {end_total - start_total:.2f}秒")
                return FileResponse(tts_path, media_type="audio/mpeg", filename="reply.mp3")
            except Exception as e:
                print(f"[音声合成エラー] {e}")
                return {"reply": reply_text, "audio": False}
        else:
            return {"reply": reply_text, "audio": False}
        
    except Exception as e:
        print(f"[会話エラー] {e}")
        raise HTTPException(500, f"Internal server error: {str(e)}")

@app.post("/predict")
async def predict(file: UploadFile):
    global latest_health
    
    try:
        file_content = await file.read()
        
        async with httpx.AsyncClient(timeout=30.0) as client:
            files = {"file": (file.filename, io.BytesIO(file_content), file.content_type)}
            response = await client.post(f"{INFERENCE_SERVER_URL}/predict", files=files)
            
            if response.status_code != 200:
                raise HTTPException(status_code=response.status_code, 
                                  detail=f"Inference server error: {response.text}")
            
            health_status = response.headers.get("X-Health-Status", "Unknown")
            latest_health = health_status
            
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
    return {"message": "Medaka Fish App is running!", "status": "healthy"}

@app.get("/health")
async def health_check():
    try:
        with pg_conn.cursor() as cur:
            cur.execute("SELECT COUNT(*) as count FROM conversations")
            count = cur.fetchone()['count']
            return {
                "status": "healthy", 
                "database": "connected",
                "conversations": count,
                "openai": "configured" if openai_client else "not configured",
                "gemini": "configured" if model_gemini else "not configured"
            }
    except Exception as e:
        return {"status": "unhealthy", "error": str(e)}

# デバッグ用エンドポイント
@app.get("/conversation_history")
async def get_conversation_history():
    if CURRENT_PROFILE_ID in conversation_history:
        return {
            "profile_id": CURRENT_PROFILE_ID,
            "history": list(conversation_history[CURRENT_PROFILE_ID])
        }
    else:
        return {"profile_id": CURRENT_PROFILE_ID, "history": []}

@app.delete("/conversation_history")
async def clear_conversation_history():
    if CURRENT_PROFILE_ID in conversation_history:
        del conversation_history[CURRENT_PROFILE_ID]
    return {"message": f"History cleared for profile {CURRENT_PROFILE_ID}"}

@app.post("/test_vector_search")
async def test_vector_search(request: Request):
    data = await request.json()
    user_input = data.get("user_input", "")
    stage = data.get("stage", "stage_1")
    
    result = await find_similar_conversation(user_input, stage)
    return {"query": user_input, "stage": stage, "result": result}

if __name__ == "__main__":
    print("🚀 Medaka Fish App starting...")
    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", 8000)))