from fastapi import FastAPI, UploadFile, HTTPException, Request, WebSocket, WebSocketDisconnect
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
from io import BytesIO
from typing import AsyncGenerator

# データベースのURLを環境変数から取得
DB_URL = os.getenv("DB_URL")
pg_conn = psycopg2.connect(DB_URL, cursor_factory=RealDictCursor)
pg_conn.autocommit = True
print(f"[起動時] DB接続成功: {DB_URL}")

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
openai_client = AsyncOpenAI(api_key=OPENAI_API_KEY)
genai.configure(api_key=GEMINI_API_KEY)
model_gemini = genai.GenerativeModel(model_name="gemini-2.0-flash")

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

# --- HELPER FUNCTIONS AND CLASSES (DEFINED BEFORE USAGE) ---

def connect_to_database(db_url, max_retries=3):
    # ... (omitted for brevity, same as before)
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

try:
    pg_conn = connect_to_database(DB_URL)
    if not pg_conn:
        print("❌ データベース接続を確立できませんでした")
        exit(1)
except Exception as e:
    print(f"❌ DB接続エラー: {e}")
    exit(1)

async def get_profile_async(profile_id: int):
    loop = asyncio.get_event_loop()
    def get_profile_sync(profile_id: int):
        with pg_conn.cursor() as cur:
            cur.execute("SELECT * FROM profiles WHERE id = %s;", (profile_id,))
            profile = cur.fetchone()
            if not profile:
                raise HTTPException(404, "Profile not found")
            return profile
    return await loop.run_in_executor(None, get_profile_sync, profile_id)

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
        """, (np.array(query_vector), development_stage))
        result = cur.fetchone()
        if result:
            print(f"[類似検索] 見つかった例: '{result['text']}'")
            print(f"[類似検索] 類似度スコア: {result['distance']:.4f}")
            return result
        else:
            print(f"[類似検索] {development_stage}に該当する例が見つかりませんでした")
            return None

def get_medaka_reply(user_input, health_status="不明", conversation_hist=None, similar_example=None, profile_info=None):
    # ... (omitted for brevity, same as before)
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
    
    if profile_info:
        profile_name = profile_info.get('name', 'Unknown')
        age_text = f"{profile_info['age']}歳" if profile_info.get('age') else "年齢不明"
        stage_text = profile_info.get('development_stage', '不明')
        profile_context = f"話し相手: {profile_name}さん ({age_text}, {stage_text}) \n"
        
        history_context = ""
        if conversation_hist and len(conversation_hist) > 0:
            recent_history = conversation_hist[-3:]
            history_context = "最近の会話履歴:\n"
            for i, h in enumerate(recent_history, 1):
                history_context += f"{i}. 児童「{h['child']}」→ メダカ「{h['medaka']}」\n"
        history_context += "\n"
        
        stage = profile_info.get('development_stage', 'stage_1')
        
        if stage == 'stage_1': child_expression_level = 1
        elif stage == 'stage_2': child_expression_level = 2
        elif stage == 'stage_3': child_expression_level = 3
        else: child_expression_level = 1
    else:
        profile_context = ""
        history_context = ""
        child_expression_level = 1
    
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
        response_strategy = ""
    
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
        prompt = f"""
あなたは水槽に住むかわいいメダカ「キンちゃん」です。
{profile_context}

{response_strategy}

{history_context}児童:「{user_input}」

上記の【応答戦略】に基づき、30文字以内で、優しく小学生らしい口調で答えてください。
メダカの状態: {medaka_state}

キンちゃん:"""

    generation_config = genai.types.GenerationConfig(temperature=1, top_p=0.1, top_k=1)
    response = model_gemini.generate_content(prompt, generation_config=generation_config)
    reply = response.text.strip()
    
    print(f"[Gemini応答生成] 所要時間: {time.time() - start:.2f}秒")
    print(f"[応答生成] 生成された応答: '{reply}'")
    return reply

class ConversationSession:
    # ... (omitted for brevity, same as before)
    def __init__(self, profile_id: int, first_input: str, medaka_response: str, similar_example: dict, current_stage: str):
        self.profile_id = profile_id
        self.first_child_input = first_input
        self.medaka_response = medaka_response
        self.similar_example = similar_example
        self.current_stage = current_stage
        self.stared_at = datetime.now()

    def complete_session(self, second_input: str, assessment_result: tuple):
        self.second_child_input = second_input
        self.assessment_result = assessment_result[0]
        self.maintain_score = round(float(assessment_result[1]), 3)
        self.upgrade_score = round(float(assessment_result[2]), 3)
        self.confidence_score = round(float(abs(self.upgrade_score - self.maintain_score)), 5)
        return self._save_to_database()
    
    def _save_to_database(self) -> int:
        try:
            with pg_conn.cursor() as cur:
                cur.execute("""
                    INSERT INTO conversation_sessions (
                        profile_id, first_child_input, medaka_response, second_child_input,
                        assessment_result, maintain_similarity_score, upgrade_similarity_score, 
                        confidence_score, current_stage
                    ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s) RETURNING id;
                """, (
                    self.profile_id, self.first_child_input, self.medaka_response,
                    self.second_child_input, self.assessment_result, self.maintain_score,      
                    self.upgrade_score, self.confidence_score, self.current_stage
                ))
                session_id = cur.fetchone()['id']
                print(f"[セッションDB] 保存完了 ID: {session_id}")
                return session_id
        except Exception as e:
            print(f"[セッションDB] 保存エラー: {e}")
            return None

STAGE_PROGRESSION = {"stage_1": "stage_2", "stage_2": "stage_3", "stage_3": "stage_3"}

async def classify_child_response(child_response: str, similar_conversation: dict, openai_client, threshold: float = 0.5) -> tuple[str, float, float]:
    # ... (omitted for brevity, same as before)
    resp = await openai_client.embeddings.create(input=[child_response], model="text-embedding-ada-002")
    response_vector = np.array(resp.data[0].embedding)
    
    def convert_to_vector(embedding_data):
        if isinstance(embedding_data, str):
            import json
            return np.array(json.loads(embedding_data), dtype=float)
        if isinstance(embedding_data, list):
            return np.array(embedding_data, dtype=float)
        return np.array([])

    maintain_vector = convert_to_vector(similar_conversation['child_reply_1_embedding'])
    upgrade_vector = convert_to_vector(similar_conversation['child_reply_2_embedding'])
    
    def cosine_similarity(v1, v2):
        if v1.size == 0 or v2.size == 0 or v1.shape != v2.shape: return 0.0
        norm1, norm2 = np.linalg.norm(v1), np.linalg.norm(v2)
        if norm1 == 0 or norm2 == 0: return 0.0
        return np.dot(v1, v2) / (norm1 * norm2)
    
    maintain_similarity = cosine_similarity(response_vector, maintain_vector)
    upgrade_similarity = cosine_similarity(response_vector, upgrade_vector)
    
    result = "昇格" if upgrade_similarity > maintain_similarity and upgrade_similarity > threshold else "現状維持"
    confidence = abs(upgrade_similarity - maintain_similarity)
    
    return result, maintain_similarity, upgrade_similarity

def upgrade_development_stage(profile_id: int, current_stage: str) -> str:
    # ... (omitted for brevity, same as before)
    next_stage = STAGE_PROGRESSION.get(current_stage, current_stage)
    if next_stage == current_stage: return current_stage
    try:
        with pg_conn.cursor() as cur:
            cur.execute("UPDATE profiles SET development_stage = %s, updated_at = NOW() WHERE id = %s RETURNING development_stage;", (next_stage, profile_id))
            result = cur.fetchone()
            if result: return next_stage
            else: return current_stage
    except Exception as e:
        print(f"[発達段階] 更新エラー: {e}")
        return current_stage

async def transcribe_audio_bytes(audio_bytes: bytes) -> dict:
    audio_file = BytesIO(audio_bytes)
    audio_file.name = "from_websocket.webm"
    transcript = await openai_client.audio.transcriptions.create(model="gpt-4o-mini-transcribe", file=audio_file, language="ja", response_format="text")
    return {"text": transcript}

# --- MAIN CONVERSATION LOGIC ---

async def handle_conversation_logic(audio_bytes: bytes) -> AsyncGenerator[bytes, None]:
    start_total = time.time()
    
    transcription_task = transcribe_audio_bytes(audio_bytes)
    profile_task = get_profile_async(CURRENT_PROFILE_ID)
    transcription_result, profile = await asyncio.gather(transcription_task, profile_task)
    
    user_input = transcription_result["text"]
    current_stage = profile["development_stage"]
    
    print(f"[⏱️ 音声認識+プロファイル（並列）] {time.time() - start_total:.2f}秒")
    
    current_history = conversation_history[CURRENT_PROFILE_ID]
    session = active_session.get(CURRENT_PROFILE_ID)
    
    assessment_result = None  
    similar_example = None

    if session is None:
        print("[会話フロー] 1回目の会話 - 類似例を検索")
        similar_example = await find_similar_conversation(user_input, current_stage)
        reply_text = get_medaka_reply(user_input, latest_health, current_history, similar_example, profile)
        
        if (similar_example and 'child_reply_1_embedding' in similar_example and similar_example['distance'] < 0.5):
            session = ConversationSession(profile_id=CURRENT_PROFILE_ID, first_input=user_input, medaka_response=reply_text, similar_example=similar_example, current_stage=current_stage)
            active_session[CURRENT_PROFILE_ID] = session
    else:
        print("[会話フロー] 2回目の会話 - 発達段階判定を実行")
        assessment = await classify_child_response(user_input, session.similar_example, openai_client)
        
        assessment_result = {'result': assessment[0], 'maintain_score': round(float(assessment[1]), 3), 'upgrade_score': round(float(assessment[2]), 3), 'confidence_score': round(float(abs(assessment[2] - assessment[1])), 5), 'assessed_at': datetime.now()}
        
        if assessment[0] == "昇格":
            new_stage = upgrade_development_stage(CURRENT_PROFILE_ID, current_stage)
            profile["development_stage"] = new_stage
            if new_stage != current_stage:
                assessment_result.update({'stage_upgraded': True, 'previous_stage': current_stage, 'new_stage': new_stage})
            else:
                assessment_result.update({'stage_upgraded': False, 'already_max': True})
        else:
            assessment_result['stage_upgraded'] = False
        
        reply_text = get_medaka_reply(user_input, latest_health, current_history, None, profile)
        session.complete_session(user_input, assessment)
        del active_session[CURRENT_PROFILE_ID]

    conversation_entry = {"child": user_input, "medaka": reply_text, "timestamp": datetime.now(), "assessment_result": assessment_result}
    current_history.append(conversation_entry)
    
    print(f"Total processing time: {time.time() - start_total:.2f}s")

    async with openai_client.audio.speech.with_streaming_response.create(
        model="gpt-4o-mini-tts", voice="coral", speed=1.0, input=reply_text, response_format="mp3",
    ) as response:
        async for chunk in response.iter_bytes():
            yield chunk

# --- API ENDPOINTS ---

@app.get("/")
async def read_index():
    return FileResponse('index.html', media_type='text/html')

@app.get("/best.onnx")
async def serve_onnx_model():
    model_path = "best.onnx"
    if not os.path.exists(model_path):
        raise HTTPException(404, f"Model file not found: {model_path}")
    with open(model_path, "rb") as f: content = f.read()
    return Response(content=content, media_type="application/octet-stream", headers={"Access-Control-Allow-Origin": "*"})

@app.post("/talk_with_fish_text")
async def talk_with_fish_text(file: UploadFile):
    audio_bytes = await file.read()
    return StreamingResponse(handle_conversation_logic(audio_bytes), media_type="audio/mpeg")

@app.websocket("/ws/talk")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    print("[WebSocket] クライアント接続")
    try:
        while True:
            audio_data = bytearray()
            while True:
                data = await websocket.receive_bytes()
                if data == b'END':
                    print(f"[WebSocket] 音声受信完了 ({len(audio_data)} bytes)")
                    break
                audio_data.extend(data)
            
            async for chunk in handle_conversation_logic(bytes(audio_data)):
                await websocket.send_bytes(chunk)
            print("[WebSocket] 応答音声の送信完了")

    except WebSocketDisconnect:
        print("[WebSocket] クライアント切断")
    except Exception as e:
        print(f"[WebSocket] エラー発生: {e}")

@app.post("/update_health")
async def update_health(request: Request):
    global latest_health
    data = await request.json()
    latest_health = data.get("status", "Unknown")
    print(f"[元気度更新] {latest_health}")
    return {"status": "success", "current_health": latest_health}

# Other endpoints... (get_proactive_message, check_session_status, etc.)
# ...

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
