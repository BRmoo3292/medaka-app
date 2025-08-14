import os
import sys
import traceback
from dotenv import load_dotenv

print("=" * 50)
print("アプリケーション起動開始")
print("=" * 50)

# 環境変数読み込み
try:
    load_dotenv()
    print("✅ .env読み込み成功")
except Exception as e:
    print(f"⚠️ .env読み込みエラー（続行）: {e}")

# 環境変数の確認
DB_URL = os.getenv("DB_URL")
PORT = int(os.getenv("PORT", 8000))

print(f"DB_URL: {'設定済み' if DB_URL else '未設定'}")
print(f"PORT: {PORT}")

# 必須ライブラリのインポートテスト
required_libs = {
    "fastapi": "FastAPI",
    "uvicorn": "uvicorn",
    "psycopg2": "psycopg2",
}

missing_libs = []
for module, name in required_libs.items():
    try:
        __import__(module)
        print(f"✅ {name} インポート成功")
    except ImportError as e:
        print(f"❌ {name} インポート失敗: {e}")
        missing_libs.append(name)

if missing_libs:
    print(f"\n❌ 必須ライブラリが不足: {', '.join(missing_libs)}")
    print("requirements.txtを確認してください")
    sys.exit(1)

# データベース接続テスト（シンプル版）
if DB_URL:
    print("\n" + "=" * 50)
    print("データベース接続テスト")
    print("=" * 50)
    
    import psycopg2
    
    try:
        # 最小限の接続パラメータ
        conn = psycopg2.connect(DB_URL)
        conn.autocommit = True
        
        with conn.cursor() as cur:
            cur.execute("SELECT 1")
            result = cur.fetchone()
            print(f"✅ DB接続成功: {result}")
            
            # テーブル確認
            cur.execute("""
                SELECT table_name 
                FROM information_schema.tables 
                WHERE table_schema = 'public' 
                LIMIT 3
            """)
            tables = cur.fetchall()
            print(f"検出テーブル: {[t[0] for t in tables]}")
        
        conn.close()
        
    except Exception as e:
        print(f"❌ DB接続エラー: {e}")
        print(f"エラータイプ: {type(e).__name__}")
        traceback.print_exc()

# FastAPIアプリケーション
print("\n" + "=" * 50)
print("FastAPIアプリケーション起動")
print("=" * 50)

try:
    from fastapi import FastAPI
    from fastapi.responses import JSONResponse
    import uvicorn
    
    app = FastAPI()
    
    @app.get("/")
    async def root():
        return {"status": "running", "message": "Minimal debug app"}
    
    @app.get("/health")
    async def health():
        health_status = {
            "status": "healthy",
            "db_configured": bool(DB_URL),
            "port": PORT
        }
        
        # DB接続テスト（オプション）
        if DB_URL:
            try:
                import psycopg2
                conn = psycopg2.connect(DB_URL)
                conn.close()
                health_status["db_connection"] = "ok"
            except Exception as e:
                health_status["db_connection"] = f"error: {str(e)[:100]}"
        
        return health_status
    
    @app.exception_handler(Exception)
    async def global_exception_handler(request, exc):
        print(f"グローバルエラー: {exc}")
        traceback.print_exc()
        return JSONResponse(
            status_code=500,
            content={"error": str(exc)}
        )
    
    # サーバー起動
    if __name__ == "__main__":
        print(f"サーバー起動: http://0.0.0.0:{PORT}")
        uvicorn.run(
            app,
            host="0.0.0.0",
            port=PORT,
            log_level="info"
        )

except Exception as e:
    print(f"❌ 起動エラー: {e}")
    traceback.print_exc()
    sys.exit(1)