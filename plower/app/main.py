# ~/data/plower/app/main.py

import os
from fastapi import FastAPI, Request, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import requests
from google import genai
from dotenv import load_dotenv

# .envファイルから環境変数をロード
load_dotenv()

# 環境変数からAPIキーを取得
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    # 実際にはサーバー起動時にチェックする
    print("FATAL: GEMINI_API_KEY is not set in the .env file.") 

# Geminiクライアントの初期化
# キーがない場合、この初期化処理でエラーになる可能性があるため、キーチェックを強化
if GEMINI_API_KEY:
    client = genai.Client(api_key=GEMINI_API_KEY)
else:
    # APIキーがない場合はダミーを設定 (後続のチェックでエラーを発生させる)
    client = None
    
app = FastAPI()

# ⚠️ CORS設定: フロントエンドのポートと合わせてください
# ローカルで開発する場合、許可するオリジンを設定します。
origins = [
    "http://127.0.0.1",
    "http://localhost",
    "http://localhost:8000",  # HTML簡易サーバーのポート
    "http://127.0.0.1:8000",  # HTML簡易サーバーのポート (IP指定時)
    "http://localhost:8001",  # FastAPIサーバーのポート
    "http://localhost:5500",  # Live ServerなどのHTML実行環境のポート
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # 開発中は * でも良いが、本番では上記リストに絞るべき
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# リクエストボディの型定義
class GeminiRequest(BaseModel):
    model: str = "gemini-1.5-flash" # デフォルトモデル
    prompt: str
    temperature: float = 0.1
    
# ヘルスチェックエンドポイント
@app.get("/")
def read_root():
    return {"message": "Plower Gemini Proxy is running"}

# Gemini APIを中継するプロキシエンドポイント
@app.post("/api/gemini_proxy")
async def gemini_proxy(request_data: GeminiRequest):
    """
    フロントエンドからのプロンプトを受け取り、Gemini APIに安全にリクエストを送信する。
    """
    if not client:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Server API Client is not configured. (Missing GEMINI_API_KEY)"
        )

    # 🚀 モデル名マッピング処理 (404 NOT FOUNDエラー対策)
    # フロントエンドで選択されたモデル名を使用可能な最新/互換モデル名に置き換える
    model_name = request_data.model
    if "1.5-flash" in model_name or model_name == "gemini-flash":
        # 1.5-flash / gemini-flash が指定された場合、現行の 2.5-flash にマッピング
        actual_model = "gemini-2.5-flash"
    elif "1.5-pro" in model_name or model_name == "gemini-pro":
        # 1.5-pro / gemini-pro が指定された場合、現行の 2.5-pro にマッピング
        actual_model = "gemini-2.5-pro"
    else:
        # その他のモデル名 (ollamaなど) が指定された場合はそのまま使用（エラーになる可能性あり）
        actual_model = model_name

    try:
        # Gemini APIの呼び出し (修正したモデル名を使用)
        response = client.models.generate_content(
            model=actual_model,
            contents=request_data.prompt,
            config=genai.types.GenerateContentConfig(
                temperature=request_data.temperature
            )
        )
        # 返却するレスポンスを整形
        return {"response": response.text}
        
    except Exception as e:
        error_detail = str(e)
        # 404 NOT_FOUNDなど具体的なエラーをサーバーログに出力
        print(f"Gemini API Call Error: {error_detail}") 
        
        # クライアントには汎用的なエラーを返す
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An error occurred during AI generation. Please check the server logs."
        )

# Sarasina (Ollama経由またはローカルサーバー) 用のプロキシエンドポイント
@app.post("/api/sarasina")
async def sarasina_proxy(request_data: GeminiRequest):
    """
    SoftBank sarasina (Ollamaに手動登録したモデル等) へのプロキシ
    """
    # Ollama Native API (コンテキストサイズ制御のためこちらを使用)
    target_url = "http://localhost:11434/api/chat"

    try:
        response = requests.post(
            target_url,
            json={
                "model": request_data.model,
                "messages": [{"role": "user", "content": request_data.prompt}],
                "stream": False,
                "options": {
                    "temperature": request_data.temperature,
                    "num_ctx": 8192  # RAG用にコンテキストサイズを拡張 (デフォルト2048 -> 8192)
                }
            }
        )
        response.raise_for_status()
        data = response.json()
        return {"response": data["message"]["content"]}

    except Exception as e:
        print(f"Sarasina API Error: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Sarasina model error. Check Ollama/Local server. Error: {str(e)}"
        )