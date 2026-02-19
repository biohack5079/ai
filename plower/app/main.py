import os
import asyncio
from fastapi import FastAPI, Request, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import requests
from google import genai
from dotenv import load_dotenv

# .envロード
load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
client = None
AVAILABLE_GEMINI_MODELS = [] # 利用可能なモデルリストを保持

app = FastAPI()

# --- 🚀 起動時に利用可能なモデルを動的に取得 ---
@app.on_event("startup")
async def startup_event():
    global client, AVAILABLE_GEMINI_MODELS
    if GEMINI_API_KEY:
        try:
            client = genai.Client(api_key=GEMINI_API_KEY)
            # models.list() で利用可能な全モデルを取得
            # names は "models/gemini-2.5-flash" の形式なので "models/" を除去
            models = client.models.list()
            AVAILABLE_GEMINI_MODELS = [m.name.replace("models/", "") for m in models]
            print(f"✅ Loaded Gemini models: {AVAILABLE_GEMINI_MODELS}")
        except Exception as e:
            print(f"❌ Failed to load Gemini models: {e}")

# --- 🧠 モデル名のインテリジェント・マッピング関数 ---
def map_model_name(user_model: str) -> str:
    """
    ユーザーが指定したモデル名（例: 'flash', 'gemini-1.5-pro'）を
    現在利用可能な最新の正式モデル名に変換する
    """
    if not AVAILABLE_GEMINI_MODELS:
        return user_model # リストが空ならそのまま返す

    # 1. 完全一致があればそれを使用
    if user_model in AVAILABLE_GEMINI_MODELS:
        return user_model

    # 2. キーワード（flash, pro）が含まれるモデルをフィルタリング
    # 例: "gemini-flash" -> "flash" で検索
    # Gemini 3系やLiteにも対応 (大文字小文字無視)
    search_keyword = user_model.lower().replace("gemini-", "").replace("1.5-", "").replace("2.5-", "").replace("3.0-", "").replace("3-", "")
    
    candidates = [
        m for m in AVAILABLE_GEMINI_MODELS 
        if search_keyword in m.lower() and "vision" not in m.lower() # vision専用モデル等は除外
    ]

    if candidates:
        # 文字列順でソートして最新（例: 2.5 > 1.5）を選択
        return sorted(candidates)[-1]

    # 3. 見つからなければ（Ollama用など）入力をそのまま返す
    return user_model

# --- CORS設定 ---
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class GeminiRequest(BaseModel):
    model: str = "gemini-flash" # フロントからの抽象的な指定
    prompt: str
    temperature: float = 0.1
    
@app.get("/")
def read_root():
    return {"status": "online", "available_models_count": len(AVAILABLE_GEMINI_MODELS)}

# Gemini APIを中継するプロキシエンドポイント
@app.post("/api/gemini_proxy")
async def gemini_proxy(request_data: GeminiRequest):
    if not client:
        raise HTTPException(status_code=503, detail="Gemini Client not initialized.")

    # ✨ ここで動的マッピングを適用
    actual_model = map_model_name(request_data.model)
    print(f"🔀 Mapping: {request_data.model} -> {actual_model}")

    max_retries = 3
    for attempt in range(max_retries + 1):
        try:
            response = client.models.generate_content(
                model=actual_model,
                contents=request_data.prompt,
                config=genai.types.GenerateContentConfig(
                    temperature=request_data.temperature
                )
            )
            return {"response": response.text, "model_used": actual_model}
            
        except Exception as e:
            error_str = str(e)
            # 429 Resource Exhausted エラーのハンドリング
            if ("429" in error_str or "RESOURCE_EXHAUSTED" in error_str) and attempt < max_retries:
                wait_time = 2 * (2 ** attempt) # 指数バックオフ: 2s, 4s, 8s
                print(f"⚠️ Quota exceeded for {actual_model}. Retrying in {wait_time}s... (Attempt {attempt + 1}/{max_retries})")
                await asyncio.sleep(wait_time)
                continue
            
            print(f"Gemini API Error: {e}")
            status_code = 429 if "429" in error_str else 500
            raise HTTPException(status_code=status_code, detail=str(e))

# Sarasina (Ollama経由またはローカルサーバー) 用のプロキシエンドポイント
@app.post("/api/sarasina")
async def sarasina_proxy(request_data: GeminiRequest):
    # (Sarasinaのロジックは変更なしでOK)
    target_url = "http://localhost:11434/api/chat"
    try:
        response = requests.post(
            target_url,
            json={
                "model": request_data.model,
                "messages": [{"role": "user", "content": request_data.prompt}],
                "stream": False,
                "options": {"temperature": request_data.temperature, "num_ctx": 8192}
            }
        )
        response.raise_for_status()
        return {"response": response.json()["message"]["content"]}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ollama Error: {str(e)}")