# backend/app/core/config.py
import os
import secrets

class Settings:
    # 強制要求 JWT_SECRET,沒有預設值
    JWT_SECRET = os.getenv("JWT_SECRET")
    
    if not JWT_SECRET:
        # 開發環境自動生成
        if os.getenv("ENV") == "development":
            JWT_SECRET = secrets.token_urlsafe(32)
            print(f"⚠️ 使用臨時 JWT_SECRET: {JWT_SECRET}")
        else:
            raise RuntimeError("JWT_SECRET 環境變數未設定!")
    
    JWT_ALG = "HS256"
    JWT_EXPIRE_MINUTES = int(os.getenv("JWT_EXPIRE_MINUTES", "129600"))
    DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./app.db")
    ALLOWED_ORIGINS = os.getenv(
        "ALLOWED_ORIGINS",
        "https://emobot-plus.vercel.app,http://localhost:5173,http://localhost:3000",
    )
    
    # OpenAI API Key 驗證
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    if not OPENAI_API_KEY:
        print("⚠️ OPENAI_API_KEY 未設定,聊天功能將無法使用")

settings = Settings()