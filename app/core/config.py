from pydantic import Field
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    # 後端部署請用環境變數設定，勿把敏感資訊寫死
    DATABASE_URL: str = Field(..., description="PostgreSQL URL")
    JWT_SECRET: str = Field(..., description="JWT secret for signing tokens")
    JWT_EXPIRE_MINUTES: int = 60 * 24 * 30  # 30天
    # 允許的前端網域（逗號分隔）
    ALLOWED_ORIGINS: str = "https://emobot-plus.vercel.app,http://localhost:3000,http://localhost:5173"

    class Config:
        env_file = ".env"
        extra = "ignore"

settings = Settings()
