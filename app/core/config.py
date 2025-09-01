# app/core/config.py
from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv())

import os
class Settings:
    JWT_SECRET = os.getenv("JWT_SECRET", "dev-secret-change-me")
    JWT_ALG = os.getenv("JWT_ALG", "HS256")
    JWT_EXPIRE_MINUTES = int(os.getenv("JWT_EXPIRE_MINUTES", "129600"))
    DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./app.db")
    ALLOWED_ORIGINS = os.getenv(
        "ALLOWED_ORIGINS",
        "https://emobot-plus.vercel.app,http://localhost:5173,http://localhost:3000",
    )
settings = Settings()