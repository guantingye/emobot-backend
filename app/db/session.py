# app/db/session.py
import os
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

# 讓本機的 .env 生效；在雲端（Render）不會覆蓋系統環境變數
try:
    from dotenv import load_dotenv, find_dotenv
    load_dotenv(find_dotenv())
except Exception:
    pass

# 依序嘗試讀取各種常見變數名稱（方便你未來搬部署）
DATABASE_URL = (
    os.getenv("DATABASE_URL")
    or os.getenv("POSTGRES_PRISMA_URL")     # Vercel x Supabase 會給
    or os.getenv("POSTGRES_URL")            # Vercel 直連
    or ""
)

# 若仍未設定，回退到本機 SQLite（避免匯入期就炸掉；你也可以改成 raise）
if not DATABASE_URL:
    print("[WARN] DATABASE_URL not set; falling back to local SQLite ./app.db")
    DATABASE_URL = "sqlite:///./app.db"

# Supabase 走 postgres，務必確保 sslmode=require
if DATABASE_URL.startswith(("postgres://", "postgresql://")) and "sslmode=" not in DATABASE_URL:
    sep = "&" if "?" in DATABASE_URL else "?"
    DATABASE_URL = f"{DATABASE_URL}{sep}sslmode=require"

# SQLite 與 Postgres 的引擎建立方式略有不同
engine_kwargs = dict(pool_pre_ping=True, future=True)
connect_args = {}

if DATABASE_URL.startswith("sqlite"):
    # SQLite 需允許多執行緒
    connect_args["check_same_thread"] = False

engine = create_engine(DATABASE_URL, connect_args=connect_args, **engine_kwargs)

SessionLocal = sessionmaker(bind=engine, autocommit=False, autoflush=False)

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
