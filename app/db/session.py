# app/db/session.py
import os
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

# 讓本機的 .env 生效
try:
    from dotenv import load_dotenv, find_dotenv
    load_dotenv(find_dotenv())
except Exception:
    pass

# 讀取 DATABASE_URL 環境變數
DATABASE_URL = (
    os.getenv("DATABASE_URL")
    or os.getenv("POSTGRES_PRISMA_URL")
    or os.getenv("POSTGRES_URL")
    or ""
)

def process_supabase_url(url: str) -> str:
    """處理 Supabase URL，確保正確的 SSL 設定"""
    if not url:
        return "sqlite:///./app.db"
    
    # 如果是 PostgreSQL 連接字串
    if url.startswith(("postgres://", "postgresql://")):
        # ★ 關鍵修正：移除可能導致問題的參數處理，只保留基本的 sslmode
        if "sslmode=" not in url:
            # 簡單添加 sslmode=require
            separator = "&" if "?" in url else "?"
            url = f"{url}{separator}sslmode=require"
        
        # 記錄處理過程（用於調試）
        print(f"[INFO] Processed DATABASE_URL: {url[:50]}...")
        return url
    
    # 如果不是 PostgreSQL URL，直接返回
    return url

# 處理資料庫 URL
DATABASE_URL = process_supabase_url(DATABASE_URL)

# 如果仍未設定，回退到本機 SQLite
if not DATABASE_URL or DATABASE_URL.startswith("sqlite"):
    if not DATABASE_URL:
        print("[WARN] DATABASE_URL not set; falling back to local SQLite ./app.db")
        DATABASE_URL = "sqlite:///./app.db"

# 建立引擎
engine_kwargs = dict(pool_pre_ping=True)
connect_args = {}

if DATABASE_URL.startswith("sqlite"):
    # SQLite 設定
    connect_args["check_same_thread"] = False
elif DATABASE_URL.startswith(("postgres://", "postgresql://")):
    # PostgreSQL 設定 - 移除複雜的池化設定
    engine_kwargs.update({
        "pool_size": 5,
        "max_overflow": 10,
        "pool_timeout": 30,
        "pool_recycle": 1800,
    })

try:
    engine = create_engine(
        DATABASE_URL, 
        connect_args=connect_args, 
        **engine_kwargs,
        echo=False  # 設為 True 可查看 SQL 查詢
    )
    print(f"✅ Database engine created successfully")
except Exception as e:
    print(f"❌ Failed to create database engine: {e}")
    # 緊急回退到 SQLite
    print("[WARN] Falling back to SQLite")
    DATABASE_URL = "sqlite:///./app.db"
    engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False}, pool_pre_ping=True)

SessionLocal = sessionmaker(bind=engine, autocommit=False, autoflush=False)

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()