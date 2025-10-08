# app/db/session.py
import os
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

try:
    from dotenv import load_dotenv, find_dotenv
    load_dotenv(find_dotenv())
except Exception:
    pass

# 讀取資料庫 URL
DATABASE_URL = (
    os.getenv("DATABASE_URL")
    or os.getenv("POSTGRES_PRISMA_URL")
    or os.getenv("POSTGRES_URL")
    or ""
)

def process_supabase_url(url: str) -> str:
    """處理 Supabase URL"""
    if not url:
        return "sqlite:///./app.db"
    
    if url.startswith(("postgres://", "postgresql://")):
        # 確保使用 SSL
        if "sslmode=" not in url:
            separator = "&" if "?" in url else "?"
            url = f"{url}{separator}sslmode=require"
        
        print(f"[INFO] Database URL processed (first 50 chars): {url[:50]}...")
        return url
    
    return url

DATABASE_URL = process_supabase_url(DATABASE_URL)

if not DATABASE_URL or DATABASE_URL.startswith("sqlite"):
    if not DATABASE_URL:
        print("[WARN] DATABASE_URL not set; using SQLite")
        DATABASE_URL = "sqlite:///./app.db"

# 建立引擎
engine_kwargs = dict(pool_pre_ping=True)
connect_args = {}

if DATABASE_URL.startswith("sqlite"):
    connect_args["check_same_thread"] = False
elif DATABASE_URL.startswith(("postgres://", "postgresql://")):
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
        echo=False
    )
    print(f"✅ Database engine created successfully")
except Exception as e:
    print(f"❌ Failed to create database engine: {e}")
    print("[WARN] Falling back to SQLite")
    DATABASE_URL = "sqlite:///./app.db"
    engine = create_engine(
        DATABASE_URL, 
        connect_args={"check_same_thread": False}, 
        pool_pre_ping=True
    )

SessionLocal = sessionmaker(bind=engine, autocommit=False, autoflush=False)

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()