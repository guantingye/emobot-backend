# app/main.py
from __future__ import annotations

import os
import re
from datetime import datetime, timedelta
from typing import Any

from fastapi import FastAPI, Depends, HTTPException, Query, Response, Request, Header
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy import text
from sqlalchemy.orm import Session
from jose import jwt, JWTError

# ---- 環境變數 ----
DATABASE_URL = os.getenv("DATABASE_URL")
JWT_SECRET = os.getenv("JWT_SECRET") or os.getenv("SECRET_KEY") or "dev-secret-change-me"
JWT_ALG = os.getenv("JWT_ALG", "HS256")
JWT_EXPIRE_MINUTES = int(os.getenv("JWT_EXPIRE_MINUTES", "129600"))  # 90天

# 🔧 修正 CORS 設定：使用更寬鬆的設定來解決問題
ALLOWED_ORIGINS = os.getenv(
    "ALLOWED_ORIGINS",
    "https://emobot-plus.vercel.app,http://localhost:5173,http://localhost:3000,https://*.vercel.app",
)

# ---- DB / Models（沿用你現有的專案結構）----
from app.db.session import get_db, engine  # type: ignore
from app.db.base import Base               # type: ignore

from app.models.user import User                    # type: ignore
from app.models.assessment import Assessment        # type: ignore
from app.models.recommendation import Recommendation# type: ignore
from app.models.chat import ChatMessage             # type: ignore
from app.models.mood import MoodRecord              # type: ignore

# ====================== 應用初始化與 CORS ======================
app = FastAPI(title="Emobot+ API", version="1.0.0")

# 關閉自動尾斜線轉向，避免 307/308 沒帶 CORS 標頭
app.router.redirect_slashes = False

# 🔧 更強化的 CORS 設定
_allowed_list = [o.strip() for o in ALLOWED_ORIGINS.split(",") if o.strip()]

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 🔧 暫時使用 * 來解決問題，後續可以改回特定域名
    allow_credentials=False,  # 🔧 改為 False，因為使用 Bearer token 而非 cookies
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS", "PATCH"],
    allow_headers=["*"],
    expose_headers=["*"],
    max_age=86400,
)

# 🔧 簡化的 CORS middleware，確保所有回應都有 CORS 標頭
@app.middleware("http")
async def add_cors_headers(request: Request, call_next):
    response = await call_next(request)
    
    # 為所有回應添加 CORS 標頭
    origin = request.headers.get("origin")
    if origin:
        response.headers["Access-Control-Allow-Origin"] = "*"  # 或者用 origin
        response.headers["Access-Control-Allow-Methods"] = "GET, POST, PUT, DELETE, OPTIONS, PATCH"
        response.headers["Access-Control-Allow-Headers"] = "*"
        response.headers["Access-Control-Expose-Headers"] = "*"
        response.headers["Access-Control-Max-Age"] = "86400"
    
    return response

# 🔧 確保所有 OPTIONS 請求都能正確處理
@app.options("/{full_path:path}")
async def handle_options(request: Request):
    origin = request.headers.get("origin")
    response = Response()
    if origin:
        response.headers["Access-Control-Allow-Origin"] = "*"
        response.headers["Access-Control-Allow-Methods"] = "GET, POST, PUT, DELETE, OPTIONS, PATCH"
        response.headers["Access-Control-Allow-Headers"] = "*"
        response.headers["Access-Control-Max-Age"] = "86400"
    return response

# 啟動建表（若你用 Alembic，可移除此段）
@app.on_event("startup")
def on_startup():
    Base.metadata.create_all(bind=engine)

# ====================== JWT 工具 ======================
def create_access_token_for_user(user: User) -> str:
    """產出與前端相容的 token：同時包含 sub 與 id"""
    exp = datetime.utcnow() + timedelta(minutes=JWT_EXPIRE_MINUTES)
    payload = {
        "sub": str(user.id),   # 標準
        "id": user.id,         # 舊前端也會用到
        "pid": user.pid,
        "nickname": user.nickname,
        "role": "user",
        "exp": exp,
        "iat": datetime.utcnow(),
    }
    return jwt.encode(payload, JWT_SECRET, algorithm=JWT_ALG)

def get_current_user(
    db: Session = Depends(get_db),
    authorization: str | None = Header(None),
) -> User:
    if not authorization or not authorization.lower().startswith("bearer "):
        raise HTTPException(status_code=401, detail="Not authenticated")
    token = authorization.split(" ", 1)[1]

    # 解析 JWT：容忍多種 claim 名稱
    try:
        claims = jwt.decode(token, JWT_SECRET, algorithms=[JWT_ALG])
        uid_raw = claims.get("sub") or claims.get("id") or claims.get("user_id")
        user_id = int(uid_raw) if uid_raw is not None else None
    except JWTError:
        raise HTTPException(status_code=401, detail="Invalid token")

    if not user_id:
        raise HTTPException(status_code=401, detail="Invalid token payload")

    user = db.get(User, user_id)
    if not user:
        raise HTTPException(status_code=401, detail="User not found")
    return user

# ====================== 健康檢查 ======================
@app.get("/api/health")
def health():
    try:
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        db_status = "connected"
    except Exception as e:
        db_status = f"error: {e}"
    return {
        "ok": True,
        "time": datetime.utcnow().isoformat() + "Z",
        "database": db_status,
        "allowed_origins": _allowed_list,
    }

# ====================== Auth ======================
@app.post("/api/auth/join")
def join(payload: dict, db: Session = Depends(get_db)):
    """
    body: { "pid": "12AB", "nickname": "ting" }
    回傳：{ token, user }
    """
    pid = (payload.get("pid") or "").strip()
    nickname = (payload.get("nickname") or "").strip()
    if not pid:
        raise HTTPException(status_code=422, detail="pid required")

    user = db.query(User).filter(User.pid == pid).first()
    if not user:
        user = User(pid=pid, nickname=nickname or "user")
        db.add(user); db.commit(); db.refresh(user)
    else:
        if nickname and nickname != user.nickname:
            user.nickname = nickname
            db.commit(); db.refresh(user)

    token = create_access_token_for_user(user)
    return {
        "token": token,
        "user": {
            "id": user.id,
            "pid": user.pid,
            "nickname": user.nickname,
            "selected_bot": user.selected_bot,
        },
    }

# ====================== User Profile（前端用） ======================
@app.get("/api/user/profile")
def user_profile(user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    last_assessment = (
        db.query(Assessment)
        .filter(Assessment.user_id == user.id)
        .order_by(Assessment.id.desc())
        .first()
    )
    last_reco = (
        db.query(Recommendation)
        .filter(Recommendation.user_id == user.id)
        .order_by(Recommendation.id.desc())
        .first()
    )
    return {
        "id": user.id,
        "pid": user.pid,
        "nickname": user.nickname,
        "selected_bot": user.selected_bot,
        "latest_assessment": ({
            "id": last_assessment.id,
            "mbti_raw": last_assessment.mbti_raw,
            "mbti_encoded": last_assessment.mbti_encoded,
            "submitted_at": last_assessment.submitted_at.isoformat() if last_assessment and last_assessment.submitted_at else None,
        } if last_assessment else None),
        "latest_recommendation": ({
            "top_bot": last_reco.top_bot,
            "scores": last_reco.scores,
            "features": last_reco.features,
        } if last_reco else None),
    }

# ====================== Assessments Upsert ======================
@app.post("/api/assessments/upsert")
def upsert_assessment(
    body: dict,
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """
    🔧 修正：加強錯誤處理和日誌記錄
    支援字段：
      mbti_raw: "ENTP"
      mbti_encoded: [1,1,1,1] 或 dict
      step2Answers / step3Answers / step4Answers: dict
      ai_preference: dict
      submittedAt: ISO8601
    """
    print(f"📝 Received assessment data: {body}")  # 🔧 加入除錯日誌
    
    try:
        mbti_raw = body.get("mbti_raw")
        mbti_encoded = body.get("mbti_encoded")
        step2 = body.get("step2Answers")
        step3 = body.get("step3Answers")
        step4 = body.get("step4Answers")
        ai_pref = body.get("ai_preference")
        submitted_at = body.get("submittedAt")

        # 解析時間
        dt = None
        if isinstance(submitted_at, str):
            try:
                dt = datetime.fromisoformat(submitted_at.replace("Z", "+00:00"))
            except Exception as e:
                print(f"⚠️ Time parse error: {e}")
                dt = None

        record = Assessment(
            user_id=user.id,
            mbti_raw=(mbti_raw.upper().strip() if isinstance(mbti_raw, str) else None),
            mbti_encoded={"encoded": mbti_encoded} if isinstance(mbti_encoded, list) else (mbti_encoded if isinstance(mbti_encoded, dict) else None),
            step2_answers=step2 if isinstance(step2, dict) else None,
            step3_answers=step3 if isinstance(step3, dict) else None,
            step4_answers=step4 if isinstance(step4, dict) else None,
            ai_preference=ai_pref if isinstance(ai_pref, dict) else None,
            submitted_at=dt,
        )
        
        db.add(record)
        db.commit()
        db.refresh(record)

        result = {
            "id": record.id,
            "mbti_raw": record.mbti_raw,
            "mbti_encoded": record.mbti_encoded,
            "submitted_at": record.submitted_at.isoformat() if record.submitted_at else None,
        }
        
        print(f"✅ Assessment saved: {result}")  # 🔧 加入成功日誌
        return result
        
    except Exception as e:
        print(f"❌ Assessment save error: {e}")  # 🔧 加入錯誤日誌
        raise HTTPException(status_code=500, detail=f"Database error: {str(e)}")

# ====================== Recommendation（示例） ======================
VALID_BOTS = {"empathy", "insight", "solution", "cognitive"}

def _score_bots(mbti_raw: str | None) -> dict[str, float]:
    base = {b: 0.0 for b in VALID_BOTS}
    m = (mbti_raw or "").upper()
    if "F" in m: base["empathy"] += 1.0
    if "N" in m: base["insight"] += 1.0
    if "T" in m: base["solution"] += 1.0
    if "P" in m: base["cognitive"] += 1.0
    return base

@app.post("/api/match/recommend")
def recommend(user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    assessment = (
        db.query(Assessment)
        .filter(Assessment.user_id == user.id)
        .order_by(Assessment.id.desc())
        .first()
    )
    if not assessment:
        raise HTTPException(status_code=400, detail="No assessment found")

    scores = _score_bots(assessment.mbti_raw)
    top = max(scores, key=scores.get)

    rec = Recommendation(
        user_id=user.id,
        assessment_id=assessment.id,
        scores=scores,
        top_bot=top,
        features={"rule": "mbti_rule_v1"},
    )
    db.add(rec)
    db.commit()

    return {"scores": scores, "top": top, "features": {"rule": "mbti_rule_v1"}}

@app.post("/api/match/choose")
def choose_bot(payload: dict, user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    bot_type = (payload.get("botType") or "").strip().lower()
    if bot_type not in VALID_BOTS:
        raise HTTPException(status_code=400, detail="Invalid botType")
    user.selected_bot = bot_type
    db.commit()
    return {"ok": True, "selected_bot": bot_type}

# ====================== Chat ======================
@app.post("/api/chat/messages")
def create_message(body: dict, user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    msg = ChatMessage(
        user_id=user.id,
        message_type=(body.get("message_type") or "user")[:8],
        bot_type=(body.get("bot_type") or None),
        content=str(body.get("content") or ""),
        user_mood=(body.get("user_mood") or None),
        mood_intensity=int(body.get("mood_intensity")) if body.get("mood_intensity") is not None else None,
    )
    db.add(msg); db.commit(); db.refresh(msg)
    return {
        "id": msg.id,
        "message_type": msg.message_type,
        "bot_type": msg.bot_type,
        "content": msg.content,
        "user_mood": msg.user_mood,
        "mood_intensity": msg.mood_intensity,
        "created_at": msg.created_at.isoformat(),
    }

@app.get("/api/chat/messages")
def list_messages(limit: int = Query(50, ge=1, le=200), user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    rows = (
        db.query(ChatMessage)
        .filter(ChatMessage.user_id == user.id)
        .order_by(ChatMessage.id.desc())
        .limit(limit)
        .all()
    )
    return [
        {
            "id": r.id,
            "message_type": r.message_type,
            "bot_type": r.bot_type,
            "content": r.content,
            "user_mood": r.user_mood,
            "mood_intensity": r.mood_intensity,
            "created_at": r.created_at.isoformat(),
        }
        for r in reversed(rows)
    ]

# ====================== Mood ======================
@app.post("/api/mood/records")
def create_mood(body: dict, user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    rec = MoodRecord(
        user_id=user.id,
        mood=str(body.get("mood") or ""),
        intensity=int(body.get("intensity")) if body.get("intensity") is not None else None,
        note=(body.get("note") or None),
    )
    db.add(rec); db.commit(); db.refresh(rec)
    return {
        "id": rec.id,
        "mood": rec.mood,
        "intensity": rec.intensity,
        "note": rec.note,
        "created_at": rec.created_at.isoformat(),
    }

@app.get("/api/mood/records")
def list_mood(days: int = Query(30, ge=1, le=180), user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    rows = (
        db.query(MoodRecord)
        .filter(MoodRecord.user_id == user.id)
        .order_by(MoodRecord.id.desc())
        .limit(500)
        .all()
    )
    return [
        {
            "id": r.id,
            "mood": r.mood,
            "intensity": r.intensity,
            "note": r.note,
            "created_at": r.created_at.isoformat(),
        }
        for r in reversed(rows)
    ]