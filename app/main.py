# app/main.py
import os
from datetime import datetime, timedelta
from typing import Optional, List, Dict, Any

from fastapi import FastAPI, Depends, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from sqlalchemy import text, and_
from sqlalchemy.orm import Session

# Core imports
from app.core.config import settings
from app.core.security import create_access_token, get_current_user
from app.db.session import get_db, engine
from app.db.base import Base

# Models - 確保正確順序導入
from app.models.user import User
from app.models.assessment import Assessment  
from app.models.recommendation import Recommendation
from app.models.chat import ChatMessage
from app.models.mood import MoodRecord

# Chat router
from app.chat import router as chat_router

# Fallback recommendation engine
def _fallback_build_reco(user: Dict[str, Any] | None, assessment: Dict[str, Any] | None) -> Dict[str, Any]:
    empathy = insight = solution = cognitive = 0.25
    if assessment:
        enc = assessment.get("mbti_encoded")
        if isinstance(enc, (list, tuple)) and len(enc) >= 4:
            def norm(v):
                try:
                    v = float(v)
                    if v > 1: v = v / 100.0
                    return min(max(v, 0.0), 1.0)
                except Exception:
                    return 0.5
            E, N, T, J = [norm(v) for v in enc[:4]]
            empathy = 0.55 * (1 - T) + 0.25 * (1 - J) + 0.20 * E
            insight = 0.50 * N + 0.30 * (1 - T) + 0.20 * J
            solution = 0.45 * J + 0.30 * T + 0.25 * (1 - E)
            cognitive = 0.40 * T + 0.30 * (1 - N) + 0.30 * J

    raw = {"empathy": empathy, "insight": insight, "solution": solution, "cognitive": cognitive}
    ranked = [{"type": k, "score": round(v * 100, 2)} for k, v in sorted(raw.items(), key=lambda kv: kv[1], reverse=True)]
    return {"ok": True, "scores": raw, "ranked": ranked, "top": {"type": ranked[0]["type"], "score": ranked[0]["score"]}}

# ============================================================================
# FastAPI App - 超級簡單的設定
# ============================================================================

app = FastAPI(title="Emobot Backend", version="0.7.0")

# 最簡單的 CORS 設定 - 不使用任何自訂中介層
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://emobot-plus.vercel.app",
        "http://localhost:5173", 
        "http://localhost:3000"
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 啟動時建表
@app.on_event("startup")
async def startup():
    try:
        Base.metadata.create_all(bind=engine)
        print("✅ Database tables created")
    except Exception as e:
        print(f"❌ Startup error: {e}")

# 包含聊天路由
app.include_router(chat_router)

# ============================================================================
# Schemas  
# ============================================================================

class JoinRequest(BaseModel):
    pid: str
    nickname: Optional[str] = None

class AssessmentUpsert(BaseModel):
    mbti_raw: Optional[str] = None
    mbti_encoded: Optional[List[float]] = None
    step2_answers: Optional[List[Any]] = None
    step3_answers: Optional[List[Any]] = None
    step4_answers: Optional[List[Any]] = None
    submittedAt: Optional[datetime] = None

class MatchChoice(BaseModel):
    bot_type: str

class ChatMessageCreate(BaseModel):
    content: str
    role: str
    bot_type: Optional[str] = None
    mode: Optional[str] = "text"

class MoodRecordCreate(BaseModel):
    mood: str
    intensity: Optional[int] = None
    note: Optional[str] = None

# ============================================================================
# Routes - 超級簡化版本
# ============================================================================

@app.get("/")
async def root():
    return {"message": "Emobot Backend", "version": "0.7.0"}

@app.get("/api/health")
async def health():
    return {"ok": True, "time": datetime.utcnow().isoformat() + "Z"}

@app.get("/api/debug/db-test")
async def db_test(db: Session = Depends(get_db)):
    try:
        result = db.execute(text("SELECT 1")).scalar()
        user_count = db.query(User).count()
        return {"ok": True, "result": result, "user_count": user_count}
    except Exception as e:
        return {"ok": False, "error": str(e)}

# Auth - 最簡化版本
@app.post("/api/auth/join")
async def join(body: JoinRequest, db: Session = Depends(get_db)):
    try:
        pid = (body.pid or "").strip()
        if not pid:
            raise HTTPException(status_code=422, detail="pid required")

        user = db.query(User).filter(User.pid == pid).first()
        if not user:
            user = User(pid=pid, nickname=body.nickname)
            db.add(user)
            db.commit()
            db.refresh(user)

        token = create_access_token(user_id=user.id, pid=user.pid)
        return {
            "token": token,
            "user": {"id": user.id, "pid": user.pid, "nickname": user.nickname, "selected_bot": user.selected_bot}
        }
    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=str(e))

# Profile
@app.get("/api/user/profile")
async def profile(user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    try:
        return {"user": {"id": user.id, "pid": user.pid, "nickname": user.nickname, "selected_bot": user.selected_bot}}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Assessment
@app.post("/api/assessments/upsert")
async def upsert_assessment(body: AssessmentUpsert, user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    try:
        a = Assessment(
            user_id=user.id,
            mbti_raw=body.mbti_raw,
            mbti_encoded=body.mbti_encoded,
            step2_answers=body.step2_answers,
            step3_answers=body.step3_answers,
            step4_answers=body.step4_answers,
            submitted_at=body.submittedAt,
        )
        db.add(a)
        db.commit()
        return {"ok": True, "assessment_id": a.id}
    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/assessments/me")
async def my_assessment(user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    try:
        a = db.query(Assessment).filter(Assessment.user_id == user.id).order_by(Assessment.id.desc()).first()
        if not a:
            return {"assessment": None}
        return {"assessment": {"id": a.id, "mbti_raw": a.mbti_raw, "mbti_encoded": a.mbti_encoded}}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Matching
@app.post("/api/match/recommend")
async def recommend(user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    try:
        a = db.query(Assessment).filter(Assessment.user_id == user.id).order_by(Assessment.id.desc()).first()
        if not a:
            raise HTTPException(status_code=400, detail="No assessment found")

        result = _fallback_build_reco(
            {"id": user.id, "pid": user.pid},
            {"mbti_encoded": a.mbti_encoded, "step2_answers": a.step2_answers}
        )
        
        rec = Recommendation(
            user_id=user.id,
            assessment_id=a.id,
            selected_bot=result["top"]["type"],
            scores=result["scores"],
            ranked=result["ranked"],
        )
        db.add(rec)
        db.commit()

        return {"ok": True, "scores": result["scores"], "ranked": result["ranked"], "top": result["top"]}
    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/match/choose")
async def choose_bot(body: MatchChoice, user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    try:
        user.selected_bot = body.bot_type
        db.add(user)
        db.commit()
        return {"ok": True, "selected_bot": user.selected_bot}
    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/match/me")
async def my_match(user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    return {"selected_bot": user.selected_bot}

# Chat (Legacy)
@app.post("/api/chat/messages")
async def save_chat_message(body: ChatMessageCreate, user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    try:
        msg = ChatMessage(
            user_id=user.id,
            role=body.role,
            content=body.content,
            bot_type=body.bot_type,
            mode=body.mode,
        )
        db.add(msg)
        db.commit()
        return {"ok": True, "id": msg.id}
    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/chat/messages")
async def get_chat_messages(limit: int = Query(50), user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    try:
        rows = db.query(ChatMessage).filter(ChatMessage.user_id == user.id).limit(limit).all()
        return {"messages": [{"id": r.id, "role": r.role, "content": r.content} for r in rows]}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Mood
@app.post("/api/mood/records")
async def create_mood(body: MoodRecordCreate, user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    try:
        rec = MoodRecord(user_id=user.id, mood=body.mood, intensity=body.intensity, note=body.note)
        db.add(rec)
        db.commit()
        return {"ok": True, "id": rec.id}
    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/mood/records")
async def list_mood(days: int = Query(30), user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    try:
        since = datetime.utcnow() - timedelta(days=days)
        rows = db.query(MoodRecord).filter(and_(MoodRecord.user_id == user.id, MoodRecord.created_at >= since)).all()
        return {"records": [{"id": r.id, "mood": r.mood, "note": r.note} for r in rows]}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)