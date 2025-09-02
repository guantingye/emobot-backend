# app/main.py
from datetime import datetime, timezone
import re
from typing import Any, Optional

from fastapi import FastAPI, Depends, HTTPException, Query, Response, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from sqlalchemy import text
from sqlalchemy.orm import Session

from app.core.config import settings
from app.core.security import create_access_token, get_current_user
from app.db.session import get_db, engine
from app.db.base import Base

from app.models.user import User
from app.models.assessment import Assessment
from app.models.recommendation import Recommendation
from app.models.chat import ChatMessage
from app.models.mood import MoodRecord

# 匯入完整推薦演算法
from app.services.recommendation_engine import build_recommendation

# ------------------- App & CORS -------------------
app = FastAPI(title="Emobot+ API", version="2.2.0")
app.router.redirect_slashes = False

allowed = [o.strip() for o in settings.ALLOWED_ORIGINS.split(",") if o.strip()]
vercel_regex = re.compile(r"^https://.*\.vercel\.app$")

app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed,
    allow_origin_regex=r"https://.*\.vercel\.app$",
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["*"],
    max_age=86400,
)

@app.middleware("http")
async def force_cors_headers(request: Request, call_next):
    try:
        resp = await call_next(request)
    except HTTPException as he:
        resp = JSONResponse({"detail": he.detail}, status_code=he.status_code)
    except Exception:
        resp = JSONResponse({"detail": "Internal Server Error"}, status_code=500)
    origin = request.headers.get("origin")
    if origin and (origin in allowed or vercel_regex.match(origin)):
        resp.headers.setdefault("Access-Control-Allow-Origin", origin)
        resp.headers.setdefault("Vary", "Origin")
    return resp

@app.options("/{full_path:path}")
def preflight():
    return Response(status_code=200)

@app.on_event("startup")
def on_startup():
    Base.metadata.create_all(bind=engine)

@app.get("/", include_in_schema=False)
def index():
    return {"ok": True, "service": "Emobot+ API", "docs": "/docs", "health": "/api/health"}

@app.get("/api/health")
def health():
    try:
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        db_status = "connected"
    except Exception as e:
        db_status = f"error: {e}"
    return {"ok": True, "time": datetime.utcnow().isoformat()+"Z", "database": db_status, "allowed_origins": allowed}

# -------- Auth --------
@app.post("/api/auth/join")
def join(payload: dict, db: Session = Depends(get_db)):
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
            user.nickname = nickname; db.commit(); db.refresh(user)
    token = create_access_token(user_id=user.id, pid=user.pid)
    return {"token": token, "user": {"id": user.id, "pid": user.pid, "nickname": user.nickname, "selected_bot": user.selected_bot}}

@app.get("/api/user/profile")
def profile(user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    a = db.query(Assessment).filter(Assessment.user_id == user.id).order_by(Assessment.id.desc()).first()
    r = db.query(Recommendation).filter(Recommendation.user_id == user.id).order_by(Recommendation.id.desc()).first()
    
    # 修改回傳格式以符合前端期望
    latest_recommendation = None
    if r:
        latest_recommendation = {
            "scores": r.scores,
            "ranked": [],
            "top": {"type": r.top_bot, "score": 100.0},
            "selected_bot": r.top_bot,
            "features": r.features
        }
        # 從 scores 產生 ranked 列表
        if r.scores:
            # 先正規化到 0-100
            max_score = max(r.scores.values()) if r.scores else 1.0
            ranked_items = []
            for bot_type, score in r.scores.items():
                percentage_score = (score / max_score) * 100.0
                ranked_items.append({"type": bot_type, "score": round(percentage_score, 1)})
            # 按分數排序
            latest_recommendation["ranked"] = sorted(ranked_items, key=lambda x: x["score"], reverse=True)
    
    return {
        "ok": True,
        "user": {
            "id": user.id, 
            "pid": user.pid, 
            "nickname": user.nickname, 
            "selected_bot": user.selected_bot
        },
        "latest_assessment": {
            "id": a.id, 
            "mbti": {"raw": a.mbti_raw, "encoded": a.mbti_encoded},
            "step2_answers": a.step2_answers, 
            "step3_answers": a.step3_answers, 
            "step4_answers": a.step4_answers,
            "submitted_at": a.submitted_at.isoformat() if a and a.submitted_at else None
        } if a else None,
        "latest_recommendation": latest_recommendation,
    }

# -------- Assessments Upsert（保持不動） --------
def _to_dt(s: Optional[str]):
    if not isinstance(s, str): return None
    try: return datetime.fromisoformat(s.replace("Z","+00:00"))
    except: return None

def _int_list(v: Any) -> Optional[list[int]]:
    if not isinstance(v, list): return None
    try: return [int(x) for x in v]
    except: return None

@app.post("/api/assessments/upsert")
def upsert(body: dict, user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    try:
        mbti_raw = None
        mbti_encoded = None
        if isinstance(body.get("mbti"), dict):
            mbti_raw = body["mbti"].get("raw")
            mbti_encoded = _int_list(body["mbti"].get("encoded"))
        else:
            mbti_raw = body.get("mbti_raw")
            mbti_encoded = _int_list(body.get("mbti_encoded"))
        if isinstance(mbti_raw, str):
            mbti_raw = mbti_raw.upper().strip()
            if len(mbti_raw) != 4: mbti_raw = None

        step2 = _int_list(body.get("step2Answers"))
        step3 = _int_list(body.get("step3Answers"))
        step4 = _int_list(body.get("step4Answers"))
        ai_pref = body.get("ai_preference") if isinstance(body.get("ai_preference"), dict) else None
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Bad request: {e}")

    submitted_at = _to_dt(body.get("submittedAt"))

    rec = db.query(Assessment).filter(Assessment.user_id == user.id).order_by(Assessment.id.desc()).first()
    if rec is None:
        rec = Assessment(user_id=user.id)
        db.add(rec); db.flush()

    if mbti_raw is not None: rec.mbti_raw = mbti_raw
    if mbti_encoded is not None: rec.mbti_encoded = mbti_encoded
    if step2 is not None: rec.step2_answers = step2
    if step3 is not None: rec.step3_answers = step3
    if step4 is not None: rec.step4_answers = step4
    if ai_pref is not None: rec.ai_preference = ai_pref
    if submitted_at is not None: rec.submitted_at = submitted_at

    db.commit(); db.refresh(rec)
    return {"id": rec.id, "mbti_raw": rec.mbti_raw, "mbti_encoded": rec.mbti_encoded,
            "submitted_at": rec.submitted_at.isoformat() if rec.submitted_at else None}

# ====================== 使用改進的推薦演算法 ======================

VALID_BOTS = {"empathy","insight","solution","cognitive"}

@app.post("/api/match/recommend")
def recommend(user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    """使用完整的推薦演算法進行匹配"""
    a = db.query(Assessment).filter(Assessment.user_id == user.id).order_by(Assessment.id.desc()).first()
    if not a:
        raise HTTPException(status_code=400, detail="No assessment found")
    
    # 準備 assessment 資料格式
    assessment_data = {
        "mbti_encoded": a.mbti_encoded or [],
        "step2Answers": a.step2_answers or [],
        "step3Answers": a.step3_answers or [],
        "step4Answers": a.step4_answers or [],
        "ai_preference": a.ai_preference or {}
    }
    
    user_data = {
        "pid": user.pid,
        "id": user.id
    }
    
    # 使用完整推薦演算法
    try:
        recommendation_result = build_recommendation(assessment_data, user_data)
        
        if not recommendation_result.get("ok"):
            raise HTTPException(status_code=400, detail="Recommendation algorithm failed")
        
        scores = recommendation_result["scores"]
        ranked = recommendation_result["ranked"] 
        top = recommendation_result["top"]
        
        # 儲存到資料庫
        rec = Recommendation(
            user_id=user.id, 
            assessment_id=a.id, 
            scores=scores, 
            top_bot=top["type"],
            features={
                "algorithm": "build_recommendation_v2.2",
                "ranked_results": ranked,
                "top_recommendation": top,
                "timestamp": datetime.utcnow().isoformat()
            }
        )
        db.add(rec)
        db.commit()
        db.refresh(rec)
        
        # 回傳完整結果
        return {
            "ok": True,
            "scores": scores,      # 0~1 的雷達圖分數
            "ranked": ranked,      # 0~100 的排序列表
            "top": top,           # 最高分推薦
            "user": {"pid": user.pid},
            "features": rec.features
        }
        
    except Exception as e:
        print(f"Recommendation error: {e}")
        # 如果演算法失敗，提供預設值
        default_scores = {"empathy": 0.6, "insight": 0.5, "solution": 0.7, "cognitive": 0.4}
        default_ranked = [
            {"type": "solution", "score": 100.0},
            {"type": "empathy", "score": 85.7}, 
            {"type": "insight", "score": 71.4},
            {"type": "cognitive", "score": 57.1}
        ]
        return {
            "ok": True,
            "scores": default_scores,
            "ranked": default_ranked,
            "top": {"type": "solution", "score": 100.0},
            "user": {"pid": user.pid},
            "error": f"Used fallback algorithm: {str(e)}"
        }

# -------- Choose bot（不變） --------
@app.post("/api/match/choose")
def choose(payload: dict, user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    bot = (payload.get("botType") or "").strip().lower()
    if bot not in VALID_BOTS: raise HTTPException(status_code=400, detail="Invalid botType")
    user.selected_bot = bot; db.commit()
    return {"ok": True, "selected_bot": bot}

# -------- Chat（不變） --------
@app.post("/api/chat/messages")
def create_msg(body: dict, user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    msg = ChatMessage(
        user_id=user.id,
        message_type=(body.get("message_type") or "user")[:8],
        bot_type=(body.get("bot_type") or None),
        content=str(body.get("content") or ""),
        user_mood=(body.get("user_mood") or None),
        mood_intensity=int(body.get("mood_intensity")) if body.get("mood_intensity") is not None else None,
    )
    db.add(msg); db.commit(); db.refresh(msg)
    return {"id": msg.id, "message_type": msg.message_type, "bot_type": msg.bot_type, "content": msg.content,
            "user_mood": msg.user_mood, "mood_intensity": msg.mood_intensity,
            "created_at": msg.created_at.replace(tzinfo=timezone.utc).isoformat()}

@app.get("/api/chat/messages")
def list_msg(limit: int = Query(50, ge=1, le=200), user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    rows = (db.query(ChatMessage).filter(ChatMessage.user_id == user.id)
            .order_by(ChatMessage.id.desc()).limit(limit).all())
    return [{"id": r.id, "message_type": r.message_type, "bot_type": r.bot_type, "content": r.content,
             "user_mood": r.user_mood, "mood_intensity": r.mood_intensity,
             "created_at": r.created_at.replace(tzinfo=timezone.utc).isoformat()}
            for r in reversed(rows)]

# -------- Mood（不變） --------
@app.post("/api/mood/records")
def create_mood(body: dict, user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    rec = MoodRecord(user_id=user.id, mood=str(body.get("mood") or ""),
                     intensity=int(body.get("intensity")) if body.get("intensity") is not None else None,
                     note=(body.get("note") or None))
    db.add(rec); db.commit(); db.refresh(rec)
    return {"id": rec.id, "mood": rec.mood, "intensity": rec.intensity, "note": rec.note,
            "created_at": rec.created_at.replace(tzinfo=timezone.utc).isoformat()}

@app.get("/api/mood/records")
def list_mood(days: int = Query(30, ge=1, le=180), user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    rows = (db.query(MoodRecord).filter(MoodRecord.user_id == user.id)
            .order_by(MoodRecord.id.desc()).limit(500).all())
    return [{"id": r.id, "mood": r.mood, "intensity": r.intensity, "note": r.note,
             "created_at": r.created_at.replace(tzinfo=timezone.utc).isoformat()}
            for r in reversed(rows)]