# app/main.py
from __future__ import annotations

import os
import re
from datetime import datetime, timedelta
from typing import Optional, List, Dict, Any

from fastapi import FastAPI, Depends, HTTPException, Query, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from sqlalchemy import text, and_
from sqlalchemy.orm import Session

# ---- App internals (沿用你原架構) ----
from app.core.config import settings
from app.core.security import create_access_token, get_current_user
from app.db.session import get_db, engine
from app.db.base import Base

from app.models.user import User
from app.models.assessment import Assessment
from app.models.recommendation import Recommendation
from app.models.chat import ChatMessage
from app.models.mood import MoodRecord

# ---- Optional: 外部推薦引擎，失敗時走 fallback ----
try:
    from app.services.recommendation_engine import recommend_endpoint_payload as _build_reco  # type: ignore
except Exception:
    _build_reco = None  # type: ignore


def _fallback_build_reco(user: Dict[str, Any] | None, assessment: Dict[str, Any] | None) -> Dict[str, Any]:
    """
    簡單、可重現的回退推薦：由 mbti_encoded[4] 推出四型分數（0~1），回傳 0~100 的排序結果。
    """
    empathy = insight = solution = cognitive = 0.25
    if assessment:
        enc = assessment.get("mbti_encoded")
        if isinstance(enc, (list, tuple)) and len(enc) >= 4:
            def norm(v):
                try:
                    v = float(v)
                    if v > 1:  # 可能是 0~100
                        v = v / 100.0
                    return min(max(v, 0.0), 1.0)
                except Exception:
                    return 0.5
            E, N, T, J = [norm(v) for v in enc[:4]]
            empathy = 0.55 * (1 - T) + 0.25 * (1 - J) + 0.20 * E
            insight = 0.50 * N + 0.30 * (1 - T) + 0.20 * J
            solution = 0.45 * J + 0.30 * T + 0.25 * (1 - E)
            cognitive = 0.40 * T + 0.30 * (1 - N) + 0.30 * J

    raw = {
        "empathy": float(min(max(empathy, 0.0), 1.0)),
        "insight": float(min(max(insight, 0.0), 1.0)),
        "solution": float(min(max(solution, 0.0), 1.0)),
        "cognitive": float(min(max(cognitive, 0.0), 1.0)),
    }
    ranked = [{"type": k, "score": round(v * 100, 2)} for k, v in sorted(raw.items(), key=lambda kv: kv[1], reverse=True)]
    top = {"type": ranked[0]["type"], "score": ranked[0]["score"]}
    return {
        "ok": True,
        "user": {"pid": (user or {}).get("pid")} if user else None,
        "scores": raw,
        "ranked": ranked,
        "top": top,
        "algorithm_version": "fallback_v1",
        "params": {},
    }


def build_recommendation_payload(user: Dict[str, Any] | None, assessment: Dict[str, Any] | None) -> Dict[str, Any]:
    if callable(_build_reco):
        try:
            return _build_reco(user=user, assessment=assessment)  # type: ignore
        except Exception:
            return _fallback_build_reco(user, assessment)
    return _fallback_build_reco(user, assessment)


# -----------------------------------------------------------------------------
# FastAPI App
# -----------------------------------------------------------------------------

app = FastAPI(title="Emobot Backend", version="0.5.0")

# ---- CORS（官方 + 強化補丁）----
ALLOWED = getattr(settings, "ALLOWED_ORIGINS", os.getenv(
    "ALLOWED_ORIGINS",
    "https://emobot-plus.vercel.app,http://localhost:5173,http://localhost:3000"
))


def _parse_allowed(origins_str: str) -> List[str]:
    out: List[str] = []
    for s in (origins_str or "").split(","):
        s = s.strip()
        if not s or s in ("*", "null"):
            continue
        out.append(s)
    return out


_ALLOWED_ORIGINS = _parse_allowed(ALLOWED)
_VERCEL_REGEX_STR = r"^https://.*\.vercel\.app$"
_VERCEL_REGEX = re.compile(_VERCEL_REGEX_STR, re.IGNORECASE)

# 1) 官方 middleware（處理多數情形）
app.add_middleware(
    CORSMiddleware,
    allow_origins=_ALLOWED_ORIGINS,
    allow_origin_regex=_VERCEL_REGEX_STR,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["*"],
)


# 2) 自訂補丁：確保錯誤時也帶 CORS，並正確處理預檢 OPTIONS
@app.middleware("http")
async def _force_cors_headers(request: Request, call_next):
    origin = request.headers.get("origin")
    is_allowed = bool(origin and (origin in _ALLOWED_ORIGINS or _VERCEL_REGEX.match(origin or "")))

    # 預檢：直接 204
    if request.method.upper() == "OPTIONS":
        acrm = request.headers.get("access-control-request-method", "GET,POST,PUT,PATCH,DELETE,OPTIONS")
        acrh = request.headers.get("access-control-request-headers", "Authorization,Content-Type,X-Requested-With")
        headers = {
            "Access-Control-Allow-Origin": origin if is_allowed else "",
            "Access-Control-Allow-Credentials": "true" if is_allowed else "false",
            "Access-Control-Allow-Methods": acrm,
            "Access-Control-Allow-Headers": acrh,
            "Access-Control-Max-Age": "86400",
            "Access-Control-Expose-Headers": "*",
            "Vary": "Origin",
        }
        return Response(status_code=204, headers=headers)

    # 一般請求：就算內層丟錯，也改包成 JSONResponse 再補 CORS
    try:
        resp = await call_next(request)
    except HTTPException as he:
        resp = JSONResponse({"detail": he.detail}, status_code=he.status_code)
    except Exception:
        resp = JSONResponse({"detail": "Internal Server Error"}, status_code=500)

    if is_allowed:
        resp.headers.setdefault("Access-Control-Allow-Origin", origin)
        resp.headers.setdefault("Access-Control-Allow-Credentials", "true")
        resp.headers.setdefault("Access-Control-Expose-Headers", "*")
        vary = resp.headers.get("Vary")
        resp.headers["Vary"] = "Origin" if not vary else (vary if "Origin" in vary else f"{vary}, Origin")
    return resp


# ---- 啟動時建表（若你用 Alembic 可拿掉）----
@app.on_event("startup")
def on_startup():
    Base.metadata.create_all(bind=engine)


# -----------------------------------------------------------------------------
# Schemas
# -----------------------------------------------------------------------------

class JoinRequest(BaseModel):
    pid: str = Field(..., min_length=1, max_length=50)
    nickname: Optional[str] = Field(default=None, max_length=100)


class AssessmentUpsert(BaseModel):
    mbti_raw: Optional[str] = None
    mbti_encoded: Optional[List[float]] = None
    step2_answers: Optional[List[Any]] = None
    step3_answers: Optional[List[Any]] = None
    step4_answers: Optional[List[Any]] = None
    ai_preference: Optional[Dict[str, Any]] = None
    submittedAt: Optional[datetime] = None
    # ★ 新增：是否為重新測驗
    is_retest: Optional[bool] = False


class MatchChoice(BaseModel):
    bot_type: str = Field(..., description="empathy | insight | solution | cognitive")


class ChatSendRequest(BaseModel):
    message: str
    bot_type: Optional[str] = None
    mode: Optional[str] = "text"


class ChatMessageCreate(BaseModel):
    content: str
    role: str = Field(..., description="user | ai")
    bot_type: Optional[str] = None
    mode: Optional[str] = "text"
    user_mood: Optional[str] = None
    mood_intensity: Optional[int] = None


class MoodRecordCreate(BaseModel):
    mood: str
    intensity: Optional[int] = None
    note: Optional[str] = None


# -----------------------------------------------------------------------------
# Health & Debug
# -----------------------------------------------------------------------------

@app.get("/api/health")
def health():
    return {"ok": True, "time": datetime.utcnow().isoformat() + "Z"}


@app.get("/api/debug/db-test")
def db_test(db: Session = Depends(get_db)):
    try:
        db.execute(text("select 1"))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"DB not ready: {e}")
    return {"ok": True}


# -----------------------------------------------------------------------------
# Auth & Profile
# -----------------------------------------------------------------------------

def _auth_join(body: JoinRequest, db: Session):
    pid = (body.pid or "").strip()
    if not pid:
        raise HTTPException(status_code=422, detail="pid is required")

    user = db.query(User).filter(User.pid == pid).first()
    if not user:
        user = User(pid=pid, nickname=body.nickname or None)
        db.add(user)
        db.commit()
        db.refresh(user)
    else:
        if body.nickname and user.nickname != body.nickname:
            user.nickname = body.nickname
            db.add(user)
            db.commit()
            db.refresh(user)

    token = create_access_token(user_id=user.id, pid=user.pid)
    return {
        "token": token,
        "user": {
            "id": user.id,
            "pid": user.pid,
            "nickname": user.nickname,
            "selected_bot": user.selected_bot
        }
    }


@app.post("/api/auth/join")
def join(body: JoinRequest, db: Session = Depends(get_db)):
    return _auth_join(body, db)


@app.get("/api/user/profile")
def profile(user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    a = (
        db.query(Assessment)
        .filter(Assessment.user_id == user.id)
        .order_by(Assessment.id.desc())
        .first()
    )
    r = (
        db.query(Recommendation)
        .filter(Recommendation.user_id == user.id)
        .order_by(Recommendation.id.desc())
        .first()
    )

    latest_recommendation = None
    if r:
        ranked = sorted(
            [{"type": k, "score": round(float(v) * 100, 2)} for k, v in (r.scores or {}).items()],
            key=lambda x: x["score"], reverse=True
        ) if r.scores else []
        latest_recommendation = {
            "scores": r.scores,
            "ranked": ranked,
            "top": {"type": r.selected_bot or (ranked[0]["type"] if ranked else None), "score": ranked[0]["score"] if ranked else 0},
            "selected_bot": r.selected_bot,
            "created_at": r.created_at.isoformat() + "Z",
        }

    return {
        "user": {"id": user.id, "pid": user.pid, "nickname": user.nickname, "selected_bot": user.selected_bot},
        "latest_assessment_id": a.id if a else None,
        "latest_recommendation": latest_recommendation,
    }


# -----------------------------------------------------------------------------
# Assessments
# -----------------------------------------------------------------------------

@app.post("/api/assessments/upsert")
def upsert_assessment(
    body: AssessmentUpsert,
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    # ★ 如果是重新測驗，清除用戶的 selected_bot
    if body.is_retest:
        user.selected_bot = None
        db.add(user)
        db.commit()

    a = Assessment(
        user_id=user.id,
        mbti_raw=(body.mbti_raw or None),
        mbti_encoded=(body.mbti_encoded or None),
        step2_answers=body.step2_answers,
        step3_answers=body.step3_answers,
        step4_answers=body.step4_answers,
        created_at=datetime.utcnow(),
    )
    db.add(a)
    db.commit()
    db.refresh(a)
    return {"ok": True, "assessment_id": a.id, "is_retest": body.is_retest or False}


@app.get("/api/assessments/me")
def my_assessment(user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    a = (
        db.query(Assessment)
        .filter(Assessment.user_id == user.id)
        .order_by(Assessment.id.desc())
        .first()
    )
    if not a:
        return {"assessment": None}
    return {
        "assessment": {
            "id": a.id,
            "mbti_raw": a.mbti_raw,
            "mbti_encoded": a.mbti_encoded,
            "step2_answers": a.step2_answers,
            "step3_answers": a.step3_answers,
            "step4_answers": a.step4_answers,
            "created_at": a.created_at.isoformat() + "Z",
        }
    }


# -----------------------------------------------------------------------------
# Matching
# -----------------------------------------------------------------------------

@app.post("/api/match/recommend")
def recommend(
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    a = (
        db.query(Assessment)
        .filter(Assessment.user_id == user.id)
        .order_by(Assessment.id.desc())
        .first()
    )
    if not a:
        raise HTTPException(status_code=400, detail="No assessment found")

    user_payload = {"id": user.id, "pid": user.pid, "nickname": user.nickname}
    assess_payload = {
        "id": a.id,
        "mbti_raw": a.mbti_raw,
        "mbti_encoded": a.mbti_encoded,
        "step2_answers": a.step2_answers,
        "step3_answers": a.step3_answers,
        "step4_answers": a.step4_answers,
    }
    result = build_recommendation_payload(user_payload, assess_payload)
    if not result or not result.get("scores"):
        raise HTTPException(status_code=500, detail="Recommendation engine failed")

    scores = result["scores"]
    ranked = sorted(
        [{"type": k, "score": round(float(v) * 100, 2)} for k, v in scores.items()],
        key=lambda x: x["score"], reverse=True
    )
    top_type = ranked[0]["type"] if ranked else None

    # ★ 建立新的推薦記錄，但不自動設定 selected_bot
    rec = Recommendation(
        user_id=user.id,
        scores=scores,
        selected_bot=None,  # 不自動選擇，等用戶手動選擇
        created_at=datetime.utcnow(),
    )
    db.add(rec)
    db.commit()
    db.refresh(rec)

    return {
        "ok": True,
        "scores": scores,
        "ranked": ranked,
        "top": {"type": top_type, "score": ranked[0]["score"] if ranked else 0},
        "recommendation_id": rec.id,
        "algorithm_version": result.get("algorithm_version"),
    }


@app.post("/api/match/choose")
def choose_bot(
    body: MatchChoice,
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    valid = {"empathy", "insight", "solution", "cognitive"}
    if body.bot_type not in valid:
        raise HTTPException(status_code=422, detail=f"Invalid bot_type, must be one of {sorted(valid)}")

    # ★ 更新用戶選擇的機器人
    user.selected_bot = body.bot_type
    db.add(user)
    db.commit()

    # ★ 同時更新最新的推薦記錄
    latest_rec = (
        db.query(Recommendation)
        .filter(Recommendation.user_id == user.id)
        .order_by(Recommendation.id.desc())
        .first()
    )
    if latest_rec:
        latest_rec.selected_bot = body.bot_type
        db.add(latest_rec)
        db.commit()

    return {"ok": True, "selected_bot": user.selected_bot}


@app.get("/api/match/me")
def my_match(user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    rec = (
        db.query(Recommendation)
        .filter(Recommendation.user_id == user.id)
        .order_by(Recommendation.id.desc())
        .first()
    )
    if not rec:
        return {"selected_bot": user.selected_bot, "latest_recommendation": None}

    ranked = sorted(
        [{"type": k, "score": round(float(v) * 100, 2)} for k, v in (rec.scores or {}).items()],
        key=lambda x: x["score"], reverse=True
    ) if rec.scores else []
    return {
        "selected_bot": user.selected_bot,
        "latest_recommendation": {
            "id": rec.id,
            "scores": rec.scores,
            "top": {"type": rec.selected_bot or (ranked[0]["type"] if ranked else None), "score": ranked[0]["score"] if ranked else 0},
            "selected_bot": rec.selected_bot,
            "created_at": rec.created_at.isoformat() + "Z",
        }
    }


# -----------------------------------------------------------------------------
# Chat
# -----------------------------------------------------------------------------

@app.post("/api/chat/messages")
def save_chat_message(
    body: ChatMessageCreate,
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    if body.role not in ("user", "ai"):
        raise HTTPException(status_code=422, detail="role must be 'user' or 'ai'")

    msg = ChatMessage(
        user_id=user.id,
        bot_type=body.bot_type or (user.selected_bot or "empathy"),
        mode=body.mode or "text",
        role=body.role,
        content=body.content,
        created_at=datetime.utcnow(),
        meta={"user_mood": body.user_mood, "mood_intensity": body.mood_intensity},
    )
    db.add(msg)
    db.commit()
    db.refresh(msg)
    return {"ok": True, "id": msg.id, "created_at": msg.created_at.isoformat() + "Z"}


@app.get("/api/chat/messages")
def get_chat_messages(
    limit: int = Query(50, ge=1, le=200),
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    rows = (
        db.query(ChatMessage)
        .filter(ChatMessage.user_id == user.id)
        .order_by(ChatMessage.id.desc())
        .limit(limit)
        .all()
    )
    out = [
        {
            "id": r.id,
            "bot_type": r.bot_type,
            "mode": r.mode,
            "role": r.role,
            "content": r.content,
            "created_at": r.created_at.isoformat() + "Z",
            "meta": r.meta,
        }
        for r in rows[::-1]
    ]
    return {"messages": out}


@app.post("/api/chat/send")
def chat_send(
    body: ChatSendRequest,
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    user_text = (body.message or "").strip()
    if not user_text:
        raise HTTPException(status_code=422, detail="message is required")

    # 1) 儲存使用者訊息
    umsg = ChatMessage(
        user_id=user.id,
        bot_type=body.bot_type or (user.selected_bot or "empathy"),
        mode=body.mode or "text",
        role="user",
        content=user_text,
        created_at=datetime.utcnow(),
    )
    db.add(umsg)
    db.commit()
    db.refresh(umsg)

    # 2) 呼叫 OpenAI（失敗時會回傳 debug 文本）
    reply_text = None
    try:
        from openai import OpenAI  # openai>=1.0
        client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        system_prompt = "You are Emobot, a supportive counseling assistant. Keep responses concise and empathetic."
        completion = client.chat.completions.create(
            model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_text},
            ],
            temperature=float(os.getenv("OPENAI_TEMPERATURE", "0.7")),
            max_tokens=int(os.getenv("OPENAI_MAX_TOKENS", "512")),
        )
        reply_text = completion.choices[0].message.content if completion and completion.choices else None
    except Exception as e:
        reply_text = f"(debug) OpenAI unavailable: {e}. Echo: {user_text[:200]}"

    # 3) 儲存 AI 回覆
    atext = reply_text or "..."
    amsg = ChatMessage(
        user_id=user.id,
        bot_type=body.bot_type or (user.selected_bot or "empathy"),
        mode=body.mode or "text",
        role="ai",
        content=atext,
        created_at=datetime.utcnow(),
    )
    db.add(amsg)
    db.commit()
    db.refresh(amsg)

    return {"reply": atext, "message_id": amsg.id}


# -----------------------------------------------------------------------------
# Mood
# -----------------------------------------------------------------------------

@app.post("/api/mood/records")
def create_mood(
    body: MoodRecordCreate,
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    rec = MoodRecord(
        user_id=user.id,
        mood=body.mood,
        intensity=body.intensity,
        note=body.note,
        created_at=datetime.utcnow(),
    )
    db.add(rec)
    db.commit()
    db.refresh(rec)
    return {"ok": True, "id": rec.id, "created_at": rec.created_at.isoformat() + "Z"}


@app.get("/api/mood/records")
def list_mood(
    days: int = Query(30, ge=1, le=365),
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    since = datetime.utcnow() - timedelta(days=days)
    rows = (
        db.query(MoodRecord)
        .filter(and_(MoodRecord.user_id == user.id, MoodRecord.created_at >= since))
        .order_by(MoodRecord.id.desc())
        .all()
    )
    return {
        "records": [
            {
                "id": r.id,
                "mood": r.mood,
                "intensity": r.intensity,
                "note": r.note,
                "created_at": r.created_at.isoformat() + "Z",
            }
            for r in rows[::-1]
        ]
    }