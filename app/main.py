# app/main.py
from __future__ import annotations

import os
import re
from datetime import datetime, timezone
from typing import Any, Dict, Optional, List

import httpx
from fastapi import FastAPI, Depends, HTTPException, Query, Request, Header, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from sqlalchemy import text
from sqlalchemy.orm import Session

# ---------------- Internal modules ----------------
try:
    from app.core.config import settings
    _ALLOWED = getattr(settings, "ALLOWED_ORIGINS", os.getenv("ALLOWED_ORIGINS", ""))
except Exception:
    _ALLOWED = os.getenv("ALLOWED_ORIGINS", "")

from app.core.security import create_access_token, get_current_user
from app.db.session import get_db, engine
from app.db.base import Base
from app.models.user import User
from app.models.assessment import Assessment
from app.models.recommendation import Recommendation
from app.models.chat import ChatMessage
from app.models.mood import MoodRecord

# ---- 推薦引擎：名稱相容（compute_ / make_ 都可） ----
try:
    from app.services.recommendation_engine import compute_recommendation as _run_recommend
except ImportError:
    from app.services.recommendation_engine import make_recommendation as _run_recommend

# ==================================================
# App & CORS
# ==================================================
app = FastAPI(title="Emobot+ Backend")

# 初始化資料庫（若已建立會自動跳過）
Base.metadata.create_all(bind=engine)

def _parse_allowed(origins_str: str) -> List[str]:
    out: List[str] = []
    for s in (origins_str or "").split(","):
        s = s.strip()
        if not s or s == "*" or s.lower() == "null":
            continue
        out.append(s)
    return out

_ALLOWED_ORIGINS = _parse_allowed(_ALLOWED)
# 你的正式站：請務必在環境變數 ALLOWED_ORIGINS 加上 https://emobot-plus.vercel.app
# 例如：ALLOWED_ORIGINS="https://emobot-plus.vercel.app,http://localhost:5173"
_VERCEL_REGEX_STR = r"^https://.*\.vercel\.app$"
_VERCEL_REGEX = re.compile(_VERCEL_REGEX_STR, re.IGNORECASE)

# 1) 官方 CORS middleware（主要處理）
app.add_middleware(
    CORSMiddleware,
    allow_origins=_ALLOWED_ORIGINS,         # 明確網域列表（不能含 "*"）
    allow_origin_regex=_VERCEL_REGEX_STR,   # 允許所有 Vercel preview 子網域
    allow_credentials=True,                 # ✅ 必須 true 才能搭配 fetch credentials:"include"
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["*"],
)

# 2) 保底 CORS middleware（萬一上層代理或舊版套件沒加 header，也會補上）
@app.middleware("http")
async def _force_cors_headers(request: Request, call_next):
    origin = request.headers.get("origin")
    acrh = request.headers.get("access-control-request-headers")
    acrm = request.headers.get("access-control-request-method")

    # 若是預檢請求，優先直接回 204 並補 header（避免被路由 404/405 擋掉）
    if request.method == "OPTIONS" and origin:
        if (origin in _ALLOWED_ORIGINS) or (_VERCEL_REGEX.match(origin) is not None):
            headers = {
                "Access-Control-Allow-Origin": origin,
                "Access-Control-Allow-Credentials": "true",
                "Access-Control-Allow-Methods": "GET,POST,PUT,PATCH,DELETE,OPTIONS",
                "Access-Control-Max-Age": "86400",
                "Vary": "Origin",
                "Access-Control-Expose-Headers": "*",
                "Access-Control-Allow-Headers": acrh or "Authorization,Content-Type,X-User-Id,*",
            }
            return Response(status_code=204, headers=headers)
        # 不在允許清單：走正常流程，讓官方 CORSMiddleware 覆判
    # 非預檢：先走後續處理，再補 header
    response = await call_next(request)
    if origin and ((origin in _ALLOWED_ORIGINS) or (_VERCEL_REGEX.match(origin) is not None)):
        response.headers["Access-Control-Allow-Origin"] = origin
        response.headers["Access-Control-Allow-Credentials"] = "true"
        # 若官方 CORSMiddleware 已加 Vary，此處不覆蓋；否則補上
        vary = response.headers.get("Vary")
        response.headers["Vary"] = "Origin" if not vary else (vary if "Origin" in vary else f"{vary}, Origin")
        response.headers.setdefault("Access-Control-Expose-Headers", "*")
    return response

# ==================================================
# Helpers
# ==================================================
def now_iso() -> str:
    return datetime.utcnow().replace(tzinfo=timezone.utc).isoformat()

def get_user_by_header(x_user_id: Optional[str], db: Session) -> Optional[User]:
    if not x_user_id:
        return None
    try:
        uid = int(x_user_id)
    except Exception:
        return None
    return db.get(User, uid)

def latest_assessment(db: Session, user_id: int) -> Optional[Assessment]:
    return (
        db.query(Assessment)
        .filter(Assessment.user_id == user_id)
        .order_by(Assessment.id.desc())
        .first()
    )

def normalize_recommendation_output(raw: Any) -> Dict[str, Any]:
    """統一回傳格式：{scores, ranked, top}"""
    scores: Dict[str, float] = {}
    if isinstance(raw, dict):
        if "scores" in raw and isinstance(raw["scores"], dict):
            scores = {k: float(v) for k, v in raw["scores"].items()}
        elif "ranked" in raw and isinstance(raw["ranked"], list):
            tmp = {}
            for it in raw["ranked"]:
                if isinstance(it, dict):
                    k = it.get("type") or it.get("name"); v = it.get("score")
                else:
                    k, v = it
                if k is not None and v is not None:
                    tmp[str(k)] = float(v)
            scores = tmp
        elif all(isinstance(v, (int, float)) for v in raw.values()):
            scores = {k: float(v) for k, v in raw.items()}
    elif isinstance(raw, list):
        tmp = {}
        for it in raw:
            if isinstance(it, dict):
                k = it.get("type") or it.get("name"); v = it.get("score")
            else:
                k, v = it
            if k is not None and v is not None:
                tmp[str(k)] = float(v)
        scores = tmp
    if not scores:
        scores = {"empathy": 0.5, "insight": 0.5, "solution": 0.5, "cognitive": 0.5}
    ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
    top = {"type": ranked[0][0], "score": ranked[0][1]}
    return {"scores": scores, "ranked": ranked, "top": top}

# ==================================================
# Health
# ==================================================
@app.get("/api/health")
def health():
    try:
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
    except Exception as e:
        return {"ok": False, "db": False, "error": str(e)}
    return {"ok": True, "db": True, "time": now_iso()}

# ==================================================
# Auth（兩條相容路徑）
# ==================================================
def _auth_join(body: dict, db: Session) -> Dict[str, Any]:
    pid = (body.get("pid") or "").strip()
    nickname = (body.get("nickname") or None)
    if not pid:
        raise HTTPException(400, "pid required")
    user = db.query(User).filter(User.pid == pid).first()
    if not user:
        user = User(pid=pid, nickname=nickname)
        db.add(user); db.commit(); db.refresh(user)
    elif nickname and user.nickname != nickname:
        user.nickname = nickname; db.commit()
    token = create_access_token(user.id, user.pid)
    return {"ok": True, "token": token, "user": {"id": user.id, "pid": user.pid, "nickname": user.nickname}}

@app.post("/api/auth/login")
def login(body: dict, db: Session = Depends(get_db)):
    return _auth_join(body, db)

@app.post("/api/auth/join")
def join(body: dict, db: Session = Depends(get_db)):
    return _auth_join(body, db)

@app.get("/api/me")
def me(user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    ass = latest_assessment(db, user.id)
    rec = (
        db.query(Recommendation)
        .filter(Recommendation.user_id == user.id)
        .order_by(Recommendation.id.desc())
        .first()
    )
    def ass_dump(a: Optional[Assessment]):
        if not a: return None
        return {
            "id": a.id,
            "mbti": {"raw": a.mbti_raw, "encoded": a.mbti_encoded},
            "step2Answers": a.step2_answers,
            "step3Answers": a.step3_answers,
            "step4Answers": a.step4_answers,
            "ai_preference": a.ai_preference,
            "submitted_at": a.submitted_at.replace(tzinfo=timezone.utc).isoformat() if a.submitted_at else None,
        }
    def rec_dump(r: Optional[Recommendation]):
        if not r: return None
        return {
            "id": r.id,
            "scores": r.scores,
            "top_bot": r.top_bot,
            "created_at": r.created_at.replace(tzinfo=timezone.utc).isoformat(),
            "selected_bot": user.selected_bot,
        }
    return {
        "ok": True,
        "user": {"id": user.id, "pid": user.pid, "nickname": user.nickname, "selected_bot": user.selected_bot},
        "latest_assessment": ass_dump(ass),
        "latest_recommendation": rec_dump(rec),
    }

# ==================================================
# Assessment
# ==================================================
@app.post("/api/assessments/upsert")
def upsert_assessment(
    body: dict,
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    ass = latest_assessment(db, user.id) or Assessment(user_id=user.id)
    if "mbti_raw" in body or "mbti" in body:
        ass.mbti_raw = str(body.get("mbti_raw") or body.get("mbti") or "")
    if "mbti_encoded" in body or "encoded" in body:
        ass.mbti_encoded = body.get("mbti_encoded") or body.get("encoded")
    if "step2Answers" in body or "step2_answers" in body:
        ass.step2_answers = body.get("step2Answers") or body.get("step2_answers")
    if "step3Answers" in body or "step3_answers" in body:
        ass.step3_answers = body.get("step3Answers") or body.get("step3_answers")
    if "step4Answers" in body or "step4_answers" in body:
        ass.step4_answers = body.get("step4Answers") or body.get("step4_answers")
    if "ai_preference" in body:
        ass.ai_preference = body.get("ai_preference")
    if "submittedAt" in body:
        try:
            ass.submitted_at = datetime.fromisoformat(body["submittedAt"].replace("Z", "+00:00"))
        except Exception:
            ass.submitted_at = datetime.utcnow()
    db.add(ass); db.commit(); db.refresh(ass)
    return {"ok": True, "assessment_id": ass.id}

# ==================================================
# Recommendation（兩個路徑相容）
# ==================================================
def _run_and_save_recommendation(user: User, db: Session) -> Dict[str, Any]:
    ass = latest_assessment(db, user.id)
    if not ass:
        raise HTTPException(400, "no assessment yet")
    raw = _run_recommend(
        mbti=ass.mbti_encoded or [],
        aas=ass.step2_answers or [],
        ders=ass.step3_answers or [],
        bpns=ass.step4_answers or [],
        ai_preference=(ass.ai_preference or {}),
    )
    result = normalize_recommendation_output(raw)
    rec = Recommendation(
        user_id=user.id,
        assessment_id=ass.id,
        scores=result["scores"],
        top_bot=result["top"]["type"],
        features={"ranked": result["ranked"]},
    )
    db.add(rec); db.commit(); db.refresh(rec)
    return {"ok": True, "id": rec.id, **result}

@app.post("/api/recommend/run")
def recommend_run(user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    return _run_and_save_recommendation(user, db)

@app.post("/api/match/recommend")
def recommend_run_compat(user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    return _run_and_save_recommendation(user, db)

@app.post("/api/match/choose")
def match_choose(body: dict, user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    bot_type = (body.get("bot_type") or body.get("type") or "").strip()
    if not bot_type:
        raise HTTPException(400, "bot_type required")
    user.selected_bot = bot_type
    db.commit()
    return {"ok": True, "selected_bot": user.selected_bot}

# ==================================================
# Chat（CRUD + send 對接 OpenAI）
# ==================================================
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
        "id": msg.id, "message_type": msg.message_type, "bot_type": msg.bot_type, "content": msg.content,
        "user_mood": msg.user_mood, "mood_intensity": msg.mood_intensity,
        "created_at": msg.created_at.replace(tzinfo=timezone.utc).isoformat()
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
            "id": r.id, "message_type": r.message_type, "bot_type": r.bot_type, "content": r.content,
            "user_mood": r.user_mood, "mood_intensity": r.mood_intensity,
            "created_at": r.created_at.replace(tzinfo=timezone.utc).isoformat()
        }
        for r in reversed(rows)
    ]

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_BASE = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

def bot_system_prompt(bot_type: str) -> str:
    base = "你是 Emobot+ 的心理支持助理。請以中文、溫和、具體、短句回覆，避免醫療診斷。"
    t = (bot_type or "").lower()
    if t == "empathy":   return base + "你的風格是同理型：優先傾聽、反映感受、給出安撫與情緒標籤。"
    if t == "insight":   return base + "你的風格是洞察型：用溫柔的提問與重述，幫助釐清想法與行為模式。"
    if t == "solution":  return base + "你的風格是解決型：將問題拆小、提供可行的一步行動與清單。"
    if t == "cognitive": return base + "你的風格是認知型：結構化表達，提出認知重評練習與作業。"
    return base

@app.post("/api/chat/send")
async def chat_send(
    body: dict,
    request: Request,
    x_user_id: Optional[str] = Header(default=None, alias="X-User-Id"),
    db: Session = Depends(get_db),
):
    # /api/chat/send：不強制 token，允許透過 X-User-Id 使用
    user = get_user_by_header(x_user_id, db)

    bot_type = (body.get("bot_type") or "").lower()
    message = (body.get("message") or "").strip()
    history = body.get("history") or []
    if not message:
        raise HTTPException(400, "message required")

    msgs = [{"role": "system", "content": bot_system_prompt(bot_type)}]
    for h in history[-12:]:
        r = h.get("role"); c = h.get("content")
        if r in ("user", "assistant") and isinstance(c, str):
            msgs.append({"role": r, "content": c})
    msgs.append({"role": "user", "content": message})

    reply_text: str = ""
    used_model = OPENAI_MODEL

    if OPENAI_API_KEY:
        try:
            async with httpx.AsyncClient(timeout=60.0) as client:
                resp = await client.post(
                    f"{OPENAI_BASE}/chat/completions",
                    headers={"Authorization": f"Bearer {OPENAI_API_KEY}"},
                    json={"model": used_model, "messages": msgs, "temperature": 0.6, "max_tokens": 400},
                )
                if resp.status_code >= 300:
                    raise RuntimeError(f"OpenAI {resp.status_code}: {resp.text[:200]}")
                data = resp.json()
                reply_text = (data.get("choices") or [{}])[0].get("message", {}).get("content", "").strip()
        except Exception:
            reply_text = f"（備援回覆）我收到你的訊息：「{message[:80]}」。目前外部服務暫時無法連線，但我仍在這裡陪你。能再說說你此刻最在意的是什麼嗎？"
    else:
        reply_text = f"（示範回覆）你說：「{message[:80]}」。若要啟用真實 AI 回覆，請在後端設定 OPENAI_API_KEY。"

    # 紀錄聊天（若辨識到 user）
    try:
        if user:
            db.add(ChatMessage(user_id=user.id, message_type="user", bot_type=bot_type, content=message))
            db.add(ChatMessage(user_id=user.id, message_type="bot",  bot_type=bot_type, content=reply_text))
            db.commit()
    except Exception:
        db.rollback()

    return {"ok": True, "reply": reply_text, "model": used_model, "time": now_iso()}

# ==================================================
# Mood
# ==================================================
@app.post("/api/mood/create")
def create_mood(
    body: dict,
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    rec = MoodRecord(
        user_id=user.id,
        mood=str(body.get("mood") or ""),
        intensity=int(body.get("intensity")) if body.get("intensity") is not None else None,
        note=(body.get("note") or None),
    )
    db.add(rec); db.commit(); db.refresh(rec)
    return {"ok": True, "id": rec.id, "created_at": rec.created_at.replace(tzinfo=timezone.utc).isoformat()}

# 相容舊版：POST /api/mood/records 建立
@app.post("/api/mood/records")
def create_mood_compat(
    body: dict,
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    return create_mood(body, user, db)

@app.get("/api/mood/records")
def list_mood(
    days: int = Query(30, ge=1, le=180),
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    rows = (
        db.query(MoodRecord)
        .filter(MoodRecord.user_id == user.id)
        .order_by(MoodRecord.id.desc())
        .limit(500)
        .all()
    )
    return [
        {"id": r.id, "mood": r.mood, "intensity": r.intensity, "note": r.note,
         "created_at": r.created_at.replace(tzinfo=timezone.utc).isoformat()}
        for r in reversed(rows)
    ]
