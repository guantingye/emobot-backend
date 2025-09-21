# app/main.py - 最終修復版，確保 CORS、HeyGen 與 AV 路由正確註冊
from __future__ import annotations

import os
import re
from datetime import datetime, timedelta
from typing import Optional, List, Dict, Any

from fastapi import FastAPI, Depends, HTTPException, Query, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from sqlalchemy import text
from sqlalchemy.orm import Session

# ---- App internals ----
from app.core.config import settings
from app.core.security import create_access_token, get_current_user
from app.db.session import get_db, engine
from app.db.base import Base
from app.models.user import User
from app.models.assessment import Assessment
from app.models.recommendation import Recommendation
from app.models.chat import ChatMessage
from app.models.mood import MoodRecord
from app.models.allowed_pid import AllowedPid
from app.models.chat_session import ChatSession

# 路由
from app.routers import av                   # ← 新增 lipsync 視音訊路由
from app.routers import did_router           # ← HeyGen / DID 既有路由（保留）

# ---- Optional: 外部推薦引擎，失敗時走 fallback ----
try:
    from app.services.recommendation_engine import recommend_endpoint_payload as _build_reco
except Exception:
    _build_reco = None


def _fallback_build_reco(user: Dict[str, Any] | None, assessment: Dict[str, Any] | None) -> Dict[str, Any]:
    """簡易可重現的回退推薦（保留你原本邏輯）"""
    empathy = insight = solution = cognitive = 0.25
    if assessment:
        enc = assessment.get("mbti_encoded")
        if isinstance(enc, (list, tuple)) and len(enc) >= 4:
            def norm(v):
                try:
                    v = float(v)
                    if v > 1:
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
            return _build_reco(user=user, assessment=assessment)
        except Exception:
            return _fallback_build_reco(user, assessment)
    return _fallback_build_reco(user, assessment)


# -----------------------------------------------------------------------------
# FastAPI App
# -----------------------------------------------------------------------------
app = FastAPI(title="Emobot Backend", version="0.5.1")

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
        out.append(s.rstrip("/"))
    return out

_ALLOWED_ORIGINS = _parse_allowed(ALLOWED)

# 允許 Vercel 預覽網域（更廣但安全）
_VERCEL_REGEX_STR = r"^https:\/\/[a-z0-9\-]+\.vercel\.app$"
_VERCEL_REGEX = re.compile(_VERCEL_REGEX_STR, re.IGNORECASE)

# 1) 官方 CORSMiddleware：需在註冊任何 router 前掛上
app.add_middleware(
    CORSMiddleware,
    allow_origins=_ALLOWED_ORIGINS,
    allow_origin_regex=_VERCEL_REGEX_STR,
    allow_credentials=True,  # 前端若使用 Bearer Token 並未帶 Cookie 也相容
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["*"],
    max_age=86400,
)

# 2) 自訂補丁：預檢與錯誤時也帶 CORS
@app.middleware("http")
async def _force_cors_headers(request: Request, call_next):
    origin = request.headers.get("origin")
    is_allowed = bool(origin and (origin.rstrip("/") in _ALLOWED_ORIGINS or _VERCEL_REGEX.match(origin or "")))

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


# ====== 掛載靜態與路由（CORS 已掛好）======
# /static/av -> 影音暫存
mount_path, static_app, name = av.get_static_mount()
app.mount(mount_path, static_app, name)

# HeyGen / DID 舊路由保留（移到 CORS 之後，確保帶到 CORS 標頭）
app.include_router(did_router.router)

# *** 修復：更強健的 chat router 註冊 ***
chat_router_loaded = False
chat_router_error = None

try:
    import app.chat as chat_module
    if not hasattr(chat_module, 'router'):
        raise ImportError("chat.py 中沒有定義 router")
    chat_router = chat_module.router
    app.include_router(chat_router, prefix="/api/chat", tags=["chat"])
    chat_router_loaded = True
    print("✅ Chat router 註冊成功")
    heygen_routes = []
    for route in chat_router.routes:
        if hasattr(route, 'path') and 'heygen' in route.path:
            heygen_routes.append(route.path)
    if heygen_routes:
        print(f"✅ HeyGen 路由註冊成功: {heygen_routes}")
    else:
        print("⚠️ 警告：沒有找到 HeyGen 相關路由")
except ImportError as e:
    chat_router_error = f"導入錯誤: {e}"
    print(f"❌ Chat router 導入失敗: {e}")
except Exception as e:
    chat_router_error = f"註冊錯誤: {e}"
    print(f"❌ Chat router 註冊失敗: {e}")

# 如果 chat router 載入失敗，創建緊急備用路由
if not chat_router_loaded:
    from fastapi import APIRouter
    emergency_router = APIRouter()

    @emergency_router.get("/health")
    async def emergency_health():
        return {
            "ok": False,
            "error": "Chat router 載入失敗",
            "details": chat_router_error,
            "emergency_mode": True
        }

    @emergency_router.get("/status")
    async def emergency_status():
        return {
            "chat_router_loaded": False,
            "error": chat_router_error,
            "available_endpoints": ["/api/chat/health", "/api/chat/status"]
        }

    app.include_router(emergency_router, prefix="/api/chat", tags=["emergency"])
    print("🚨 緊急備用路由已啟動")

# 開源影音路由（/api/av/*）
app.include_router(av.router)


# -----------------------------------------------------------------------------
# Schemas（保留原版）
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
    is_retest: Optional[bool] = False


class MatchChoice(BaseModel):
    bot_type: str = Field(..., description="empathy | insight | solution | cognitive")


class ChatSendRequest(BaseModel):
    message: str
    bot_type: Optional[str] = None
    mode: Optional[str] = "text"
    history: Optional[List[Dict[str, str]]] = []
    demo: Optional[bool] = False


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


class AllowedPidCreate(BaseModel):
    pid: str = Field(..., min_length=1, max_length=50)
    description: Optional[str] = Field(default=None, max_length=200)


class AllowedPidUpdate(BaseModel):
    is_active: Optional[bool] = None
    description: Optional[str] = Field(default=None, max_length=200)


class ChatSessionCreate(BaseModel):
    bot_type: Optional[str] = Field(default="solution")


class ChatSessionEnd(BaseModel):
    reason: str = Field(..., pattern="^(user_ended|timeout|system)$")


# -----------------------------------------------------------------------------
# Helper Functions（保留原版）
# -----------------------------------------------------------------------------
def get_system_prompt(bot_type: str) -> str:
    prompts = {
        "empathy": "你是 Lumi，同理型 AI。以溫柔、非評判、短句的反映傾聽與情緒標記來回應。優先肯認、共感與陪伴。用繁體中文回覆，保持溫暖支持的語調。",
        "insight": "你是 Solin，洞察型 AI。以蘇格拉底式提問、澄清與重述，幫助使用者澄清想法，維持中性、尊重、結構化。用繁體中文回覆。",
        "solution": "你是 Niko，解決型 AI。以務實、具體的建議與分步行動為主，給出小目標、工具與下一步，語氣鼓勵但不強迫。用繁體中文回覆。",
        "cognitive": "你是 Clara，認知型 AI。以 CBT 語氣幫助辨識自動想法、認知偏誤與替代想法，提供簡短表格式步驟與練習。用繁體中文回覆。"
    }
    return prompts.get(bot_type, prompts["solution"])


def get_bot_name(bot_type: str) -> str:
    names = {"empathy": "Lumi", "insight": "Solin", "solution": "Niko", "cognitive": "Clara"}
    return names.get(bot_type, "Niko")


def call_openai(system_prompt: str, messages: List[Dict[str, str]]) -> str:
    api_key = os.getenv("OPENAI_API_KEY", "")
    if not api_key:
        return "我在這裡陪著你。想聊聊今天最讓你在意的事情嗎？"

    try:
        from openai import OpenAI
        client = OpenAI(api_key=api_key)
        chat_messages = [{"role": "system", "content": system_prompt}] + messages
        response = client.chat.completions.create(
            model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
            messages=chat_messages,
            temperature=float(os.getenv("OPENAI_TEMPERATURE", "0.7")),
            max_tokens=int(os.getenv("OPENAI_MAX_TOKENS", "600")),
        )
        return response.choices[0].message.content.strip() if response.choices else ""
    except Exception as e:
        print(f"OpenAI API failed: {e}")
        return "我在這裡陪著你。想聊聊今天最讓你在意的事情嗎？"


def is_pid_allowed(pid: str, db: Session) -> bool:
    allowed_pid = db.query(AllowedPid).filter(
        AllowedPid.pid == pid,
        AllowedPid.is_active == True
    ).first()
    return allowed_pid is not None


def get_or_create_active_session(user_id: int, bot_type: str, db: Session) -> ChatSession:
    active_session = db.query(ChatSession).filter(
        ChatSession.user_id == user_id,
        ChatSession.is_active == True
    ).first()
    if active_session:
        active_session.last_activity = datetime.utcnow()
        active_session.bot_type = bot_type
        db.add(active_session)
        db.commit()
        return active_session

    new_session = ChatSession(
        user_id=user_id,
        bot_type=bot_type,
        session_start=datetime.utcnow(),
        last_activity=datetime.utcnow(),
        is_active=True
    )
    db.add(new_session)
    db.commit()
    db.refresh(new_session)
    return new_session


def end_inactive_sessions(db: Session, timeout_minutes: int = 5):
    timeout_threshold = datetime.utcnow() - timedelta(minutes=timeout_minutes)
    inactive_sessions = db.query(ChatSession).filter(
        ChatSession.is_active == True,
        ChatSession.last_activity < timeout_threshold
    ).all()
    for session in inactive_sessions:
        session.is_active = False
        session.session_end = datetime.utcnow()
        session.end_reason = "timeout"
        db.add(session)
    if inactive_sessions:
        db.commit()
        print(f"已結束 {len(inactive_sessions)} 個非活躍會話")
    return len(inactive_sessions)


def update_session_activity(user_id: int, db: Session):
    active_session = db.query(ChatSession).filter(
        ChatSession.user_id == user_id,
        ChatSession.is_active == True
    ).first()
    if active_session:
        active_session.last_activity = datetime.utcnow()
        active_session.message_count += 1
        db.add(active_session)
        db.commit()
    return active_session


# -----------------------------------------------------------------------------
# Health & Debug（保留原版＋略調整）
# -----------------------------------------------------------------------------
@app.get("/api/health")
def health():
    return {
        "ok": True,
        "time": datetime.utcnow().isoformat() + "Z",
        "chat_router_loaded": chat_router_loaded,
        "chat_router_error": chat_router_error,
        "heygen_enabled": bool(os.getenv("HEYGEN_API_KEY")),
        "routes_count": len(app.routes)
    }

@app.get("/api/debug/db-test")
def db_test(db: Session = Depends(get_db)):
    try:
        db.execute(text("select 1"))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"DB not ready: {e}")
    return {"ok": True}

@app.get("/api/debug/routes")
def list_all_routes():
    routes = []
    for route in app.routes:
        if hasattr(route, 'methods') and hasattr(route, 'path'):
            routes.append({
                "path": route.path,
                "methods": list(route.methods),
                "name": getattr(route, 'name', 'unnamed')
            })
    return {
        "total_routes": len(routes),
        "routes": routes,
        "chat_routes": [r for r in routes if '/chat/' in r['path']],
        "heygen_routes": [r for r in routes if 'heygen' in r['path']],
        "chat_router_status": {
            "loaded": chat_router_loaded,
            "error": chat_router_error
        }
    }


# -----------------------------------------------------------------------------
# Auth & Profile（保留原版）
# -----------------------------------------------------------------------------
class JoinRequest(BaseModel):
    pid: str = Field(..., min_length=1, max_length=50)
    nickname: Optional[str] = Field(default=None, max_length=100)

def _auth_join(body: JoinRequest, db: Session):
    pid = (body.pid or "").strip()
    if not pid:
        raise HTTPException(status_code=422, detail="pid is required")
    if not is_pid_allowed(pid, db):
        raise HTTPException(status_code=403, detail="此 PID 未被授權使用系統，請聯繫管理員")
    user = db.query(User).filter(User.pid == pid).first()
    if not user:
        user = User(pid=pid, nickname=body.nickname or None)
        db.add(user); db.commit(); db.refresh(user)
    else:
        if body.nickname and user.nickname != body.nickname:
            user.nickname = body.nickname
            db.add(user); db.commit(); db.refresh(user)
    token = create_access_token(user_id=user.id, pid=user.pid)
    return {"token": token, "user": {"id": user.id, "pid": user.pid, "nickname": user.nickname, "selected_bot": user.selected_bot}}

@app.post("/api/auth/join")
def join(body: JoinRequest, db: Session = Depends(get_db)):
    return _auth_join(body, db)

@app.get("/api/user/profile")
def profile(user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    a = db.query(Assessment).filter(Assessment.user_id == user.id).order_by(Assessment.id.desc()).first()
    r = db.query(Recommendation).filter(Recommendation.user_id == user.id).order_by(Recommendation.id.desc()).first()
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
# Assessments / Matching（保留原版）
# -----------------------------------------------------------------------------
class AssessmentUpsert(BaseModel):
    mbti_raw: Optional[str] = None
    mbti_encoded: Optional[List[float]] = None
    step2_answers: Optional[List[Any]] = None
    step3_answers: Optional[List[Any]] = None
    step4_answers: Optional[List[Any]] = None
    ai_preference: Optional[Dict[str, Any]] = None
    submittedAt: Optional[datetime] = None
    is_retest: Optional[bool] = False

@app.post("/api/assessments/upsert")
def upsert_assessment(body: AssessmentUpsert, user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    if body.is_retest:
        user.selected_bot = None
        db.add(user); db.commit()
    a = Assessment(
        user_id=user.id,
        mbti_raw=(body.mbti_raw or None),
        mbti_encoded=(body.mbti_encoded or None),
        step2_answers=body.step2_answers,
        step3_answers=body.step3_answers,
        step4_answers=body.step4_answers,
        created_at=datetime.utcnow(),
    )
    db.add(a); db.commit(); db.refresh(a)
    return {"ok": True, "assessment_id": a.id, "is_retest": body.is_retest or False}

@app.get("/api/assessments/me")
def my_assessment(user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    a = db.query(Assessment).filter(Assessment.user_id == user.id).order_by(Assessment.id.desc()).first()
    if not a:
        return {"assessment": None}
    return {
        "assessment": {
            "id": a.id, "mbti_raw": a.mbti_raw, "mbti_encoded": a.mbti_encoded,
            "step2_answers": a.step2_answers, "step3_answers": a.step3_answers, "step4_answers": a.step4_answers,
            "created_at": a.created_at.isoformat() + "Z",
        }
    }

@app.post("/api/match/recommend")
def recommend(user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    a = db.query(Assessment).filter(Assessment.user_id == user.id).order_by(Assessment.id.desc()).first()
    if not a:
        raise HTTPException(status_code=400, detail="No assessment found")
    user_payload = {"id": user.id, "pid": user.pid, "nickname": user.nickname}
    assess_payload = {
        "id": a.id, "mbti_raw": a.mbti_raw, "mbti_encoded": a.mbti_encoded,
        "step2_answers": a.step2_answers, "step3_answers": a.step3_answers, "step4_answers": a.step4_answers,
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
    rec = Recommendation(user_id=user.id, scores=scores, selected_bot=None, created_at=datetime.utcnow())
    db.add(rec); db.commit(); db.refresh(rec)
    return {"ok": True, "scores": scores, "ranked": ranked, "top": {"type": top_type, "score": ranked[0]["score"] if ranked else 0},
            "recommendation_id": rec.id, "algorithm_version": result.get("algorithm_version")}

class MatchChoice(BaseModel):
    bot_type: str = Field(..., description="empathy | insight | solution | cognitive")

@app.post("/api/match/choose")
def choose_bot(body: MatchChoice, user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    valid = {"empathy", "insight", "solution", "cognitive"}
    if body.bot_type not in valid:
        raise HTTPException(status_code=422, detail=f"Invalid bot_type, must be one of {sorted(valid)}")
    user.selected_bot = body.bot_type
    db.add(user); db.commit()
    latest_rec = db.query(Recommendation).filter(Recommendation.user_id == user.id).order_by(Recommendation.id.desc()).first()
    if latest_rec:
        latest_rec.selected_bot = body.bot_type
        db.add(latest_rec); db.commit()
    return {"ok": True, "selected_bot": user.selected_bot}

@app.get("/api/match/me")
def my_match(user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    rec = db.query(Recommendation).filter(Recommendation.user_id == user.id).order_by(Recommendation.id.desc()).first()
    if not rec:
        return {"selected_bot": user.selected_bot, "latest_recommendation": None}
    ranked = sorted(
        [{"type": k, "score": round(float(v) * 100, 2)} for k, v in (rec.scores or {}).items()],
        key=lambda x: x["score"], reverse=True
    ) if rec.scores else []
    return {
        "selected_bot": user.selected_bot,
        "latest_recommendation": {
            "id": rec.id, "scores": rec.scores,
            "top": {"type": rec.selected_bot or (ranked[0]["type"] if ranked else None), "score": ranked[0]["score"] if ranked else 0},
            "selected_bot": rec.selected_bot, "created_at": rec.created_at.isoformat() + "Z",
        }
    }


# -----------------------------------------------------------------------------
# Chat（保留原版）
# -----------------------------------------------------------------------------
@app.post("/api/chat/send")
def chat_send(
    body: ChatSendRequest,
    request: Request,
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    user_msg = (body.message or "").strip()
    if not user_msg:
        raise HTTPException(status_code=400, detail="Empty message")

    user_id = user.id
    bot_type = body.bot_type or user.selected_bot or "solution"

    try:
        end_inactive_sessions(db)
        chat_session = get_or_create_active_session(user_id, bot_type, db)

        user_message = ChatMessage(
            user_id=user_id,
            bot_type=bot_type,
            mode=body.mode or "text",
            role="user",
            content=user_msg,
            meta={"demo": body.demo, "session_id": chat_session.id}
        )
        db.add(user_message); db.commit()
        update_session_activity(user_id, db)

        system_prompt = get_system_prompt(bot_type)
        messages = []
        for h in (body.history or [])[-10:]:
            role = "assistant" if h.get("role") == "assistant" else "user"
            messages.append({"role": role, "content": h.get("content", "")})
        messages.append({"role": "user", "content": user_msg})

        reply_text = call_openai(system_prompt, messages)

        ai_message = ChatMessage(
            user_id=user_id,
            bot_type=bot_type,
            mode=body.mode or "text",
            role="ai",
            content=reply_text,
            meta={"provider": "openai", "model": os.getenv("OPENAI_MODEL", "gpt-4o-mini"), "session_id": chat_session.id}
        )
        db.add(ai_message); db.commit()
        update_session_activity(user_id, db)

        return {"ok": True, "reply": reply_text, "bot": {"type": bot_type, "name": get_bot_name(bot_type)},
                "message_id": ai_message.id, "session_id": chat_session.id, "error": None}
    except Exception as e:
        print(f"Chat send error: {e}")
        db.rollback()
        return {"ok": False, "reply": "抱歉，我暫時無法回應。請稍後再試。",
                "bot": {"type": bot_type, "name": get_bot_name(bot_type)}, "error": str(e)[:100]}


# -----------------------------------------------------------------------------
# Mood Records（保留原版）
# -----------------------------------------------------------------------------
@app.post("/api/mood/create")
def create_mood_record(body: MoodRecordCreate, user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    mood_record = MoodRecord(user_id=user.id, mood=body.mood, intensity=body.intensity, note=body.note, created_at=datetime.utcnow())
    db.add(mood_record); db.commit(); db.refresh(mood_record)
    return {"ok": True, "mood_record_id": mood_record.id}

@app.get("/api/mood/me")
def my_mood_records(user: User = Depends(get_current_user), db: Session = Depends(get_db), limit: int = Query(default=10, le=100)):
    records = (db.query(MoodRecord).filter(MoodRecord.user_id == user.id).order_by(MoodRecord.created_at.desc()).limit(limit).all())
    return {"mood_records": [{"id": r.id, "mood": r.mood, "intensity": r.intensity, "note": r.note,
                               "created_at": r.created_at.isoformat() + "Z"} for r in records]}


# -----------------------------------------------------------------------------
# Chat Messages（保留原版）
# -----------------------------------------------------------------------------
@app.post("/api/chat/messages")
def create_chat_message(body: ChatMessageCreate, user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    chat_message = ChatMessage(
        user_id=user.id,
        content=body.content,
        role=body.role,
        bot_type=body.bot_type,
        mode=body.mode,
        user_mood=body.user_mood,
        mood_intensity=body.mood_intensity,
        created_at=datetime.utcnow(),
    )
    db.add(chat_message); db.commit(); db.refresh(chat_message)
    return {"ok": True, "message_id": chat_message.id}

@app.get("/api/chat/messages/me")
def my_chat_messages(user: User = Depends(get_current_user), db: Session = Depends(get_db),
                     limit: int = Query(default=20, le=100), bot_type: Optional[str] = Query(default=None)):
    query = db.query(ChatMessage).filter(ChatMessage.user_id == user.id)
    if bot_type:
        query = query.filter(ChatMessage.bot_type == bot_type)
    messages = query.order_by(ChatMessage.created_at.desc()).limit(limit).all()
    return {"messages": [{
        "id": m.id, "content": m.content, "role": m.role, "bot_type": m.bot_type, "mode": m.mode,
        "user_mood": m.user_mood, "mood_intensity": m.mood_intensity, "created_at": m.created_at.isoformat() + "Z"
    } for m in messages]}


# -----------------------------------------------------------------------------
# Chat Sessions（保留原版）
# -----------------------------------------------------------------------------
@app.post("/api/chat/sessions")
def create_chat_session(body: ChatSessionCreate, user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    existing_sessions = db.query(ChatSession).filter(ChatSession.user_id == user.id, ChatSession.is_active == True).all()
    for session in existing_sessions:
        session.is_active = False; session.session_end = datetime.utcnow(); session.end_reason = "user_ended"; db.add(session)
    new_session = ChatSession(user_id=user.id, bot_type=body.bot_type, session_start=datetime.utcnow(),
                              last_activity=datetime.utcnow(), is_active=True)
    db.add(new_session); db.commit(); db.refresh(new_session)
    return {"ok": True, "session_id": new_session.id}

@app.post("/api/chat/sessions/{session_id}/end")
def end_chat_session(session_id: int, body: ChatSessionEnd, user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    session = db.query(ChatSession).filter(ChatSession.id == session_id, ChatSession.user_id == user.id, ChatSession.is_active == True).first()
    if not session:
        raise HTTPException(status_code=404, detail="Active session not found")
    session.is_active = False; session.session_end = datetime.utcnow(); session.end_reason = body.reason
    db.add(session); db.commit()
    return {"ok": True, "session_ended": True}

@app.get("/api/chat/sessions/me")
def my_chat_sessions(user: User = Depends(get_current_user), db: Session = Depends(get_db), limit: int = Query(default=10, le=50)):
    sessions = (db.query(ChatSession).filter(ChatSession.user_id == user.id).order_by(ChatSession.session_start.desc()).limit(limit).all())
    return {"sessions": [{
        "id": s.id, "bot_type": s.bot_type,
        "session_start": s.session_start.isoformat() + "Z",
        "session_end": s.session_end.isoformat() + "Z" if s.session_end else None,
        "last_activity": s.last_activity.isoformat() + "Z" if s.last_activity else None,
        "is_active": s.is_active, "message_count": s.message_count, "end_reason": s.end_reason,
    } for s in sessions]}


# -----------------------------------------------------------------------------
# Admin: Allowed PIDs（保留原版）
# -----------------------------------------------------------------------------
@app.post("/api/admin/allowed-pids")
def create_allowed_pid(body: AllowedPidCreate, db: Session = Depends(get_db)):
    existing = db.query(AllowedPid).filter(AllowedPid.pid == body.pid).first()
    if existing:
        raise HTTPException(status_code=400, detail="PID already exists")
    allowed_pid = AllowedPid(pid=body.pid, description=body.description, is_active=True, created_at=datetime.utcnow())
    db.add(allowed_pid); db.commit(); db.refresh(allowed_pid)
    return {"ok": True, "allowed_pid_id": allowed_pid.id}

@app.get("/api/admin/allowed-pids")
def list_allowed_pids(db: Session = Depends(get_db), limit: int = Query(default=50, le=200)):
    pids = db.query(AllowedPid).order_by(AllowedPid.created_at.desc()).limit(limit).all()
    return {"allowed_pids": [{"id": p.id, "pid": p.pid, "description": p.description, "is_active": p.is_active,
                              "created_at": p.created_at.isoformat() + "Z"} for p in pids]}

@app.patch("/api/admin/allowed-pids/{pid_id}")
def update_allowed_pid(pid_id: int, body: AllowedPidUpdate, db: Session = Depends(get_db)):
    allowed_pid = db.query(AllowedPid).filter(AllowedPid.id == pid_id).first()
    if not allowed_pid:
        raise HTTPException(status_code=404, detail="Allowed PID not found")
    if body.is_active is not None:
        allowed_pid.is_active = body.is_active
    if body.description is not None:
        allowed_pid.description = body.description
    db.add(allowed_pid); db.commit()
    return {"ok": True}


# -----------------------------------------------------------------------------
# System Status & Root（保留原版）
# -----------------------------------------------------------------------------
@app.get("/api/system/status")
def system_status(db: Session = Depends(get_db)):
    try:
        total_users = db.query(User).count()
        total_assessments = db.query(Assessment).count()
        total_chat_messages = db.query(ChatMessage).count()
        active_sessions = db.query(ChatSession).filter(ChatSession.is_active == True).count()
        yesterday = datetime.utcnow() - timedelta(days=1)
        recent_messages = db.query(ChatMessage).filter(ChatMessage.created_at >= yesterday).count()
        recent_assessments = db.query(Assessment).filter(Assessment.created_at >= yesterday).count()
        return {
            "ok": True,
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "stats": {
                "total_users": total_users, "total_assessments": total_assessments,
                "total_chat_messages": total_chat_messages, "active_sessions": active_sessions,
                "recent_24h": {"messages": recent_messages, "assessments": recent_assessments}
            },
            "services": {
                "database": True,
                "openai": bool(os.getenv("OPENAI_API_KEY")),
                "heygen": bool(os.getenv("HEYGEN_API_KEY")),
                "chat_router": chat_router_loaded,
            }
        }
    except Exception as e:
        return {"ok": False, "error": str(e), "timestamp": datetime.utcnow().isoformat() + "Z"}

@app.get("/")
def root():
    return {
        "service": "Emobot Backend API",
        "version": "0.5.1",
        "status": "running",
        "docs": "/docs",
        "health": "/api/health",
        "chat_router_loaded": chat_router_loaded,
        "timestamp": datetime.utcnow().isoformat() + "Z",
    }
