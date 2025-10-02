# backend/app/main.py - 修正 CORS 和端點路徑
from __future__ import annotations

import os
import re
from datetime import datetime, timedelta
from typing import Optional, List, Dict, Any

from fastapi import FastAPI, Depends, HTTPException, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

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
from app.models.allowed_pid import AllowedPid

# 路由註冊狀態
router_status = {
    "chat": {"loaded": False, "error": None},
    "avatar_animation": {"loaded": False, "error": None}
}

# 載入 Chat 路由
try:
    from app import chat as chat_module
    if not hasattr(chat_module, 'router'):
        raise ImportError("chat.py 中沒有定義 router")
    router_status["chat"]["loaded"] = True
    print("✅ Chat 模組載入成功")
except Exception as e:
    router_status["chat"]["error"] = str(e)
    print(f"❌ Chat 模組載入失敗: {e}")

# 載入頭像動畫路由
try:
    from app.routers import avatar_animation
    router_status["avatar_animation"]["loaded"] = True
    print("✅ 頭像動畫模組載入成功")
except Exception as e:
    router_status["avatar_animation"]["error"] = str(e)
    print(f"⚠️ 頭像動畫模組載入失敗: {e}")

# 推薦引擎
try:
    from app.services.recommendation_engine import recommend_endpoint_payload as _build_reco
except Exception:
    _build_reco = None

def _fallback_build_reco(user: Dict[str, Any] | None, assessment: Dict[str, Any] | None) -> Dict[str, Any]:
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

# FastAPI App
app = FastAPI(
    title="Emobot Backend",
    version="0.7.0",
    description="心理對話機器人系統 - 以 PID 為主鍵"
)

# CORS 設定（強化版）
ALLOWED = os.getenv(
    "ALLOWED_ORIGINS",
    "https://emobot-plus.vercel.app,http://localhost:5173,http://localhost:3000"
)

def _parse_allowed(origins_str: str) -> List[str]:
    out: List[str] = []
    for s in (origins_str or "").split(","):
        s = s.strip()
        if s and s not in ("*", "null"):
            out.append(s)
    return out

_ALLOWED_ORIGINS = _parse_allowed(ALLOWED)
_VERCEL_REGEX_STR = r"^https://.*\.vercel\.app$"
_VERCEL_REGEX = re.compile(_VERCEL_REGEX_STR, re.IGNORECASE)

# 第一層：標準 CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=_ALLOWED_ORIGINS,
    allow_origin_regex=_VERCEL_REGEX_STR,
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "PATCH", "DELETE", "OPTIONS"],
    allow_headers=["*"],
    expose_headers=["*"],
    max_age=86400,
)

# 第二層：確保 OPTIONS 正確處理
@app.middleware("http")
async def cors_middleware(request: Request, call_next):
    origin = request.headers.get("origin", "")
    is_allowed = origin in _ALLOWED_ORIGINS or (_VERCEL_REGEX.match(origin) if origin else False)
    
    # 處理 OPTIONS preflight
    if request.method == "OPTIONS":
        headers = {
            "Access-Control-Allow-Origin": origin if is_allowed else "",
            "Access-Control-Allow-Credentials": "true",
            "Access-Control-Allow-Methods": "GET, POST, PUT, PATCH, DELETE, OPTIONS",
            "Access-Control-Allow-Headers": "Authorization, Content-Type, X-Requested-With, Accept",
            "Access-Control-Max-Age": "86400",
            "Vary": "Origin",
        }
        return Response(status_code=200, headers=headers)
    
    # 處理正常請求
    try:
        response = await call_next(request)
    except Exception as e:
        print(f"Request error: {e}")
        response = JSONResponse(
            {"detail": "Internal Server Error"},
            status_code=500
        )
    
    # 添加 CORS headers
    if is_allowed:
        response.headers["Access-Control-Allow-Origin"] = origin
        response.headers["Access-Control-Allow-Credentials"] = "true"
        response.headers["Access-Control-Expose-Headers"] = "*"
        response.headers["Vary"] = "Origin"
    
    return response

@app.on_event("startup")
def on_startup():
    try:
        Base.metadata.create_all(bind=engine)
        print("✅ 資料表初始化完成")
    except Exception as e:
        print(f"⚠️ 資料表建立失敗：{e}")

# 路由註冊
if router_status["avatar_animation"]["loaded"]:
    try:
        app.include_router(avatar_animation.router, prefix="/api/chat/avatar", tags=["avatar-animation"])
        print("✅ 頭像動畫路由註冊成功")
    except Exception as e:
        print(f"❌ 頭像動畫路由註冊失敗: {e}")

if router_status["chat"]["loaded"]:
    try:
        app.include_router(chat_module.router, prefix="/api/chat", tags=["chat"])
        print("✅ Chat 路由註冊成功")
    except Exception as e:
        print(f"❌ Chat 路由註冊失敗: {e}")

# Pydantic Models
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

# Helper Functions
def is_pid_allowed(pid: str, db: Session) -> bool:
    allowed_pid = db.query(AllowedPid).filter(
        AllowedPid.pid == pid,
        AllowedPid.is_active == True
    ).first()
    return allowed_pid is not None

# ========== 認證端點 ==========

@app.post("/api/auth/join")
def join(body: JoinRequest, db: Session = Depends(get_db)):
    pid = (body.pid or "").strip()
    if not pid:
        raise HTTPException(status_code=422, detail="pid is required")

    if not is_pid_allowed(pid, db):
        raise HTTPException(
            status_code=403,
            detail="此 PID 未被授權使用系統，請聯繫管理員"
        )

    user = db.query(User).filter(User.pid == pid).first()
    if not user:
        user = User(pid=pid, nickname=body.nickname or None)
        db.add(user)
    else:
        if body.nickname and user.nickname != body.nickname:
            user.nickname = body.nickname
        user.last_login_at = datetime.utcnow()
        db.add(user)
    
    db.commit()
    db.refresh(user)

    token = create_access_token(pid=user.pid)
    return {
        "token": token,
        "user": {
            "pid": user.pid,
            "nickname": user.nickname,
            "selected_bot": user.selected_bot
        }
    }

@app.get("/api/user/me")
def get_me(user: User = Depends(get_current_user)):
    return {
        "ok": True,
        "user": {
            "pid": user.pid,
            "nickname": user.nickname,
            "selected_bot": user.selected_bot,
            "last_login_at": user.last_login_at.isoformat() + "Z" if user.last_login_at else None,
            "created_at": user.created_at.isoformat() + "Z" if user.created_at else None
        }
    }

@app.patch("/api/user/me")
def update_me(
    nickname: Optional[str] = None,
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    if nickname:
        user.nickname = nickname
        db.add(user)
        db.commit()
        db.refresh(user)
    return {"ok": True, "user": {"pid": user.pid, "nickname": user.nickname}}

# ========== Assessment 端點（注意：複數形式 assessments）==========

@app.post("/api/assessments/upsert")
def upsert_assessment(
    body: AssessmentUpsert,
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """儲存測驗資料（使用 PID）"""
    print(f"📝 Saving assessment for PID={user.pid}")
    
    # 如果是重新測驗，清除 selected_bot
    if body.is_retest:
        user.selected_bot = None
        db.add(user)
    
    # 查找或建立 assessment
    a = db.query(Assessment).filter(Assessment.pid == user.pid).first()
    
    if not a:
        a = Assessment(pid=user.pid)
        db.add(a)
    
    # 更新欄位
    if body.mbti_raw is not None:
        a.mbti_raw = body.mbti_raw
    if body.mbti_encoded is not None:
        a.mbti_encoded = body.mbti_encoded
    if body.step2_answers is not None:
        a.step2_answers = body.step2_answers
    if body.step3_answers is not None:
        a.step3_answers = body.step3_answers
    if body.step4_answers is not None:
        a.step4_answers = body.step4_answers
    if body.ai_preference is not None:
        a.ai_preference = body.ai_preference
    
    db.commit()
    db.refresh(a)
    
    print(f"✅ Assessment saved: id={a.id}, PID={user.pid}")
    
    return {
        "ok": True,
        "assessment_id": a.id,
        "is_retest": body.is_retest or False
    }

@app.get("/api/assessments/me")
def get_my_assessment(
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """取得測驗資料"""
    a = db.query(Assessment).filter(Assessment.pid == user.pid).first()
    if not a:
        return {"assessment": None}
    
    return {
        "assessment": {
            "id": a.id,
            "pid": a.pid,
            "mbti_raw": a.mbti_raw,
            "mbti_encoded": a.mbti_encoded,
            "step2_answers": a.step2_answers,
            "step3_answers": a.step3_answers,
            "step4_answers": a.step4_answers,
            "created_at": a.created_at.isoformat() + "Z"
        }
    }

# ========== Match/Recommendation 端點 ==========

@app.post("/api/match/recommend")
def recommend(
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    a = db.query(Assessment).filter(Assessment.pid == user.pid).first()
    if not a:
        raise HTTPException(status_code=400, detail="No assessment found")

    user_payload = {"pid": user.pid, "nickname": user.nickname}
    assess_payload = {
        "id": a.id,
        "mbti_raw": a.mbti_raw,
        "mbti_encoded": a.mbti_encoded,
        "step2_answers": a.step2_answers,
        "step3_answers": a.step3_answers,
        "step4_answers": a.step4_answers
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

    rec = Recommendation(
        pid=user.pid,
        scores=scores,
        selected_bot=None,
        created_at=datetime.utcnow()
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
        "algorithm_version": result.get("algorithm_version")
    }

@app.post("/api/match/choose")
def choose_bot(
    body: MatchChoice,
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    valid = {"empathy", "insight", "solution", "cognitive"}
    if body.bot_type not in valid:
        raise HTTPException(status_code=422, detail=f"Invalid bot_type")

    user.selected_bot = body.bot_type
    db.add(user)
    db.commit()

    latest_rec = (
        db.query(Recommendation)
        .filter(Recommendation.pid == user.pid)
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
        .filter(Recommendation.pid == user.pid)
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
            "top": {
                "type": rec.selected_bot or (ranked[0]["type"] if ranked else None),
                "score": ranked[0]["score"] if ranked else 0
            },
            "selected_bot": rec.selected_bot,
            "created_at": rec.created_at.isoformat() + "Z"
        }
    }

# ========== Health & Debug ==========

@app.get("/api/health")
def health():
    return {
        "ok": True,
        "time": datetime.utcnow().isoformat() + "Z",
        "version": "0.7.0",
        "features": {
            "chat_router": router_status["chat"]["loaded"],
            "avatar_animation": router_status["avatar_animation"]["loaded"]
        },
        "errors": {k: v["error"] for k, v in router_status.items() if v["error"]}
    }

@app.get("/api/debug/db-test")
def db_test(db: Session = Depends(get_db)):
    try:
        db.execute(text("select 1"))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"DB not ready: {e}")
    return {"ok": True, "message": "Database connection successful"}

@app.get("/api/system/status")
def system_status(db: Session = Depends(get_db)):
    try:
        total_users = db.query(User).count()
        total_assessments = db.query(Assessment).count()
        total_chat_messages = db.query(ChatMessage).count()

        return {
            "ok": True,
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "version": "0.7.0",
            "stats": {
                "total_users": total_users,
                "total_assessments": total_assessments,
                "total_chat_messages": total_chat_messages
            }
        }
    except Exception as e:
        return {"ok": False, "error": str(e)}

@app.get("/")
def root():
    return {
        "service": "Emobot Backend API",
        "version": "0.7.0",
        "status": "running",
        "docs": "/docs",
        "health": "/api/health",
        "timestamp": datetime.utcnow().isoformat() + "Z"
    }

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", "10000"))
    uvicorn.run(app, host="0.0.0.0", port=port, log_level="info")