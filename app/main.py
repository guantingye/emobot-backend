# app/main.py - 完整修正版
from __future__ import annotations

import os
import re
from datetime import datetime, timedelta, timezone
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
from app.models.mood import MoodRecord
from app.models.allowed_pid import AllowedPid
from app.models.chat_session import ChatSession

# 台灣時區
TW_TZ = timezone(timedelta(hours=8))

def get_tw_time():
    """取得台灣時間"""
    return datetime.now(TW_TZ)

# 路由註冊狀態
router_status = {
    "chat": {"loaded": False, "error": None},
    "avatar_animation": {"loaded": False, "error": None},
}

# 頭像動畫路由
try:
    from app.routers import avatar_animation
    router_status["avatar_animation"]["loaded"] = True
    print("✅ 頭像動畫模組載入成功")
except Exception as e:
    router_status["avatar_animation"]["error"] = str(e)
    print(f"❌ 頭像動畫模組錯誤: {e}")

# Chat 路由
try:
    from app import chat
    router_status["chat"]["loaded"] = True
    print("✅ Chat 模組載入成功")
except Exception as e:
    router_status["chat"]["error"] = str(e)
    print(f"❌ Chat 模組錯誤: {e}")

# ============================================================================
# FastAPI App 初始化
# ============================================================================

app = FastAPI(
    title="Emobot Backend",
    version="0.7.0",
    description="心理對話機器人系統"
)

# ============================================================================
# CORS 設定 - 強化版
# ============================================================================

ALLOWED = os.getenv(
    "ALLOWED_ORIGINS",
    "https://emobot-plus.vercel.app,http://localhost:5173,http://localhost:3000"
)

def _parse_allowed(origins_str: str) -> List[str]:
    out = []
    for s in (origins_str or "").split(","):
        s = s.strip()
        if s and s not in ("*", "null"):
            out.append(s)
    return out

_ALLOWED_ORIGINS = _parse_allowed(ALLOWED)
_VERCEL_REGEX_STR = r"^https://.*\.vercel\.app$"
_VERCEL_REGEX = re.compile(_VERCEL_REGEX_STR, re.IGNORECASE)

# 官方 CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=_ALLOWED_ORIGINS,
    allow_origin_regex=_VERCEL_REGEX_STR,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["*"],
)

# 自訂補強 middleware - 確保所有回應都有 CORS headers
@app.middleware("http")
async def force_cors_headers(request: Request, call_next):
    origin = request.headers.get("origin")
    is_allowed = bool(
        origin and (
            origin in _ALLOWED_ORIGINS or 
            _VERCEL_REGEX.match(origin or "")
        )
    )

    # 預檢請求
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

    # 一般請求
    try:
        resp = await call_next(request)
    except HTTPException as he:
        resp = JSONResponse({"detail": he.detail}, status_code=he.status_code)
    except Exception as e:
        print(f"Request error: {e}")
        resp = JSONResponse({"detail": "Internal Server Error"}, status_code=500)

    if is_allowed:
        resp.headers.setdefault("Access-Control-Allow-Origin", origin)
        resp.headers.setdefault("Access-Control-Allow-Credentials", "true")
        resp.headers.setdefault("Access-Control-Expose-Headers", "*")
        vary = resp.headers.get("Vary")
        resp.headers["Vary"] = "Origin" if not vary else (vary if "Origin" in vary else f"{vary}, Origin")
    
    return resp

# 啟動時建表
@app.on_event("startup")
def on_startup():
    try:
        Base.metadata.create_all(bind=engine)
        print("✅ 資料庫表格建立完成")
    except Exception as e:
        print(f"⚠️ 資料庫表格建立失敗: {e}")

# ============================================================================
# 路由註冊
# ============================================================================

if router_status["chat"]["loaded"]:
    app.include_router(chat.router, prefix="/api/chat", tags=["chat"])
    print("✅ Chat 路由註冊成功")

if router_status["avatar_animation"]["loaded"]:
    app.include_router(avatar_animation.router, prefix="/api/chat/avatar", tags=["avatar"])
    print("✅ 頭像動畫路由註冊成功")

# ============================================================================
# Pydantic Models
# ============================================================================

class JoinRequest(BaseModel):
    pid: str = Field(..., min_length=1, max_length=50)
    nickname: Optional[str] = Field(default=None, max_length=100)

class AssessmentUpsert(BaseModel):
    mbti_raw: Optional[str] = None
    mbti_encoded: Optional[List[float]] = None
    step2_answers: Optional[List[Any]] = None
    step3_answers: Optional[List[Any]] = None
    step4_answers: Optional[List[Any]] = None
    is_retest: Optional[bool] = False

class MatchChoice(BaseModel):
    bot_type: str = Field(..., description="empathy | insight | solution | cognitive")

# ============================================================================
# Helper Functions
# ============================================================================

def is_pid_allowed(pid: str, db: Session) -> bool:
    """檢查 PID 是否在允許清單中"""
    allowed_pid = db.query(AllowedPid).filter(
        AllowedPid.pid == pid,
        AllowedPid.is_active == True
    ).first()
    return allowed_pid is not None

def build_recommendation_payload(user: Dict, assessment: Dict) -> Dict:
    """簡化版推薦演算法"""
    # 預設分數
    scores = {
        "empathy": 0.25,
        "insight": 0.25,
        "solution": 0.25,
        "cognitive": 0.25,
    }
    
    ranked = sorted(
        [{"type": k, "score": round(v * 100, 2)} for k, v in scores.items()],
        key=lambda x: x["score"],
        reverse=True
    )
    
    return {
        "ok": True,
        "scores": scores,
        "ranked": ranked,
        "top": ranked[0],
        "algorithm_version": "v1.0"
    }

# ============================================================================
# Auth & Profile API
# ============================================================================

@app.post("/api/auth/join")
def join(body: JoinRequest, db: Session = Depends(get_db)):
    """登入/註冊 - 新增記錄登入時間"""
    pid = (body.pid or "").strip()
    if not pid:
        raise HTTPException(status_code=422, detail="pid is required")

    # 檢查 PID 是否在允許清單中
    if not is_pid_allowed(pid, db):
        raise HTTPException(
            status_code=403,
            detail="此 PID 未被授權使用系統,請聯繫管理員"
        )

    user = db.query(User).filter(User.pid == pid).first()
    
    if not user:
        # 新用戶
        user = User(
            pid=pid, 
            nickname=body.nickname or None,
            last_login_at=datetime.utcnow()  # 記錄登入時間
        )
        db.add(user)
        db.commit()
        db.refresh(user)
        print(f"✅ 新用戶註冊: user_id={user.id}, pid={pid}")
    else:
        # 現有用戶 - 更新暱稱和登入時間
        if body.nickname and user.nickname != body.nickname:
            user.nickname = body.nickname
        user.last_login_at = datetime.utcnow()  # 更新登入時間
        db.add(user)
        db.commit()
        db.refresh(user)
        print(f"✅ 用戶登入: user_id={user.id}, pid={pid}")

    token = create_access_token(user_id=user.id, pid=user.pid)
    
    return {
        "token": token,
        "user": {
            "id": user.id,
            "pid": user.pid,
            "nickname": user.nickname,
            "selected_bot": user.selected_bot,
            "last_login_at": user.last_login_at.isoformat() + "Z" if user.last_login_at else None
        }
    }

@app.get("/api/user/profile")
def profile(user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    """取得用戶資料"""
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
            "top": {
                "type": r.selected_bot or (ranked[0]["type"] if ranked else None), 
                "score": ranked[0]["score"] if ranked else 0
            },
            "selected_bot": r.selected_bot,
            "created_at": r.created_at.isoformat() + "Z",
        }

    return {
        "user": {
            "id": user.id, 
            "pid": user.pid, 
            "nickname": user.nickname, 
            "selected_bot": user.selected_bot,
            "last_login_at": user.last_login_at.isoformat() + "Z" if user.last_login_at else None
        },
        "latest_assessment_id": a.id if a else None,
        "latest_recommendation": latest_recommendation,
    }

# ============================================================================
# 測驗相關 API
# ============================================================================

@app.post("/api/assessments/upsert")
def upsert_assessment(
    body: AssessmentUpsert,
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """儲存測驗結果"""
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
    
    print(f"✅ 測驗結果已儲存: user_id={user.id}, assessment_id={a.id}")
    
    return {
        "ok": True, 
        "assessment_id": a.id, 
        "is_retest": body.is_retest or False
    }

@app.get("/api/assessments/me")
def my_assessment(user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    """取得我的測驗結果"""
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

# ============================================================================
# 推薦相關 API
# ============================================================================

@app.post("/api/match/recommend")
def recommend(
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """執行推薦演算法"""
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

    rec = Recommendation(
        user_id=user.id,
        scores=scores,
        selected_bot=None,
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
    """選擇機器人"""
    valid = {"empathy", "insight", "solution", "cognitive"}
    if body.bot_type not in valid:
        raise HTTPException(status_code=422, detail=f"Invalid bot_type")

    user.selected_bot = body.bot_type
    db.add(user)

    rec = (
        db.query(Recommendation)
        .filter(Recommendation.user_id == user.id)
        .order_by(Recommendation.id.desc())
        .first()
    )
    if rec:
        rec.selected_bot = body.bot_type
        db.add(rec)

    db.commit()
    
    print(f"✅ 用戶選擇機器人: user_id={user.id}, bot_type={body.bot_type}")

    return {"ok": True, "selected_bot": body.bot_type}

# ============================================================================
# 健康檢查
# ============================================================================

@app.get("/api/health")
def health():
    return {
        "ok": True,
        "time": datetime.utcnow().isoformat() + "Z",
        "version": "0.7.0",
        "features": {
            "chat_router": router_status["chat"]["loaded"],
            "avatar_animation": router_status["avatar_animation"]["loaded"],
        },
        "cors_enabled": True,
        "allowed_origins": _ALLOWED_ORIGINS
    }

@app.get("/")
def root():
    return {
        "service": "Emobot Backend API",
        "version": "0.7.0",
        "status": "running",
        "docs": "/docs",
        "health": "/api/health",
        "timestamp": datetime.utcnow().isoformat() + "Z",
    }

# ============================================================================
# 啟動
# ============================================================================

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", "10000"))
    uvicorn.run(app, host="0.0.0.0", port=port, log_level="info")