# app/main.py
from __future__ import annotations

import os
import re
from datetime import datetime, timedelta
from typing import Any

from fastapi import FastAPI, Depends, HTTPException, Query, Response, Request, Header
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
from starlette.exceptions import HTTPException as StarletteHTTPException
from sqlalchemy import text
from sqlalchemy.orm import Session
from jose import jwt, JWTError
import traceback

# ---- 環境變數 ----
DATABASE_URL = os.getenv("DATABASE_URL")
JWT_SECRET = os.getenv("JWT_SECRET") or os.getenv("SECRET_KEY") or "dev-secret-change-me"
JWT_ALG = os.getenv("JWT_ALG", "HS256")
JWT_EXPIRE_MINUTES = int(os.getenv("JWT_EXPIRE_MINUTES", "129600"))  # 90天
ALLOWED_ORIGINS = os.getenv(
    "ALLOWED_ORIGINS",
    "https://emobot-plus.vercel.app,http://localhost:5173,http://localhost:3000",
)
PORT = int(os.getenv("PORT", "8000"))

# ---- DB / Models（沿用你現有的專案結構）----
from app.db.session import get_db, engine  # type: ignore
from app.db.base import Base               # type: ignore

from app.models.user import User                    # type: ignore
from app.models.assessment import Assessment        # type: ignore
from app.models.recommendation import Recommendation# type: ignore
from app.models.chat import ChatMessage             # type: ignore
from app.models.mood import MoodRecord              # type: ignore

# ====================== 應用初始化與 CORS ======================
app = FastAPI(
    title="Emobot+ API", 
    version="1.0.0",
    description="Emobot+ Backend API with enhanced CORS support"
)

# 關閉自動尾斜線轉向，避免 307/308 沒帶 CORS 標頭
app.router.redirect_slashes = False

# 解析允許的來源
_allowed_list = [o.strip() for o in ALLOWED_ORIGINS.split(",") if o.strip()]
print(f"🌐 CORS allowed origins: {_allowed_list}")
print(f"🚀 Starting server on port: {PORT}")

# 標準 CORS 中介軟體（最寬鬆設定）
app.add_middleware(
    CORSMiddleware,
    allow_origins=_allowed_list,
    allow_origin_regex=r"https://.*\.vercel\.app$",
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["*"],
    max_age=86400,
)

# 創建 CORS 標頭的輔助函數
def create_cors_headers(origin: str = None) -> dict:
    headers = {
        "Access-Control-Allow-Methods": "GET, POST, PUT, DELETE, OPTIONS, PATCH",
        "Access-Control-Allow-Headers": "*",
        "Access-Control-Max-Age": "86400",
    }
    
    if origin and (origin in _allowed_list or re.match(r"^https://.*\.vercel\.app$", origin)):
        headers["Access-Control-Allow-Origin"] = origin
        headers["Access-Control-Allow-Credentials"] = "true"
    
    return headers

# 全域錯誤處理中介軟體 - 確保所有錯誤回應都包含 CORS 標頭
@app.middleware("http")
async def catch_all_errors(request: Request, call_next):
    origin = request.headers.get("origin")
    
    # 處理 preflight OPTIONS 請求
    if request.method == "OPTIONS":
        cors_headers = create_cors_headers(origin)
        return Response(status_code=200, headers=cors_headers)
    
    try:
        response = await call_next(request)
        
        # 為所有成功的回應添加 CORS 標頭
        if origin:
            cors_headers = create_cors_headers(origin)
            for key, value in cors_headers.items():
                response.headers[key] = value
            
        return response
        
    except Exception as e:
        # 捕獲所有未處理的異常，返回帶有 CORS 標頭的錯誤回應
        print(f"❌ Unhandled error: {str(e)}")
        print(f"📍 Traceback: {traceback.format_exc()}")
        
        cors_headers = create_cors_headers(origin)
        return JSONResponse(
            status_code=500,
            content={
                "detail": f"Internal server error: {str(e)}",
                "error": "server_error"
            },
            headers=cors_headers
        )

# 自訂錯誤處理器 - 確保驗證錯誤也有 CORS 標頭
@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    origin = request.headers.get("origin")
    cors_headers = create_cors_headers(origin)
    
    return JSONResponse(
        status_code=422,
        content={
            "detail": exc.errors(),
            "body": exc.body,
            "error": "validation_error"
        },
        headers=cors_headers
    )

@app.exception_handler(StarletteHTTPException)
async def http_exception_handler(request: Request, exc: StarletteHTTPException):
    origin = request.headers.get("origin")
    cors_headers = create_cors_headers(origin)
    
    return JSONResponse(
        status_code=exc.status_code,
        content={"detail": exc.detail, "error": "http_error"},
        headers=cors_headers
    )

# 啟動建表（若你用 Alembic，可移除此段）
@app.on_event("startup")
def on_startup():
    try:
        Base.metadata.create_all(bind=engine)
        print("✅ Database tables created successfully")
    except Exception as e:
        print(f"❌ Database initialization error: {e}")
        print(f"📍 Database URL: {DATABASE_URL}")

# ====================== JWT 工具 ======================
def create_access_token_for_user(user: User) -> str:
    """產出與前端相容的 token：同時包含 sub 與 id"""
    exp = datetime.utcnow() + timedelta(minutes=JWT_EXPIRE_MINUTES)
    payload = {
        "sub": str(user.id),
        "id": user.id,
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

    try:
        claims = jwt.decode(token, JWT_SECRET, algorithms=[JWT_ALG])
        uid_raw = claims.get("sub") or claims.get("id") or claims.get("user_id")
        user_id = int(uid_raw) if uid_raw is not None else None
    except JWTError as e:
        print(f"JWT decode error: {e}")
        raise HTTPException(status_code=401, detail="Invalid token")

    if not user_id:
        raise HTTPException(status_code=401, detail="Invalid token payload")

    user = db.get(User, user_id)
    if not user:
        raise HTTPException(status_code=401, detail="User not found")
    return user

# ====================== 根路由和健康檢查 ======================
@app.get("/")
def read_root():
    return {
        "message": "Emobot+ API is running!",
        "version": "1.0.0",
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat()
    }

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
        "status": "healthy",
        "time": datetime.utcnow().isoformat() + "Z",
        "database": db_status,
        "allowed_origins": _allowed_list,
        "cors_enabled": True,
        "port": PORT,
    }

# ====================== Auth ======================
@app.post("/api/auth/join")
def join(payload: dict, db: Session = Depends(get_db)):
    """
    body: { "pid": "12AB", "nickname": "ting" }
    回傳：{ token, user }
    """
    try:
        print(f"📝 Join request: {payload}")
        
        pid = (payload.get("pid") or "").strip()
        nickname = (payload.get("nickname") or "").strip()
        
        if not pid:
            raise HTTPException(status_code=422, detail="pid is required")

        user = db.query(User).filter(User.pid == pid).first()
        if not user:
            user = User(pid=pid, nickname=nickname or "user")
            db.add(user)
            db.commit()
            db.refresh(user)
            print(f"✅ Created new user: {user.id}")
        else:
            if nickname and nickname != user.nickname:
                user.nickname = nickname
                db.commit()
                db.refresh(user)
                print(f"✅ Updated user: {user.id}")

        token = create_access_token_for_user(user)
        result = {
            "token": token,
            "user": {
                "id": user.id,
                "pid": user.pid,
                "nickname": user.nickname,
                "selected_bot": user.selected_bot,
            },
        }
        
        print(f"✅ Join successful for user: {user.id}")
        return result
        
    except HTTPException:
        raise
    except Exception as e:
        print(f"❌ Join error: {e}")
        print(f"📍 Traceback: {traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=f"Server error: {str(e)}")

# ====================== User Profile（前端用） ======================
@app.get("/api/user/profile")
def user_profile(user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    try:
        print(f"📋 Getting profile for user: {user.id}")
        
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
    except Exception as e:
        print(f"❌ Profile error: {e}")
        raise HTTPException(status_code=500, detail=f"Server error: {str(e)}")

# ====================== 修復的 Assessments Upsert ======================
@app.post("/api/assessments/upsert")
def upsert_assessment(
    body: dict,
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """
    支援多種前端資料格式：
    - MBTI: mbti_raw (string) 和 mbti_encoded (array)
    - 測驗答案: step2Answers, step3Answers, step4Answers (arrays)
    - 時間: submittedAt (ISO string)
    """
    try:
        print(f"💾 Saving assessment for user {user.id}")
        print(f"📤 Request body: {str(body)[:500]}...")
        
        # 處理 MBTI 資料
        mbti_raw = body.get("mbti_raw")
        mbti_encoded = body.get("mbti_encoded")
        
        # 處理測驗答案 - 轉換 array 為 dict 格式
        step2 = body.get("step2Answers")
        step3 = body.get("step3Answers") 
        step4 = body.get("step4Answers")
        
        # 將 array 格式轉為 dict 格式儲存
        if isinstance(step2, list):
            step2 = {"answers": step2}
        if isinstance(step3, list):
            step3 = {"answers": step3}
        if isinstance(step4, list):
            step4 = {"answers": step4}
            
        ai_pref = body.get("ai_preference")
        submitted_at = body.get("submittedAt")

        # 解析時間
        dt = None
        if isinstance(submitted_at, str):
            try:
                # 處理各種 ISO 8601 格式
                if submitted_at.endswith('Z'):
                    dt = datetime.fromisoformat(submitted_at.replace('Z', '+00:00'))
                else:
                    dt = datetime.fromisoformat(submitted_at)
            except Exception as e:
                print(f"⚠️ Time parse error: {e}")
                dt = datetime.utcnow()

        # 處理 mbti_encoded 格式
        encoded_data = None
        if isinstance(mbti_encoded, list):
            encoded_data = {"encoded": mbti_encoded}
        elif isinstance(mbti_encoded, dict):
            encoded_data = mbti_encoded

        # 創建或更新評估記錄
        record = Assessment(
            user_id=user.id,
            mbti_raw=(mbti_raw.upper().strip() if isinstance(mbti_raw, str) else None),
            mbti_encoded=encoded_data,
            step2_answers=step2,
            step3_answers=step3,
            step4_answers=step4,
            ai_preference=ai_pref,
            submitted_at=dt or datetime.utcnow(),
        )
        
        db.add(record)
        db.commit()
        db.refresh(record)

        print(f"✅ Assessment saved successfully: ID {record.id}")
        
        return {
            "success": True,
            "id": record.id,
            "mbti_raw": record.mbti_raw,
            "mbti_encoded": record.mbti_encoded,
            "submitted_at": record.submitted_at.isoformat() if record.submitted_at else None,
        }
        
    except HTTPException:
        raise
    except Exception as e:
        print(f"❌ Assessment upsert error: {e}")
        print(f"📍 Traceback: {traceback.format_exc()}")
        print(f"📤 Request body was: {body}")
        raise HTTPException(status_code=500, detail=f"Save assessment failed: {str(e)}")

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
    try:
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
    except HTTPException:
        raise
    except Exception as e:
        print(f"❌ Recommend error: {e}")
        raise HTTPException(status_code=500, detail=f"Recommendation failed: {str(e)}")

@app.post("/api/match/choose")
def choose_bot(payload: dict, user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    try:
        bot_type = (payload.get("botType") or "").strip().lower()
        if bot_type not in VALID_BOTS:
            raise HTTPException(status_code=400, detail="Invalid botType")
        user.selected_bot = bot_type
        db.commit()
        return {"ok": True, "selected_bot": bot_type}
    except HTTPException:
        raise
    except Exception as e:
        print(f"❌ Choose bot error: {e}")
        raise HTTPException(status_code=500, detail=f"Choose bot failed: {str(e)}")

# ====================== Chat ======================
@app.post("/api/chat/messages")
def create_message(body: dict, user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    try:
        msg = ChatMessage(
            user_id=user.id,
            message_type=(body.get("message_type") or "user")[:8],
            bot_type=(body.get("bot_type") or None),
            content=str(body.get("content") or ""),
            user_mood=(body.get("user_mood") or None),
            mood_intensity=int(body.get("mood_intensity")) if body.get("mood_intensity") is not None else None,
        )
        db.add(msg)
        db.commit()
        db.refresh(msg)
        return {
            "id": msg.id,
            "message_type": msg.message_type,
            "bot_type": msg.bot_type,
            "content": msg.content,
            "user_mood": msg.user_mood,
            "mood_intensity": msg.mood_intensity,
            "created_at": msg.created_at.isoformat(),
        }
    except Exception as e:
        print(f"❌ Create message error: {e}")
        raise HTTPException(status_code=500, detail=f"Create message failed: {str(e)}")

@app.get("/api/chat/messages")
def list_messages(limit: int = Query(50, ge=1, le=200), user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    try:
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
    except Exception as e:
        print(f"❌ List messages error: {e}")
        raise HTTPException(status_code=500, detail=f"List messages failed: {str(e)}")

# ====================== Mood ======================
@app.post("/api/mood/records")
def create_mood(body: dict, user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    try:
        rec = MoodRecord(
            user_id=user.id,
            mood=str(body.get("mood") or ""),
            intensity=int(body.get("intensity")) if body.get("intensity") is not None else None,
            note=(body.get("note") or None),
        )
        db.add(rec)
        db.commit()
        db.refresh(rec)
        return {
            "id": rec.id,
            "mood": rec.mood,
            "intensity": rec.intensity,
            "note": rec.note,
            "created_at": rec.created_at.isoformat(),
        }
    except Exception as e:
        print(f"❌ Create mood error: {e}")
        raise HTTPException(status_code=500, detail=f"Create mood failed: {str(e)}")

@app.get("/api/mood/records")
def list_mood(days: int = Query(30, ge=1, le=180), user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    try:
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
    except Exception as e:
        print(f"❌ List mood error: {e}")
        raise HTTPException(status_code=500, detail=f"List mood failed: {str(e)}")

# Debug endpoints
@app.get("/api/debug/db-test")
def debug_db_test(db: Session = Depends(get_db)):
    try:
        result = db.execute(text("SELECT 1 as test")).fetchone()
        return {"ok": True, "result": result[0] if result else None}
    except Exception as e:
        return {"ok": False, "error": str(e)}

@app.get("/api/assessments/me")
def get_my_assessment(user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    try:
        assessment = (
            db.query(Assessment)
            .filter(Assessment.user_id == user.id)
            .order_by(Assessment.id.desc())
            .first()
        )
        if not assessment:
            return None
        return {
            "id": assessment.id,
            "mbti_raw": assessment.mbti_raw,
            "mbti_encoded": assessment.mbti_encoded,
            "step2_answers": assessment.step2_answers,
            "step3_answers": assessment.step3_answers,
            "step4_answers": assessment.step4_answers,
            "submitted_at": assessment.submitted_at.isoformat() if assessment.submitted_at else None,
        }
    except Exception as e:
        print(f"❌ Get my assessment error: {e}")
        raise HTTPException(status_code=500, detail=f"Get assessment failed: {str(e)}")

@app.get("/api/match/me")
def get_my_match_choice(user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    try:
        reco = (
            db.query(Recommendation)
            .filter(Recommendation.user_id == user.id)
            .order_by(Recommendation.id.desc())
            .first()
        )
        return {
            "selected_bot": user.selected_bot,
            "latest_recommendation": ({
                "top_bot": reco.top_bot,
                "scores": reco.scores,
                "features": reco.features,
            } if reco else None),
        }
    except Exception as e:
        print(f"❌ Get my match choice error: {e}")
        raise HTTPException(status_code=500, detail=f"Get match choice failed: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=PORT)