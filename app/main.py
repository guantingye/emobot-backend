# app/main.py - 增強版，支援完整用戶流程
import os
import re
from datetime import datetime, timedelta
from typing import Optional, List, Dict, Any

from fastapi import FastAPI, Depends, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, validator
from sqlalchemy import text, and_
from sqlalchemy.orm import Session

# Core imports
from app.core.config import settings
from app.core.security import create_access_token, get_current_user
from app.db.session import get_db, engine
from app.db.base import Base

# Models
from app.models.user import User
from app.models.assessment import Assessment  
from app.models.recommendation import Recommendation
from app.models.chat import ChatMessage
from app.models.mood import MoodRecord

# Chat router
from app.chat import router as chat_router

# ============================================================================
# FastAPI App
# ============================================================================

app = FastAPI(title="Emobot Backend", version="1.0.0")

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

@app.on_event("startup")
async def startup():
    try:
        Base.metadata.create_all(bind=engine)
        print("✅ Database tables created")
    except Exception as e:
        print(f"❌ Startup error: {e}")

app.include_router(chat_router)

# ============================================================================
# Schemas with Enhanced Validation
# ============================================================================

class JoinRequest(BaseModel):
    pid: str = Field(..., min_length=4, max_length=4)
    nickname: str = Field(..., min_length=2, max_length=20)
    
    @validator('pid')
    def validate_pid_format(cls, v):
        v = v.strip().upper()
        if not re.match(r'^\d{3}[A-Z]{1}$', v):
            raise ValueError('PID 格式必須為三位數字＋一位英文大寫字母（例：123A）')
        return v
    
    @validator('nickname')
    def validate_nickname(cls, v):
        v = v.strip()
        if len(v) < 2:
            raise ValueError('暱稱至少需要2個字元')
        if len(v) > 20:
            raise ValueError('暱稱不能超過20個字元')
        return v

class UserStatusResponse(BaseModel):
    user: Dict[str, Any]
    has_assessment: bool
    has_recommendation: bool
    user_flow_stage: str  # "new", "assessed", "recommended", "active"
    next_route: str

# ============================================================================
# Helper Functions
# ============================================================================

def determine_user_flow_stage(user: User, db: Session) -> Dict[str, Any]:
    """
    判斷用戶所處的流程階段
    """
    # 檢查是否有測驗記錄
    assessment = db.query(Assessment).filter(Assessment.user_id == user.id).first()
    
    # 檢查是否有推薦記錄
    recommendation = db.query(Recommendation).filter(Recommendation.user_id == user.id).first()
    
    # 判斷流程階段和下一步路由
    if user.selected_bot:
        # 已選擇機器人 → 進入會員專區
        stage = "active"
        next_route = "/dashboard"
    elif recommendation:
        # 有推薦但未選擇 → 進入選擇頁面
        stage = "recommended" 
        next_route = "/choose-bot"
    elif assessment:
        # 有測驗但無推薦 → 需要生成推薦
        stage = "assessed"
        next_route = "/matching"
    else:
        # 新用戶 → 進入測驗
        stage = "new"
        next_route = "/test"
    
    return {
        "user_flow_stage": stage,
        "next_route": next_route,
        "has_assessment": bool(assessment),
        "has_recommendation": bool(recommendation)
    }

def _fallback_build_reco(user: Dict[str, Any] | None, assessment: Dict[str, Any] | None) -> Dict[str, Any]:
    """簡化的推薦算法"""
    empathy = insight = solution = cognitive = 0.25
    if assessment and assessment.get("mbti_encoded"):
        enc = assessment["mbti_encoded"]
        if isinstance(enc, (list, tuple)) and len(enc) >= 4:
            def norm(v):
                try:
                    v = float(v)
                    return min(max(v / 100.0 if v > 1 else v, 0.0), 1.0)
                except:
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
# Routes
# ============================================================================

@app.get("/")
async def root():
    return {"message": "Emobot Backend", "version": "1.0.0", "status": "enhanced"}

@app.get("/api/health") 
async def health():
    return {"ok": True, "version": "1.0.0", "time": datetime.utcnow().isoformat() + "Z"}

@app.get("/api/debug/simple-test")
async def simple_test():
    return {"ok": True, "message": "Enhanced version running"}

# ============================================================================
# Enhanced Authentication
# ============================================================================

@app.post("/api/auth/join")
async def join(body: JoinRequest, db: Session = Depends(get_db)):
    try:
        # PID 和 nickname 已通過 Pydantic 驗證
        pid = body.pid
        nickname = body.nickname
        
        # 查找或創建用戶
        user = db.query(User).filter(User.pid == pid).first()
        if not user:
            user = User(pid=pid, nickname=nickname)
            db.add(user)
            db.commit()
            db.refresh(user)
            print(f"✅ New user created: {pid}")
        else:
            # 更新暱稱（允許用戶修改顯示名稱）
            if user.nickname != nickname:
                user.nickname = nickname
                db.add(user)
                db.commit()
                db.refresh(user)
            print(f"✅ Existing user logged in: {pid}")
        
        # 判斷用戶流程狀態
        flow_info = determine_user_flow_stage(user, db)
        
        # 生成 JWT token
        token = create_access_token(user_id=user.id, pid=user.pid)
        
        # 返回完整用戶資訊和流程狀態
        return {
            "token": token,
            "user": {
                "id": user.id,
                "pid": user.pid,
                "nickname": user.nickname,
                "selected_bot": user.selected_bot,
                "created_at": user.created_at.isoformat() + "Z" if user.created_at else None
            },
            **flow_info
        }
        
    except ValueError as ve:
        # Pydantic 驗證錯誤
        raise HTTPException(status_code=422, detail=str(ve))
    except Exception as e:
        print(f"❌ Join error: {e}")
        db.rollback()
        raise HTTPException(status_code=500, detail="登入失敗，請稍後再試")

@app.get("/api/user/status")
async def user_status(user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    """
    獲取用戶完整狀態，用於前端路由決策
    """
    try:
        flow_info = determine_user_flow_stage(user, db)
        
        return UserStatusResponse(
            user={
                "id": user.id,
                "pid": user.pid, 
                "nickname": user.nickname,
                "selected_bot": user.selected_bot
            },
            **flow_info
        )
    except Exception as e:
        print(f"❌ User status error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/user/profile")
async def profile(user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    try:
        flow_info = determine_user_flow_stage(user, db)
        return {
            "user": {
                "id": user.id,
                "pid": user.pid,
                "nickname": user.nickname, 
                "selected_bot": user.selected_bot
            },
            **flow_info
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# ============================================================================
# Assessment Flow
# ============================================================================

@app.post("/api/assessments/upsert")
async def upsert_assessment(
    body: dict,
    user: User = Depends(get_current_user), 
    db: Session = Depends(get_db)
):
    try:
        # 檢查是否已有測驗記錄
        existing = db.query(Assessment).filter(Assessment.user_id == user.id).first()
        if existing:
            # 更新現有記錄
            for key, value in body.items():
                if hasattr(existing, key):
                    setattr(existing, key, value)
            existing.submitted_at = body.get('submittedAt') or datetime.utcnow()
            db.add(existing)
            assessment_id = existing.id
        else:
            # 創建新記錄
            a = Assessment(
                user_id=user.id,
                mbti_raw=body.get('mbti_raw'),
                mbti_encoded=body.get('mbti_encoded'),
                step2_answers=body.get('step2_answers'),
                step3_answers=body.get('step3_answers'),
                step4_answers=body.get('step4_answers'),
                ai_preference=body.get('ai_preference'),
                submitted_at=body.get('submittedAt') or datetime.utcnow()
            )
            db.add(a)
            db.commit()
            db.refresh(a)
            assessment_id = a.id
            
        db.commit()
        print(f"✅ Assessment saved for user {user.pid}")
        return {"ok": True, "assessment_id": assessment_id}
        
    except Exception as e:
        print(f"❌ Assessment error: {e}")
        db.rollback()
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/assessments/me")
async def my_assessment(user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    try:
        a = db.query(Assessment).filter(Assessment.user_id == user.id).order_by(Assessment.id.desc()).first()
        if not a:
            return {"assessment": None}
        return {
            "assessment": {
                "id": a.id,
                "mbti_raw": a.mbti_raw,
                "mbti_encoded": a.mbti_encoded,
                "submitted_at": a.submitted_at.isoformat() + "Z" if a.submitted_at else None
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# ============================================================================
# Matching & Bot Selection
# ============================================================================

@app.post("/api/match/recommend")
async def recommend(user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    try:
        # 獲取最新測驗結果
        a = db.query(Assessment).filter(Assessment.user_id == user.id).order_by(Assessment.id.desc()).first()
        if not a:
            raise HTTPException(status_code=400, detail="請先完成心理測驗")

        # 生成推薦
        result = _fallback_build_reco(
            {"id": user.id, "pid": user.pid},
            {
                "mbti_encoded": a.mbti_encoded,
                "step2_answers": a.step2_answers,
                "step3_answers": a.step3_answers,
                "step4_answers": a.step4_answers
            }
        )
        
        # 儲存推薦結果
        existing_rec = db.query(Recommendation).filter(Recommendation.user_id == user.id).first()
        if existing_rec:
            existing_rec.scores = result["scores"]
            existing_rec.ranked = result["ranked"] 
            existing_rec.selected_bot = result["top"]["type"]
            db.add(existing_rec)
        else:
            rec = Recommendation(
                user_id=user.id,
                assessment_id=a.id,
                selected_bot=result["top"]["type"],
                scores=result["scores"],
                ranked=result["ranked"]
            )
            db.add(rec)
            
        db.commit()
        print(f"✅ Recommendation generated for user {user.pid}: {result['top']['type']}")
        
        return {
            "ok": True,
            "scores": result["scores"],
            "ranked": result["ranked"], 
            "top": result["top"],
            "next_step": "choose_bot"
        }
        
    except Exception as e:
        print(f"❌ Recommendation error: {e}")
        db.rollback()
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/match/choose")
async def choose_bot(
    body: dict,
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    try:
        bot_type = body.get("bot_type")
        valid_bots = {"empathy", "insight", "solution", "cognitive"}
        
        if bot_type not in valid_bots:
            raise HTTPException(status_code=422, detail=f"無效的機器人類型，必須是 {valid_bots} 之一")

        # 更新用戶選擇
        user.selected_bot = bot_type
        db.add(user)
        db.commit()
        
        print(f"✅ User {user.pid} selected bot: {bot_type}")
        
        return {
            "ok": True,
            "selected_bot": user.selected_bot,
            "message": f"已選擇 {bot_type} 機器人",
            "next_route": "/dashboard"
        }
        
    except Exception as e:
        print(f"❌ Bot selection error: {e}")
        db.rollback()
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/match/me")
async def my_match(user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    try:
        rec = db.query(Recommendation).filter(Recommendation.user_id == user.id).order_by(Recommendation.id.desc()).first()
        
        return {
            "selected_bot": user.selected_bot,
            "latest_recommendation": {
                "scores": rec.scores if rec else None,
                "ranked": rec.ranked if rec else None,
                "created_at": rec.created_at.isoformat() + "Z" if rec else None
            } if rec else None
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# ============================================================================
# Chat & Mood (Simplified Legacy Support)
# ============================================================================

@app.post("/api/chat/messages")
async def save_chat_message(body: dict, user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    try:
        # 確保用戶已選擇機器人
        if not user.selected_bot:
            raise HTTPException(status_code=400, detail="請先選擇機器人")
            
        msg = ChatMessage(
            user_id=user.id,
            role=body.get("role", "user"),
            content=body.get("content", ""),
            bot_type=body.get("bot_type") or user.selected_bot,
            mode=body.get("mode", "text")
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
        rows = db.query(ChatMessage).filter(ChatMessage.user_id == user.id).order_by(ChatMessage.id.desc()).limit(limit).all()
        return {"messages": [{"id": r.id, "role": r.role, "content": r.content, "bot_type": r.bot_type} for r in rows]}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# ============================================================================
# Mood Records
# ============================================================================

@app.post("/api/mood/records")
async def create_mood(body: dict, user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    try:
        rec = MoodRecord(
            user_id=user.id,
            mood=body.get("mood"),
            intensity=body.get("intensity"), 
            note=body.get("note")
        )
        db.add(rec)
        db.commit()
        return {"ok": True, "id": rec.id}
    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)