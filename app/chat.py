# app/chat.py - 完整修正版 (記錄 PID + 台灣時區)
import os
import asyncio
import logging
from typing import List, Dict, Any, Optional
from datetime import datetime, timezone, timedelta

import aiohttp
from fastapi import APIRouter, Depends, BackgroundTasks, HTTPException
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session
from sqlalchemy import func

from app.db.session import get_db
from app.models.chat import ChatMessage
from app.models.user import User
from app.core.security import get_current_user

logger = logging.getLogger(__name__)
router = APIRouter()

# 台灣時區 UTC+8
TW_TZ = timezone(timedelta(hours=8))

def get_tw_time():
    """取得台灣當前時間"""
    return datetime.now(TW_TZ)

# ================= Pydantic Models =================

class SendPayload(BaseModel):
    message: str
    bot_type: Optional[str] = Field(default="solution")
    mode: Optional[str] = Field(default="text") 
    history: Optional[List[Dict[str, str]]] = Field(default_factory=list)
    demo: Optional[bool] = Field(default=False)
    session_id: Optional[str] = Field(default=None)

class SendResult(BaseModel):
    ok: bool
    reply: str
    bot: Dict[str, Any]
    error: Optional[str] = None
    message_id: Optional[int] = None
    session_id: Optional[str] = None

class HeyGenVoiceConfig(BaseModel):
    voice_id: str = Field(default="zh-TW-HsiaoChenNeural")
    rate: float = Field(default=1.0)
    emotion: str = Field(default="friendly")

class HeyGenSessionRequest(BaseModel):
    avatar_id: Optional[str] = Field(default=None)
    voice: Optional[HeyGenVoiceConfig] = Field(default=None)
    quality: str = Field(default="medium")
    language: str = Field(default="zh-TW")

class HeyGenSessionResponse(BaseModel):
    success: bool
    session_id: Optional[str] = None
    access_token: Optional[str] = None
    url: Optional[str] = None
    stream_url: Optional[str] = None
    error: Optional[str] = None
    data: Optional[Dict] = None

class HeyGenTextRequest(BaseModel):
    session_id: str
    text: str
    emotion: str = Field(default="friendly")
    rate: float = Field(default=1.0)

# ================= Persona System =================

ENHANCED_PERSONA_STYLES = {
    "empathy": {
        "name": "Lumi",
        "system": """你是 Lumi,一位溫暖的同理型 AI 夥伴。以溫柔、非評判、短句回應,優先表達共感與理解。用繁體中文回覆。""",
    },
    "insight": {
        "name": "Solin", 
        "system": """你是 Solin,一位善於引導的洞察型 AI 夥伴。以蘇格拉底式對話引導用戶自我發現,維持中性、尊重、結構化的態度。用繁體中文回覆。""",
    },
    "solution": {
        "name": "Niko",
        "system": """你是 Niko,一位務實的解決型 AI 夥伴。聚焦於可行的步驟與微目標,語氣鼓勵但不強迫。用繁體中文回覆。""",
    },
    "cognitive": {
        "name": "Clara",
        "system": """你是 Clara,一位理性的認知型 AI 夥伴。幫助辨識自動想法與認知扭曲,提供結構化的思維練習。用繁體中文回覆。""",
    }
}

def get_enhanced_system_prompt(bot_type: str) -> str:
    persona = ENHANCED_PERSONA_STYLES.get(bot_type)
    if not persona:
        return ENHANCED_PERSONA_STYLES["solution"]["system"]
    return persona["system"]

def get_bot_name(bot_type: str) -> str:
    persona = ENHANCED_PERSONA_STYLES.get(bot_type)
    if not persona:
        return "Niko"
    return persona["name"]

def get_fallback_reply(bot_type: str) -> str:
    fallbacks = {
        "empathy": "我在這裡聽你說。想和我分享一下現在的感受嗎?",
        "insight": "讓我們慢慢來。能告訴我更多關於這個情況的背景嗎?", 
        "solution": "我們一起想想辦法。能具體說說目前遇到的挑戰嗎?",
        "cognitive": "讓我們理性分析一下。這個想法是什麼時候開始的呢?"
    }
    return fallbacks.get(bot_type, fallbacks["solution"])

# ================= OpenAI Integration =================

def call_openai(system_prompt: str, messages: List[Dict[str, str]]) -> str:
    api_key = os.getenv("OPENAI_API_KEY", "")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY not set")

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
        logger.error(f"OpenAI API failed: {e}")
        raise

# ================= 核心聊天端點 =================

@router.post("/send", response_model=SendResult)
async def send_chat(
    payload: SendPayload, 
    background_tasks: BackgroundTasks,
    user: User = Depends(get_current_user),  # ✅ JWT 認證
    db: Session = Depends(get_db)
):
    """發送聊天訊息 - 記錄 PID 和台灣時間"""
    user_msg = (payload.message or "").strip()
    if not user_msg:
        raise HTTPException(status_code=400, detail="Empty message")

    # ✅ 詳細日誌
    tw_time = get_tw_time()
    print(f"📨 [TW {tw_time.strftime('%H:%M:%S')}] Chat from PID={user.pid}, user_id={user.id}, bot={payload.bot_type}")
    logger.info(f"Chat from PID={user.pid}, user_id={user.id}")

    try:
        # ✅ 1. 儲存使用者訊息 - 記錄 PID 和台灣時間
        user_message = ChatMessage(
            user_id=user.id,
            pid=user.pid,  # ✅ 記錄 PID
            bot_type=payload.bot_type,
            mode=payload.mode,
            role="user",
            content=user_msg,
            created_at=tw_time,  # ✅ 台灣時間
            meta={"demo": payload.demo, "session_id": payload.session_id}
        )
        db.add(user_message)
        db.commit()
        
        print(f"✅ User msg saved: id={user_message.id}, PID={user.pid}, time={tw_time.strftime('%H:%M:%S')}")
        
        # 2. 準備 OpenAI 請求
        system_prompt = get_enhanced_system_prompt(payload.bot_type)
        bot_name = get_bot_name(payload.bot_type)
        
        messages = []
        for h in (payload.history or [])[-10:]:
            role = "assistant" if h.get("role") == "assistant" else "user"
            messages.append({"role": role, "content": h.get("content", "")})
        
        messages.append({"role": "user", "content": user_msg})
        
        # 3. 呼叫 OpenAI
        reply_text = call_openai(system_prompt, messages)
        
        # ✅ 4. 儲存 AI 回覆 - 同樣記錄 PID 和台灣時間
        ai_tw_time = get_tw_time()
        ai_message = ChatMessage(
            user_id=user.id,
            pid=user.pid,  # ✅ 記錄 PID
            bot_type=payload.bot_type,
            mode=payload.mode,
            role="ai",
            content=reply_text,
            created_at=ai_tw_time,  # ✅ 台灣時間
            meta={
                "provider": "openai", 
                "model": os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
                "persona": payload.bot_type,
                "bot_name": bot_name,
                "session_id": payload.session_id
            }
        )
        db.add(ai_message)
        db.commit()
        
        print(f"✅ AI msg saved: id={ai_message.id}, PID={user.pid}, time={ai_tw_time.strftime('%H:%M:%S')}")
        logger.info(f"Chat success: PID={user.pid}, user_msg={user_message.id}, ai_msg={ai_message.id}")
        
        # 5. HeyGen 背景任務
        if payload.session_id and payload.mode == "video":
            background_tasks.add_task(
                send_text_to_heygen_background, 
                payload.session_id, 
                reply_text
            )
        
        return SendResult(
            ok=True,
            reply=reply_text,
            bot={
                "type": payload.bot_type, 
                "name": bot_name,
                "persona": "enhanced"
            },
            message_id=ai_message.id,
            session_id=payload.session_id,
            error=None
        )
        
    except Exception as e:
        error_time = get_tw_time()
        print(f"❌ [TW {error_time.strftime('%H:%M:%S')}] Chat error: PID={user.pid}, error={str(e)[:100]}")
        logger.error(f"Chat failed: PID={user.pid}, error={e}")
        db.rollback()
        
        fallback_text = get_fallback_reply(payload.bot_type)
        bot_name = get_bot_name(payload.bot_type)
        
        try:
            ai_message = ChatMessage(
                user_id=user.id,
                pid=user.pid,  # ✅ 記錄 PID
                bot_type=payload.bot_type,
                mode=payload.mode,
                role="ai",
                content=fallback_text,
                created_at=get_tw_time(),
                meta={
                    "provider": "fallback", 
                    "error": str(e)[:200],
                    "persona": payload.bot_type,
                    "bot_name": bot_name
                }
            )
            db.add(ai_message)
            db.commit()
        except Exception as db_error:
            logger.error(f"Failed to save fallback: {db_error}")
        
        return SendResult(
            ok=True,
            reply=fallback_text,
            bot={
                "type": payload.bot_type, 
                "name": bot_name,
                "persona": "fallback"
            },
            error=f"API temporarily unavailable: {str(e)[:100]}"
        )

# ================= 聊天歷史與統計 =================

@router.get("/history")
async def get_chat_history(
    limit: int = 50,
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """獲取當前用戶的聊天歷史"""
    try:
        messages = (
            db.query(ChatMessage)
            .filter(ChatMessage.user_id == user.id)
            .order_by(ChatMessage.created_at.desc())
            .limit(limit)
            .all()
        )
        
        result = []
        for msg in reversed(messages):
            result.append({
                "id": msg.id,
                "pid": msg.pid,  # ✅ 回傳 PID
                "role": msg.role,
                "content": msg.content,
                "bot_type": msg.bot_type,
                "mode": msg.mode,
                "created_at": msg.created_at.isoformat(),
                "meta": msg.meta or {}
            })
        
        return {
            "ok": True,
            "messages": result,
            "count": len(result)
        }
        
    except Exception as e:
        logger.error(f"Failed to fetch history: {e}")
        raise HTTPException(status_code=500, detail="Failed to fetch history")

@router.get("/stats")
async def get_chat_stats(
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """獲取用戶的聊天統計資訊"""
    try:
        total_messages = (
            db.query(ChatMessage)
            .filter(ChatMessage.user_id == user.id)
            .count()
        )
        
        messages_by_bot = (
            db.query(
                ChatMessage.bot_type,
                func.count(ChatMessage.id).label('count')
            )
            .filter(ChatMessage.user_id == user.id)
            .group_by(ChatMessage.bot_type)
            .all()
        )
        
        first_message = (
            db.query(ChatMessage)
            .filter(ChatMessage.user_id == user.id)
            .order_by(ChatMessage.created_at.asc())
            .first()
        )
        
        last_message = (
            db.query(ChatMessage)
            .filter(ChatMessage.user_id == user.id)
            .order_by(ChatMessage.created_at.desc())
            .first()
        )
        
        return {
            "ok": True,
            "pid": user.pid,  # ✅ 回傳 PID
            "stats": {
                "total_messages": total_messages,
                "messages_by_bot": {bot: count for bot, count in messages_by_bot},
                "first_message_at": first_message.created_at.isoformat() if first_message else None,
                "last_message_at": last_message.created_at.isoformat() if last_message else None
            }
        }
        
    except Exception as e:
        logger.error(f"Failed to fetch stats: {e}")
        raise HTTPException(status_code=500, detail="Failed to fetch stats")

# ================= HeyGen (保留原有功能) =================

async def send_text_to_heygen_background(session_id: str, text: str):
    """背景任務:發送文字到 HeyGen"""
    try:
        heygen_api_key = os.getenv("HEYGEN_API_KEY")
        if not heygen_api_key:
            return
            
        repeat_data = {
            "session_id": session_id,
            "text": text,
            "voice": {"emotion": "friendly", "rate": 1.0}
        }
        
        headers = {
            "X-API-KEY": heygen_api_key,
            "Content-Type": "application/json"
        }
        
        async with aiohttp.ClientSession() as session:
            async with session.post(
                "https://api.heygen.com/v2/streaming/repeat",
                headers=headers,
                json=repeat_data
            ) as response:
                if response.status != 200:
                    logger.error(f"HeyGen background task failed: {response.status}")
                    
    except Exception as e:
        logger.error(f"Background HeyGen task failed: {e}")