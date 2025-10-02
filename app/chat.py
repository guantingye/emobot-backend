# backend/app/chat.py - 完整版本
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
from app.core.timezone import get_tw_time, format_tw_time
logger = logging.getLogger(__name__)
router = APIRouter()

TW_TZ = timezone(timedelta(hours=8))

def get_tw_time():
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

# ================= Persona System =================

ENHANCED_PERSONA_STYLES = {
    "empathy": {
        "name": "Lumi",
        "system": "你是 Lumi,一位溫暖的同理型 AI 夥伴。以溫柔、非評判、短句回應,優先表達共感與理解。用繁體中文回覆。"
    },
    "insight": {
        "name": "Solin",
        "system": "你是 Solin,一位善於引導的洞察型 AI 夥伴。以蘇格拉底式對話引導用戶自我發現,維持中性、尊重、結構化的態度。用繁體中文回覆。"
    },
    "solution": {
        "name": "Niko",
        "system": "你是 Niko,一位務實的解決型 AI 夥伴。聚焦於可行的步驟與微目標,語氣鼓勵但不強迫。用繁體中文回覆。"
    },
    "cognitive": {
        "name": "Clara",
        "system": "你是 Clara,一位理性的認知型 AI 夥伴。幫助辨識自動想法與認知扭曲,提供結構化的思維練習。用繁體中文回覆。"
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
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """發送聊天訊息 - 使用台灣時間"""
    user_msg = (payload.message or "").strip()
    if not user_msg:
        raise HTTPException(status_code=400, detail="Empty message")

    tw_time = get_tw_time()
    print(f"📨 [TW {tw_time.strftime('%Y-%m-%d %H:%M:%S')}] Chat from PID={user.pid}")

    try:
        # 儲存使用者訊息
        user_message = ChatMessage(
            pid=user.pid,
            bot_type=payload.bot_type,
            mode=payload.mode,
            role="user",
            content=user_msg,
            created_at=tw_time,  # ✅ 台灣時間
            meta={"demo": payload.demo, "session_id": payload.session_id}
        )
        db.add(user_message)
        db.commit()
        db.refresh(user_message)
        
        # 呼叫 OpenAI
        system_prompt = get_enhanced_system_prompt(payload.bot_type)
        bot_name = get_bot_name(payload.bot_type)
        
        messages = []
        for h in (payload.history or [])[-10:]:
            role = "assistant" if h.get("role") == "assistant" else "user"
            messages.append({"role": role, "content": h.get("content", "")})
        messages.append({"role": "user", "content": user_msg})
        
        reply_text = call_openai(system_prompt, messages)
        
        # 儲存 AI 回覆
        ai_message = ChatMessage(
            pid=user.pid,
            bot_type=payload.bot_type,
            mode=payload.mode,
            role="ai",
            content=reply_text,
            created_at=get_tw_time(),  # ✅ 台灣時間
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
        db.refresh(ai_message)
        
        print(f"✅ Chat saved: PID={user.pid}, TW={format_tw_time(ai_message.created_at)}")
        
        return SendResult(
            ok=True,
            reply=reply_text,
            bot={"type": payload.bot_type, "name": bot_name, "persona": "enhanced"},
            message_id=ai_message.id,
            session_id=payload.session_id,
            error=None
        )
        
    except Exception as e:
        db.rollback()
        print(f"❌ Chat error: {e}")
        raise

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
            .filter(ChatMessage.pid == user.pid)
            .order_by(ChatMessage.created_at.desc())
            .limit(limit)
            .all()
        )
        
        result = []
        for msg in reversed(messages):
            result.append({
                "id": msg.id,
                "pid": msg.pid,
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
            .filter(ChatMessage.pid == user.pid)
            .count()
        )
        
        messages_by_bot = (
            db.query(
                ChatMessage.bot_type,
                func.count(ChatMessage.id).label('count')
            )
            .filter(ChatMessage.pid == user.pid)
            .group_by(ChatMessage.bot_type)
            .all()
        )
        
        first_message = (
            db.query(ChatMessage)
            .filter(ChatMessage.pid == user.pid)
            .order_by(ChatMessage.created_at.asc())
            .first()
        )
        
        last_message = (
            db.query(ChatMessage)
            .filter(ChatMessage.pid == user.pid)
            .order_by(ChatMessage.created_at.desc())
            .first()
        )
        
        return {
            "ok": True,
            "pid": user.pid,
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