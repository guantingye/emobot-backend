# app/chat.py - 完整修正版本
import os
import asyncio
import logging
from typing import List, Dict, Any, Optional
from datetime import datetime

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
    """取得增強的系統提示"""
    persona = ENHANCED_PERSONA_STYLES.get(bot_type)
    if not persona:
        return ENHANCED_PERSONA_STYLES["solution"]["system"]
    return persona["system"]

def get_bot_name(bot_type: str) -> str:
    """取得機器人名稱"""
    persona = ENHANCED_PERSONA_STYLES.get(bot_type)
    if not persona:
        return "Niko"
    return persona["name"]

def get_fallback_reply(bot_type: str) -> str:
    """取得備用回覆"""
    fallbacks = {
        "empathy": "我在這裡聽你說。想和我分享一下現在的感受嗎?",
        "insight": "讓我們慢慢來。能告訴我更多關於這個情況的背景嗎?", 
        "solution": "我們一起想想辦法。能具體說說目前遇到的挑戰嗎?",
        "cognitive": "讓我們理性分析一下。這個想法是什麼時候開始的呢?"
    }
    return fallbacks.get(bot_type, fallbacks["solution"])

# ================= OpenAI Integration =================

def call_openai(system_prompt: str, messages: List[Dict[str, str]]) -> str:
    """呼叫 OpenAI API"""
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
    user: User = Depends(get_current_user),  # 使用 JWT 認證
    db: Session = Depends(get_db)
):
    """發送聊天訊息 - 使用 JWT 認證獲取 user_id"""
    user_msg = (payload.message or "").strip()
    if not user_msg:
        raise HTTPException(status_code=400, detail="Empty message")

    logger.info(f"Chat message from user_id={user.id}, pid={user.pid}")

    try:
        # 1. 儲存使用者訊息
        user_message = ChatMessage(
            user_id=user.id,  # 從 JWT 認證取得
            bot_type=payload.bot_type,
            mode=payload.mode,
            role="user",
            content=user_msg,
            meta={"demo": payload.demo, "session_id": payload.session_id}
        )
        db.add(user_message)
        db.commit()
        
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
        
        # 4. 儲存 AI 回覆
        ai_message = ChatMessage(
            user_id=user.id,
            bot_type=payload.bot_type,
            mode=payload.mode,
            role="ai",
            content=reply_text,
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
        
        logger.info(f"Chat success: user_id={user.id}, message_id={ai_message.id}")
        
        # 5. 如果有 HeyGen session_id,加入背景任務
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
        logger.error(f"Send chat failed: user_id={user.id}, error={e}")
        db.rollback()
        
        fallback_text = get_fallback_reply(payload.bot_type)
        bot_name = get_bot_name(payload.bot_type)
        
        try:
            ai_message = ChatMessage(
                user_id=user.id,
                bot_type=payload.bot_type,
                mode=payload.mode,
                role="ai",
                content=fallback_text,
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
            logger.error(f"Failed to save fallback message: {db_error}")
        
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
                "role": msg.role,
                "content": msg.content,
                "bot_type": msg.bot_type,
                "mode": msg.mode,
                "created_at": msg.created_at.isoformat() + "Z",
                "meta": msg.meta or {}
            })
        
        return {
            "ok": True,
            "messages": result,
            "count": len(result)
        }
        
    except Exception as e:
        logger.error(f"Failed to fetch chat history: {e}")
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
            "stats": {
                "total_messages": total_messages,
                "messages_by_bot": {bot: count for bot, count in messages_by_bot},
                "first_message_at": first_message.created_at.isoformat() + "Z" if first_message else None,
                "last_message_at": last_message.created_at.isoformat() + "Z" if last_message else None
            }
        }
        
    except Exception as e:
        logger.error(f"Failed to fetch chat stats: {e}")
        raise HTTPException(status_code=500, detail="Failed to fetch stats")

# ================= HeyGen API Integration =================

@router.post("/heygen/create_session", response_model=HeyGenSessionResponse)
async def create_heygen_session(request: HeyGenSessionRequest):
    """創建 HeyGen 會話 - 使用正確的 v2 API 格式"""
    heygen_api_key = os.getenv("HEYGEN_API_KEY")
    if not heygen_api_key:
        return HeyGenSessionResponse(
            success=False, 
            error="HeyGen API key not configured"
        )
    
    avatar_id = request.avatar_id or os.getenv("HEYGEN_AVATAR_ID", "June_HR_public")
    
    session_data = {
        "avatar_name": avatar_id,
        "voice": {
            "voice_id": request.voice.voice_id if request.voice else "zh-TW-HsiaoChenNeural",
            "rate": request.voice.rate if request.voice else 1.0,
            "emotion": request.voice.emotion if request.voice else "friendly"
        },
        "quality": request.quality,
        "language": request.language
    }
    
    headers = {
        "X-API-KEY": heygen_api_key,
        "Content-Type": "application/json"
    }
    
    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(
                "https://api.heygen.com/v2/streaming/create_session",
                headers=headers,
                json=session_data,
                timeout=aiohttp.ClientTimeout(total=30)
            ) as response:
                if response.status == 200:
                    result = await response.json()
                    if result.get("code") == 100:
                        data = result.get("data", {})
                        return HeyGenSessionResponse(
                            success=True,
                            session_id=data.get("session_id"),
                            stream_url=data.get("url"),
                            data=data
                        )
                    else:
                        return HeyGenSessionResponse(
                            success=False,
                            error=result.get("message", "Session creation failed")
                        )
                else:
                    error_text = await response.text()
                    logger.error(f"HeyGen API error {response.status}: {error_text}")
                    return HeyGenSessionResponse(
                        success=False,
                        error=f"HTTP {response.status}: {error_text[:200]}"
                    )
                    
    except asyncio.TimeoutError:
        return HeyGenSessionResponse(
            success=False,
            error="Request timeout - HeyGen service may be slow"
        )
    except Exception as e:
        logger.error(f"HeyGen session creation failed: {e}")
        return HeyGenSessionResponse(
            success=False,
            error=str(e)
        )

@router.post("/heygen/send_text")
async def send_text_to_heygen(request: HeyGenTextRequest):
    """發送文字到 HeyGen Avatar"""
    heygen_api_key = os.getenv("HEYGEN_API_KEY")
    if not heygen_api_key:
        return {"success": False, "error": "HeyGen API key not configured"}
    
    repeat_data = {
        "session_id": request.session_id,
        "text": request.text,
        "voice": {
            "emotion": request.emotion,
            "rate": request.rate
        }
    }
    
    headers = {
        "X-API-KEY": heygen_api_key,
        "Content-Type": "application/json"
    }
    
    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(
                "https://api.heygen.com/v2/streaming/repeat",
                headers=headers,
                json=repeat_data,
                timeout=aiohttp.ClientTimeout(total=15)
            ) as response:
                if response.status == 200:
                    result = await response.json()
                    if result.get("code") == 100:
                        return {"success": True, "message": "Text sent successfully"}
                    else:
                        return {
                            "success": False,
                            "error": result.get("message", "Failed to send text")
                        }
                else:
                    error_text = await response.text()
                    logger.error(f"HeyGen repeat error {response.status}: {error_text}")
                    return {
                        "success": False,
                        "error": f"HTTP {response.status}: {error_text[:200]}"
                    }
                    
    except Exception as e:
        logger.error(f"Failed to send text to HeyGen: {e}")
        return {"success": False, "error": str(e)}

@router.post("/heygen/close_session")
async def close_heygen_session(request: dict):
    """關閉 HeyGen 會話"""
    session_id = request.get("session_id")
    if not session_id:
        return {"success": False, "error": "Session ID required"}
        
    heygen_api_key = os.getenv("HEYGEN_API_KEY")
    if not heygen_api_key:
        return {"success": False, "error": "HeyGen API key not configured"}
    
    headers = {
        "X-API-KEY": heygen_api_key,
        "Content-Type": "application/json"
    }
    
    close_data = {"session_id": session_id}
    
    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(
                "https://api.heygen.com/v2/streaming/close_session",
                headers=headers,
                json=close_data
            ) as response:
                return {"success": True, "message": "Session closed"}
                
    except Exception as e:
        logger.error(f"Failed to close HeyGen session: {e}")
        return {"success": False, "error": str(e)}

# ================= 背景任務 =================

async def send_text_to_heygen_background(session_id: str, text: str):
    """背景任務:發送文字到 HeyGen"""
    try:
        heygen_api_key = os.getenv("HEYGEN_API_KEY")
        if not heygen_api_key:
            return
            
        repeat_data = {
            "session_id": session_id,
            "text": text,
            "voice": {
                "emotion": "friendly",
                "rate": 1.0
            }
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

# ================= 健康檢查 =================

@router.get("/health/heygen")
async def health_heygen():
    """HeyGen 健康檢查"""
    heygen_api_key = os.getenv("HEYGEN_API_KEY")
    avatar_id = os.getenv("HEYGEN_AVATAR_ID")
    
    info = {
        "has_api_key": bool(heygen_api_key),
        "has_avatar_id": bool(avatar_id),
        "avatar_id": avatar_id if avatar_id else "not_configured",
        "api_version": "v2",
        "ok": False,
        "error": None
    }
    
    if not heygen_api_key:
        info["error"] = "HEYGEN_API_KEY not set"
        return info
    
    info["ok"] = True
    return info