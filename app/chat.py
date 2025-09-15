# app/chat.py - 修正的 HeyGen 實作，遵循官方 API 規範

import os
import asyncio
import logging
from typing import List, Dict, Any, Optional
from datetime import datetime

import aiohttp
from fastapi import APIRouter, Depends, BackgroundTasks, Request, HTTPException
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from app.db.session import get_db
from app.models.chat import ChatMessage

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

# HeyGen 正確的模型
class HeyGenTokenRequest(BaseModel):
    pass  # 創建令牌不需要參數

class HeyGenSessionRequest(BaseModel):
    avatar_id: Optional[str] = Field(default=None)
    voice: Optional[Dict[str, Any]] = Field(default=None)
    quality: str = Field(default="medium")
    version: str = Field(default="v2")

class HeyGenSessionResponse(BaseModel):
    success: bool
    session_id: Optional[str] = None
    access_token: Optional[str] = None
    url: Optional[str] = None
    error: Optional[str] = None
    data: Optional[Dict] = None

class HeyGenTextRequest(BaseModel):
    session_id: str
    text: str
    task_type: str = Field(default="repeat")
    task_mode: str = Field(default="sync")

# ================= Persona System (保持原有) =================

ENHANCED_PERSONA_STYLES = {
    "empathy": {
        "name": "Lumi",
        "system": """你是 Lumi，一位溫暖的同理型 AI 夥伴。以溫柔、非評判、短句回應，優先表達共感與理解。用繁體中文回覆。""",
    },
    "insight": {
        "name": "Solin", 
        "system": """你是 Solin，一位善於引導的洞察型 AI 夥伴。以蘇格拉底式對話引導用戶自我發現，維持中性、尊重的態度。用繁體中文回覆。""",
    },
    "solution": {
        "name": "Niko",
        "system": """你是 Niko，一位務實的解決型 AI 夥伴。聚焦於可行的步驟與微目標，語氣鼓勵但不強迫。用繁體中文回覆。""",
    },
    "cognitive": {
        "name": "Clara",
        "system": """你是 Clara，一位理性的認知型 AI 夥伴。協助辨識自動想法與認知偏誤，提供結構化的思維練習。用繁體中文回覆。""",
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
        "empathy": "我在這裡聽你說。想和我分享一下現在的感受嗎？",
        "insight": "讓我們慢慢來。能告訴我更多關於這個情況的背景嗎？", 
        "solution": "我們一起想想辦法。能具體說說目前遇到的挑戰嗎？",
        "cognitive": "讓我們理性分析一下。這個想法是什麼時候開始的呢？"
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
        print(f"OpenAI API failed: {e}")
        return "我在這裡陪著你。想聊聊今天最讓你在意的事情嗎？"

# ================= 主要聊天端點 =================

@router.post("/send", response_model=SendResult)
async def send_chat(
    payload: SendPayload, 
    request: Request, 
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db)
):
    user_msg = (payload.message or "").strip()
    if not user_msg:
        raise HTTPException(status_code=400, detail="Empty message")

    user_id_hdr = request.headers.get("X-User-Id")
    try:
        user_id = int(user_id_hdr) if user_id_hdr is not None else 0
    except ValueError:
        user_id = 0

    try:
        # 1. 儲存使用者訊息
        user_message = ChatMessage(
            user_id=user_id,
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
            user_id=user_id,
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
        
        # 5. 如果有 HeyGen session_id，加入背景任務發送文字
        if payload.session_id and payload.mode == "video":
            background_tasks.add_task(send_text_to_heygen_background, payload.session_id, reply_text)
        
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
        logger.error(f"Send chat failed: {e}")
        db.rollback()
        
        fallback_text = get_fallback_reply(payload.bot_type)
        bot_name = get_bot_name(payload.bot_type)
        
        try:
            ai_message = ChatMessage(
                user_id=user_id,
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
        except Exception:
            pass
        
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

# ================= 正確的 HeyGen API 實作 =================

@router.post("/heygen/create_token")
async def create_heygen_token():
    """創建 HeyGen 訪問令牌 - 步驟1"""
    heygen_api_key = os.getenv("HEYGEN_API_KEY")
    if not heygen_api_key:
        return {"success": False, "error": "HeyGen API key not configured"}
    
    headers = {
        "X-Api-Key": heygen_api_key,
        "Content-Type": "application/json"
    }
    
    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(
                "https://api.heygen.com/v1/streaming.create_token",
                headers=headers
            ) as response:
                if response.status == 200:
                    result = await response.json()
                    return {
                        "success": True,
                        "token": result.get("data", {}).get("token"),
                        "data": result.get("data")
                    }
                else:
                    error_text = await response.text()
                    return {
                        "success": False,
                        "error": f"HTTP {response.status}: {error_text}"
                    }
                    
    except Exception as e:
        logger.error(f"Failed to create HeyGen token: {e}")
        return {"success": False, "error": str(e)}

@router.post("/heygen/create_session", response_model=HeyGenSessionResponse)
async def create_heygen_session(request: HeyGenSessionRequest):
    """創建 HeyGen 會話 - 步驟2，使用正確的 API 端點"""
    heygen_api_key = os.getenv("HEYGEN_API_KEY")
    if not heygen_api_key:
        return HeyGenSessionResponse(
            success=False, 
            error="HeyGen API key not configured"
        )
    
    avatar_id = request.avatar_id or os.getenv("HEYGEN_AVATAR_ID", "default_avatar")
    
    # 正確的請求格式
    session_data = {
        "version": request.version,
        "avatar_id": avatar_id,
        "quality": request.quality
    }
    
    # 如果有語音設置，加入語音配置
    if request.voice:
        session_data["voice"] = request.voice
    
    headers = {
        "X-Api-Key": heygen_api_key,  # 使用 X-Api-Key 而不是 Authorization Bearer
        "Content-Type": "application/json"
    }
    
    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(
                "https://api.heygen.com/v1/streaming.new",  # 正確的端點
                headers=headers,
                json=session_data
            ) as response:
                if response.status == 200:
                    result = await response.json()
                    if result.get("code") == 100:  # HeyGen 成功代碼
                        data = result.get("data", {})
                        return HeyGenSessionResponse(
                            success=True,
                            session_id=data.get("session_id"),
                            access_token=data.get("access_token"),
                            url=data.get("url"),
                            data=data
                        )
                    else:
                        return HeyGenSessionResponse(
                            success=False,
                            error=result.get("message", "Session creation failed")
                        )
                else:
                    error_text = await response.text()
                    return HeyGenSessionResponse(
                        success=False,
                        error=f"HTTP {response.status}: {error_text}"
                    )
                    
    except Exception as e:
        logger.error(f"HeyGen session creation failed: {e}")
        return HeyGenSessionResponse(
            success=False,
            error=str(e)
        )

@router.post("/heygen/start_session")
async def start_heygen_session(request: dict):
    """啟動 HeyGen 會話 - 步驟3"""
    session_id = request.get("session_id")
    if not session_id:
        return {"success": False, "error": "Session ID required"}
        
    heygen_api_key = os.getenv("HEYGEN_API_KEY")
    if not heygen_api_key:
        return {"success": False, "error": "HeyGen API key not configured"}
    
    headers = {
        "X-Api-Key": heygen_api_key,
        "Content-Type": "application/json"
    }
    
    start_data = {
        "session_id": session_id
    }
    
    # 如果有 SDP，也加入
    if "sdp" in request:
        start_data["sdp"] = request["sdp"]
    
    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(
                "https://api.heygen.com/v1/streaming.start",
                headers=headers,
                json=start_data
            ) as response:
                if response.status == 200:
                    result = await response.json()
                    return {
                        "success": True,
                        "data": result.get("data")
                    }
                else:
                    error_text = await response.text()
                    return {
                        "success": False,
                        "error": f"HTTP {response.status}: {error_text}"
                    }
                    
    except Exception as e:
        logger.error(f"Failed to start HeyGen session: {e}")
        return {"success": False, "error": str(e)}

@router.post("/heygen/send_text")
async def send_text_to_heygen(request: HeyGenTextRequest):
    """發送文字到 HeyGen Avatar - 使用正確的格式"""
    heygen_api_key = os.getenv("HEYGEN_API_KEY")
    if not heygen_api_key:
        return {"success": False, "error": "HeyGen API key not configured"}
    
    # 使用 WebSocket 或 task 端點
    task_data = {
        "session_id": request.session_id,
        "text": request.text,
        "task_type": request.task_type,
        "task_mode": request.task_mode
    }
    
    headers = {
        "X-Api-Key": heygen_api_key,
        "Content-Type": "application/json"
    }
    
    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(
                "https://api.heygen.com/v1/streaming.task",  # 正確的任務端點
                headers=headers,
                json=task_data
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
                    return {
                        "success": False,
                        "error": f"HTTP {response.status}: {error_text}"
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
        "X-Api-Key": heygen_api_key,
        "Content-Type": "application/json"
    }
    
    close_data = {"session_id": session_id}
    
    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(
                "https://api.heygen.com/v1/streaming.stop",  # 正確的停止端點
                headers=headers,
                json=close_data
            ) as response:
                return {"success": True, "message": "Session closed"}
                
    except Exception as e:
        logger.error(f"Failed to close HeyGen session: {e}")
        return {"success": False, "error": str(e)}

# ================= 背景任務 =================

async def send_text_to_heygen_background(session_id: str, text: str):
    """背景任務：發送文字到 HeyGen"""
    try:
        heygen_api_key = os.getenv("HEYGEN_API_KEY")
        if not heygen_api_key:
            return
            
        task_data = {
            "session_id": session_id,
            "text": text,
            "task_type": "repeat",
            "task_mode": "sync"
        }
        
        headers = {
            "X-Api-Key": heygen_api_key,
            "Content-Type": "application/json"
        }
        
        async with aiohttp.ClientSession() as session:
            async with session.post(
                "https://api.heygen.com/v1/streaming.task",
                headers=headers,
                json=task_data
            ) as response:
                if response.status != 200:
                    logger.error(f"HeyGen background task failed: {response.status}")
                    
    except Exception as e:
        logger.error(f"Background HeyGen task failed: {e}")

# ================= 健康檢查 =================

@router.get("/health/heygen")
async def health_heygen():
    """HeyGen 健康檢查 - 使用正確的 API"""
    heygen_api_key = os.getenv("HEYGEN_API_KEY")
    avatar_id = os.getenv("HEYGEN_AVATAR_ID")
    
    info = {
        "has_api_key": bool(heygen_api_key),
        "has_avatar_id": bool(avatar_id),
        "avatar_id": avatar_id if avatar_id else "not_configured",
        "ok": False,
        "error": None,
        "correct_endpoints": {
            "create_token": "/v1/streaming.create_token",
            "create_session": "/v1/streaming.new",
            "start_session": "/v1/streaming.start",
            "send_task": "/v1/streaming.task",
            "close_session": "/v1/streaming.stop"
        }
    }
    
    if not heygen_api_key:
        info["error"] = "HEYGEN_API_KEY not set"
        return info
    
    try:
        headers = {
            "X-Api-Key": heygen_api_key,
            "Content-Type": "application/json"
        }
        
        # 測試創建令牌
        async with aiohttp.ClientSession() as session:
            async with session.post(
                "https://api.heygen.com/v1/streaming.create_token",
                headers=headers,
                timeout=aiohttp.ClientTimeout(total=10)
            ) as response:
                if response.status == 200:
                    info["ok"] = True
                    info["token_creation"] = "success"
                else:
                    error_text = await response.text()
                    info["error"] = f"Token creation failed - HTTP {response.status}: {error_text[:100]}"
                    
    except asyncio.TimeoutError:
        info["error"] = "Connection timeout"
    except Exception as e:
        info["error"] = f"{type(e).__name__}: {str(e)[:150]}"
    
    return info

# ================= 其他端點 =================

@router.get("/health/openai")
async def health_openai():
    """OpenAI 健康檢查"""
    model = os.getenv("OPENAI_MODEL", "gpt-4o")
    key = os.getenv("OPENAI_API_KEY")
    
    info = {
        "model": model,
        "has_key": bool(key),
        "ok": False,
        "error": None,
        "persona_system": "enhanced",
        "heygen_integration": bool(os.getenv("HEYGEN_API_KEY"))
    }
    
    if not key:
        info["error"] = "OPENAI_API_KEY not set"
        return info
    
    try:
        from openai import OpenAI
        client = OpenAI(api_key=key)
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": "ping"}],
            max_tokens=5
        )
        info["ok"] = bool(response.choices)
        return info
        
    except Exception as e:
        info["error"] = f"{type(e).__name__}: {str(e)[:150]}"
        return info

@router.get("/routes")
async def list_routes():
    """列出所有可用的路由"""
    routes = []
    for route in router.routes:
        if hasattr(route, 'methods') and hasattr(route, 'path'):
            routes.append({
                "path": route.path,
                "methods": list(route.methods),
                "name": getattr(route, 'name', 'unnamed')
            })
    return {
        "total_routes": len(routes),
        "routes": routes,
        "heygen_routes": [r for r in routes if 'heygen' in r['path']]
    }