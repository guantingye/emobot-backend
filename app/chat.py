# app/chat.py - 完整修復版本，確保 HeyGen 路由正確註冊

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

# 配置日誌
logger = logging.getLogger(__name__)

# *** 創建 router - 確保正確配置 ***
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

# *** HeyGen 相關模型 ***
class HeyGenSessionRequest(BaseModel):
    avatar_id: Optional[str] = Field(default=None)
    voice: str = Field(default="zh-TW-HsiaoChenNeural")
    quality: str = Field(default="medium")

class HeyGenSessionResponse(BaseModel):
    success: bool
    session_id: Optional[str] = None
    error: Optional[str] = None
    data: Optional[Dict] = None

class HeyGenTextRequest(BaseModel):
    session_id: str
    text: str
    emotion: str = Field(default="friendly")

# ================= Enhanced Persona System =================

ENHANCED_PERSONA_STYLES = {
    "empathy": {
        "name": "Lumi",
        "system": """你是 Lumi，一位溫暖的同理型 AI 夥伴。

**核心定位**：情緒支持者與傾聽者
**對話風格**：溫柔、非評判、短句回應
**主要技巧**：
- 反映式傾聽："聽起來你..."
- 情緒標記："感覺很..."
- 肯認回應："我聽到了..."
- 陪伴語句："我在這裡陪你"

回應時優先表達共感與理解，避免立即給建議。用繁體中文回覆，保持溫暖支持的語調。""",
    },
    "insight": {
        "name": "Solin", 
        "system": """你是 Solin，一位善於引導的洞察型 AI 夥伴。

**核心定位**：思維澄清者與洞察引導者
**對話風格**：溫柔提問、結構化探索
**主要技巧**：
- 開放式提問："什麼讓你覺得...？"
- 澄清確認："如果我理解正確..."
- 重新框架："換個角度來看..."
- 模式識別："我注意到..."

以蘇格拉底式對話引導用戶自我發現，維持中性、尊重的態度。用繁體中文回覆。""",
    },
    "solution": {
        "name": "Niko",
        "system": """你是 Niko，一位務實的解決型 AI 夥伴。

**核心定位**：行動促進者與方案提供者  
**對話風格**：務實、具體、鼓勵行動
**主要技巧**：
- 小步驟分解："我們可以先..."
- 具體建議："你可以試試..."
- 資源盤點："你有哪些..."
- 下一步引導："接下來..."

聚焦於可行的步驟與微目標，語氣鼓勵但不強迫。用繁體中文回覆。""",
    },
    "cognitive": {
        "name": "Clara",
        "system": """你是 Clara，一位理性的認知型 AI 夥伴。

**核心定位**：思維模式分析者與重建協助者
**對話風格**：CBT 導向、邏輯清晰
**主要技巧**：
- 想法檢核："這個想法有什麼證據？"
- 認知偏誤識別："這聽起來像是..."
- 替代想法："還有什麼可能性？"
- 行為實驗："我們可以測試..."

協助辨識自動想法與認知偏誤，提供結構化的思維練習。用繁體中文回覆。""",
    }
}

def get_enhanced_system_prompt(bot_type: str) -> str:
    """取得增強版系統提示"""
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
    """根據機器人類型提供後備回覆"""
    fallbacks = {
        "empathy": "我在這裡聽你說。想和我分享一下現在的感受嗎？",
        "insight": "讓我們慢慢來。能告訴我更多關於這個情況的背景嗎？", 
        "solution": "我們一起想想辦法。能具體說說目前遇到的挑戰嗎？",
        "cognitive": "讓我們理性分析一下。這個想法是什麼時候開始的呢？"
    }
    return fallbacks.get(bot_type, fallbacks["solution"])

# ================= OpenAI Integration =================

def call_openai(system_prompt: str, messages: List[Dict[str, str]]) -> str:
    """簡化的 OpenAI 呼叫"""
    api_key = os.getenv("OPENAI_API_KEY", "")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY not set")

    try:
        # 優先使用新版 SDK
        from openai import OpenAI
        client = OpenAI(api_key=api_key)
        
        # 準備訊息
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
        # 返回預設回覆而不是拋出異常
        return "我在這裡陪著你。想聊聊今天最讓你在意的事情嗎？"

# ================= 主要聊天端點 =================

@router.post("/send", response_model=SendResult)
async def send_chat(
    payload: SendPayload, 
    request: Request, 
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db)
):
    """聊天端點 - 支援增強版 Persona System Prompts + OpenAI 回覆 + HeyGen 整合"""
    
    user_msg = (payload.message or "").strip()
    if not user_msg:
        raise HTTPException(status_code=400, detail="Empty message")

    # 取得 user_id（從標頭或預設為 0）
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
        
        # 轉換歷史記錄格式，只取最近 10 條
        messages = []
        for h in (payload.history or [])[-10:]:
            role = "assistant" if h.get("role") == "assistant" else "user"
            messages.append({"role": role, "content": h.get("content", "")})
        
        # 添加當前使用者訊息
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
        
        # 6. 返回結果
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
        
        # 緊急後備回覆
        fallback_text = get_fallback_reply(payload.bot_type)
        bot_name = get_bot_name(payload.bot_type)
        
        # 儲存緊急回覆
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
            pass  # 如果連 fallback 都失敗，就不儲存了
        
        return SendResult(
            ok=True,  # 仍然回傳 ok=True，確保前端正常顯示
            reply=fallback_text,
            bot={
                "type": payload.bot_type, 
                "name": bot_name,
                "persona": "fallback"
            },
            error=f"API temporarily unavailable: {str(e)[:100]}"
        )

# ================= HeyGen 相關路由 =================

@router.post("/heygen/create_session", response_model=HeyGenSessionResponse)
async def create_heygen_session(request: HeyGenSessionRequest):
    """創建HeyGen會話 - 通過後端代理避免CORS"""
    heygen_api_key = os.getenv("HEYGEN_API_KEY")
    if not heygen_api_key:
        return HeyGenSessionResponse(
            success=False, 
            error="HeyGen API key not configured"
        )
    
    avatar_id = request.avatar_id or os.getenv("HEYGEN_AVATAR_ID", "default_avatar")
    
    session_data = {
        "avatar_name": avatar_id,
        "voice": {
            "voice_id": request.voice,
            "rate": 1.0,
            "emotion": "friendly"
        },
        "quality": request.quality,
        "language": "zh-TW"
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
                json=session_data
            ) as response:
                if response.status == 200:
                    result = await response.json()
                    if result.get("code") == 100:
                        return HeyGenSessionResponse(
                            success=True,
                            session_id=result["data"]["session_id"],
                            data=result["data"]
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

@router.post("/heygen/send_text")
async def send_text_to_heygen(request: HeyGenTextRequest):
    """發送文字到HeyGen Avatar"""
    heygen_api_key = os.getenv("HEYGEN_API_KEY")
    if not heygen_api_key:
        return {"success": False, "error": "HeyGen API key not configured"}
    
    repeat_data = {
        "session_id": request.session_id,
        "text": request.text,
        "voice": {
            "emotion": request.emotion,
            "rate": 1.0
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
                json=repeat_data
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
    """關閉HeyGen會話"""
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

# ================= 背景任務函數 =================

async def send_text_to_heygen_background(session_id: str, text: str):
    """背景任務：發送文字到HeyGen"""
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

# ================= 健康檢查端點 =================

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
        info["response_id"] = response.id if hasattr(response, 'id') else None
        return info
        
    except Exception as e:
        info["error"] = f"{type(e).__name__}: {str(e)[:150]}"
        return info

@router.get("/health/heygen")
async def health_heygen():
    """HeyGen 健康檢查"""
    heygen_api_key = os.getenv("HEYGEN_API_KEY")
    avatar_id = os.getenv("HEYGEN_AVATAR_ID")
    
    info = {
        "has_api_key": bool(heygen_api_key),
        "has_avatar_id": bool(avatar_id),
        "avatar_id": avatar_id if avatar_id else "not_configured",
        "ok": False,
        "error": None
    }
    
    if not heygen_api_key:
        info["error"] = "HEYGEN_API_KEY not set"
        return info
    
    if not avatar_id:
        info["error"] = "HEYGEN_AVATAR_ID not set"
        return info
    
    try:
        # 簡單的API連通性測試
        headers = {
            "X-API-KEY": heygen_api_key,
            "Content-Type": "application/json"
        }
        
        # 測試用的最小session請求
        test_data = {
            "avatar_name": avatar_id,
            "voice": {"voice_id": "zh-TW-HsiaoChenNeural"},
            "quality": "medium",
            "language": "zh-TW"
        }
        
        async with aiohttp.ClientSession() as session:
            async with session.post(
                "https://api.heygen.com/v2/streaming/create_session",
                headers=headers,
                json=test_data,
                timeout=aiohttp.ClientTimeout(total=10)  # 10秒超時
            ) as response:
                if response.status == 200:
                    result = await response.json()
                    if result.get("code") == 100:
                        # 立即關閉測試會話
                        test_session_id = result["data"]["session_id"]
                        await session.post(
                            "https://api.heygen.com/v2/streaming/close_session",
                            headers=headers,
                            json={"session_id": test_session_id}
                        )
                        info["ok"] = True
                        info["test_session_created"] = True
                    else:
                        info["error"] = result.get("message", "Session creation failed")
                else:
                    error_text = await response.text()
                    info["error"] = f"HTTP {response.status}: {error_text[:100]}"
                    
    except asyncio.TimeoutError:
        info["error"] = "Connection timeout"
    except Exception as e:
        info["error"] = f"{type(e).__name__}: {str(e)[:150]}"
    
    return info

# ================= Debug 端點 =================

@router.get("/personas/info")
async def get_personas_info():
    """取得所有 Persona 的基本資訊（用於除錯）"""
    personas_info = {}
    for bot_type, config in ENHANCED_PERSONA_STYLES.items():
        personas_info[bot_type] = {
            "name": config["name"],
            "system_prompt_length": len(config["system"]),
            "key_features": extract_key_features(config["system"])
        }
    
    return {
        "enhanced_personas": personas_info,
        "total_personas": len(ENHANCED_PERSONA_STYLES),
        "available_types": list(ENHANCED_PERSONA_STYLES.keys()),
        "heygen_enabled": bool(os.getenv("HEYGEN_API_KEY"))
    }

def extract_key_features(system_prompt: str) -> List[str]:
    """從 system prompt 中提取關鍵特徵"""
    features = []
    lines = system_prompt.split('\n')
    for line in lines:
        if '**' in line and ('技巧' in line or '風格' in line or '定位' in line):
            features.append(line.strip().replace('**', ''))
    return features[:3]  # 只返回前3個關鍵特徵

# ================= 路由列表檢查 =================

@router.get("/routes")
async def list_routes():
    """列出所有可用的路由（用於除錯）"""
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