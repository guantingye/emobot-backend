# app/chat.py - 整合 D-ID 的聊天路由
import os
import asyncio
import logging
from typing import List, Dict, Any, Optional
from datetime import datetime

from fastapi import APIRouter, Depends, BackgroundTasks, Request, HTTPException
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from app.db.session import get_db
from app.models.chat import ChatMessage
from app.services.did_service import DIDService, DIDVideoRequest, DIDVideoResponse

logger = logging.getLogger(__name__)
router = APIRouter()

# 初始化 D-ID 服務
did_service = DIDService()

# ================= Pydantic Models =================

class SendPayload(BaseModel):
    message: str
    bot_type: Optional[str] = Field(default="solution")
    mode: Optional[str] = Field(default="text") 
    history: Optional[List[Dict[str, str]]] = Field(default_factory=list)
    demo: Optional[bool] = Field(default=False)
    session_id: Optional[str] = Field(default=None)  # 保留 HeyGen 兼容性

class SendResult(BaseModel):
    ok: bool
    reply: str
    bot: Dict[str, Any]
    error: Optional[str] = None
    message_id: Optional[int] = None
    session_id: Optional[str] = None
    video_url: Optional[str] = None  # 新增：D-ID 視頻 URL

# D-ID 相關模型
class DIDTalkRequest(BaseModel):
    script_text: str
    voice_id: str = Field(default="zh-TW-HsiaoChenNeural")
    bot_type: str = Field(default="solution") 
    avatar_url: Optional[str] = None
    presenter_id: Optional[str] = None

class DIDTalkResponse(BaseModel):
    success: bool
    talk_id: Optional[str] = None
    video_url: Optional[str] = None
    status: Optional[str] = None
    error: Optional[str] = None

# ================= Persona System (保持原有) =================

ENHANCED_PERSONA_STYLES = {
    "empathy": {
        "name": "Lumi",
        "system": """你是 Lumi，一位溫暖的同理型 AI 夥伴。以溫柔、非評判、短句回應，優先表達共感與理解。用繁體中文回復。""",
        "avatar_url": "https://create-images-results.d-id.com/DefaultPresenters/Noelle_f/image.jpeg",
        "presenter_id": "amy-jku7W6h58r"
    },
    "insight": {
        "name": "Solin", 
        "system": """你是 Solin，一位善於引導的洞察型 AI 夥伴。以蘇格拉底式對話引導用戶自我發現，維持中性、尊重與好奇。用繁體中文回復。""",
        "avatar_url": "https://create-images-results.d-id.com/DefaultPresenters/Noelle_f/image.jpeg",
        "presenter_id": "amy-jku7W6h58r"
    },
    "solution": {
        "name": "Niko",
        "system": """你是 Niko，一位實用的解決方案型 AI 夥伴。專注提供具體建議和實用策略，結構化且行動導向。用繁體中文回復。""",
        "avatar_url": "https://create-images-results.d-id.com/DefaultPresenters/Noelle_f/image.jpeg", 
        "presenter_id": "amy-jku7W6h58r"
    },
    "cognitive": {
        "name": "Clara",
        "system": """你是 Clara，一位理性的認知重建型 AI 夥伴。幫助識別和重新框架負面思維模式，基於認知行為療法原則。用繁體中文回復。""",
        "avatar_url": "https://create-images-results.d-id.com/DefaultPresenters/Noelle_f/image.jpeg",
        "presenter_id": "amy-jku7W6h58r"
    }
}

def get_bot_name(bot_type: str) -> str:
    return ENHANCED_PERSONA_STYLES.get(bot_type, {}).get("name", "Assistant")

def get_system_prompt(bot_type: str) -> str:
    return ENHANCED_PERSONA_STYLES.get(bot_type, {}).get("system", "你是一個有幫助的AI助手。")

def get_fallback_reply(bot_type: str) -> str:
    fallbacks = {
        "empathy": "我在這裡陪伴你。想和我分享今天讓你印象深刻的事情嗎？",
        "insight": "讓我們一起探索這個問題。你覺得這背後最核心的感受是什麼？", 
        "solution": "我們來一步步解決這個問題。你想從哪個角度開始處理呢？",
        "cognitive": "讓我們重新檢視這個想法。你能告訴我具體是什麼讓你有這種感覺嗎？"
    }
    return fallbacks.get(bot_type, "讓我們繼續對話，我會盡力幫助你。")

# ================= 主要聊天端點 =================

@router.post("/send", response_model=SendResult)
async def send_chat_message(
    payload: SendPayload, 
    background_tasks: BackgroundTasks,
    request: Request,
    db: Session = Depends(get_db)
):
    """發送聊天訊息並生成回應（整合 D-ID）"""
    try:
        user_id = int(request.headers.get("X-User-Id", 0))
        
        # 儲存用戶訊息
        user_message = ChatMessage(
            user_id=user_id,
            bot_type=payload.bot_type,
            mode=payload.mode,
            role="user",
            content=payload.message,
            meta={"demo": payload.demo}
        )
        db.add(user_message)
        db.commit()
        
        # 生成 AI 回應
        system_prompt = get_system_prompt(payload.bot_type)
        
        # 構建對話歷史
        messages = [{"role": "system", "content": system_prompt}]
        for msg in payload.history[-10:]:  # 只保留最近10條
            messages.append({
                "role": msg.get("role", "user"),
                "content": msg.get("content", "")
            })
        messages.append({"role": "user", "content": payload.message})
        
        # 調用 OpenAI API
        ai_response = await generate_openai_response(messages)
        bot_name = get_bot_name(payload.bot_type)
        
        # 儲存 AI 回應
        ai_message = ChatMessage(
            user_id=user_id,
            bot_type=payload.bot_type,
            mode=payload.mode,
            role="ai",
            content=ai_response,
            meta={
                "provider": "openai",
                "persona": payload.bot_type,
                "bot_name": bot_name
            }
        )
        db.add(ai_message)
        db.commit()
        
        # 如果是視頻模式，生成 D-ID 視頻
        video_url = None
        if payload.mode == "video":
            persona = ENHANCED_PERSONA_STYLES.get(payload.bot_type, {})
            did_request = DIDVideoRequest(
                script_text=ai_response,
                voice_id="zh-TW-HsiaoChenNeural",
                avatar_url=persona.get("avatar_url"),
                presenter_id=persona.get("presenter_id")
            )
            
            # 背景任務生成視頻
            background_tasks.add_task(generate_did_video_background, did_request, ai_message.id, db)
        
        return SendResult(
            ok=True,
            reply=ai_response,
            bot={
                "type": payload.bot_type,
                "name": bot_name,
                "persona": payload.bot_type
            },
            message_id=ai_message.id,
            video_url=video_url  # 初始為 None，背景任務完成後更新
        )
        
    except Exception as e:
        logger.error(f"Send chat failed: {e}")
        db.rollback()
        
        # 返回備用回應
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

# ================= D-ID API 端點 =================

@router.post("/did/create_talk", response_model=DIDTalkResponse)
async def create_did_talk(request: DIDTalkRequest):
    """創建 D-ID 說話視頻"""
    persona = ENHANCED_PERSONA_STYLES.get(request.bot_type, {})
    
    did_request = DIDVideoRequest(
        script_text=request.script_text,
        voice_id=request.voice_id,
        avatar_url=request.avatar_url or persona.get("avatar_url"),
        presenter_id=request.presenter_id or persona.get("presenter_id")
    )
    
    result = await did_service.create_talk(did_request)
    
    return DIDTalkResponse(
        success=result.success,
        talk_id=result.talk_id,
        video_url=result.video_url,
        status=result.status,
        error=result.error
    )

@router.get("/did/talk/{talk_id}/status")
async def get_did_talk_status(talk_id: str):
    """查詢 D-ID 說話視頻狀態"""
    result = await did_service.get_talk_status(talk_id)
    
    return {
        "success": result.success,
        "talk_id": talk_id,
        "status": result.status,
        "video_url": result.video_url,
        "error": result.error,
        "data": result.data
    }

@router.delete("/did/talk/{talk_id}")
async def delete_did_talk(talk_id: str):
    """刪除 D-ID 說話視頻"""
    success = await did_service.delete_talk(talk_id)
    return {"success": success}

# ================= 相容性端點（HeyGen -> D-ID 轉換）=================

@router.post("/heygen/create_session") 
async def legacy_create_session():
    """HeyGen 相容性端點 - 轉換為 D-ID 模式"""
    return {
        "success": True,
        "session_id": "did_mode_session",
        "message": "Using D-ID service instead of HeyGen",
        "mode": "did"
    }

@router.post("/heygen/send_text")
async def legacy_send_text(request: dict):
    """HeyGen 相容性端點 - 轉換為 D-ID 請求"""
    script_text = request.get("text", "")
    if not script_text:
        return {"success": False, "error": "No text provided"}
    
    # 使用預設設定創建 D-ID 視頻
    did_request = DIDVideoRequest(
        script_text=script_text,
        voice_id="zh-TW-HsiaoChenNeural"
    )
    
    result = await did_service.create_talk(did_request)
    
    return {
        "success": result.success,
        "talk_id": result.talk_id,
        "video_url": result.video_url,
        "error": result.error,
        "message": "Converted to D-ID service"
    }

@router.post("/heygen/close_session")
async def legacy_close_session():
    """HeyGen 相容性端點"""
    return {"success": True, "message": "D-ID mode - no session to close"}

# ================= 背景任務 =================

async def generate_did_video_background(request: DIDVideoRequest, message_id: int, db: Session):
    """背景任務：生成 D-ID 視頻"""
    try:
        result = await did_service.create_talk(request)
        
        if result.success and result.video_url:
            # 更新資料庫中的訊息，添加視頻 URL
            message = db.query(ChatMessage).filter(ChatMessage.id == message_id).first()
            if message:
                meta = message.meta or {}
                meta["video_url"] = result.video_url
                meta["talk_id"] = result.talk_id
                meta["video_status"] = "completed"
                message.meta = meta
                db.commit()
                logger.info(f"D-ID video generated for message {message_id}: {result.video_url}")
        else:
            # 記錄失敗
            message = db.query(ChatMessage).filter(ChatMessage.id == message_id).first()
            if message:
                meta = message.meta or {}
                meta["video_status"] = "failed"
                meta["video_error"] = result.error
                message.meta = meta
                db.commit()
                logger.error(f"D-ID video generation failed for message {message_id}: {result.error}")
                
    except Exception as e:
        logger.error(f"Background D-ID generation failed: {e}")

# ================= OpenAI 調用 =================

async def generate_openai_response(messages: List[Dict]) -> str:
    """調用 OpenAI API 生成回應"""
    try:
        import openai
        client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        
        response = await asyncio.get_event_loop().run_in_executor(
            None,
            lambda: client.chat.completions.create(
                model=os.getenv("OPENAI_MODEL", "gpt-4o"),
                messages=messages,
                max_tokens=500,
                temperature=0.7
            )
        )
        
        return response.choices[0].message.content.strip()
        
    except Exception as e:
        logger.error(f"OpenAI API failed: {e}")
        return "抱歉，我現在無法回應。請稍後再試。"

# ================= 健康檢查端點 =================

@router.get("/health/did")
async def health_did():
    """D-ID 服務健康檢查"""
    return await did_service.health_check()

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
        "did_integration": bool(os.getenv("DID_API_KEY"))
    }
    
    if not key:
        info["error"] = "OPENAI_API_KEY not set"
        return info
    
    try:
        import openai
        client = openai.OpenAI(api_key=key)
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
        "did_routes": [r for r in routes if 'did' in r['path']],
        "legacy_heygen_routes": [r for r in routes if 'heygen' in r['path']],
        "api_version": "D-ID_integrated"
    }