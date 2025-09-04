# app/chat.py
# -*- coding: utf-8 -*-
from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from sqlalchemy.orm import Session
from time import perf_counter
import os

from app.db.session import get_db
from app.models.chat import ChatMessage

router = APIRouter(prefix="/api/chat", tags=["chat"])

# ================= Persona =================
PERSONA_STYLES: Dict[str, Dict[str, str]] = {
    "empathy":  {
        "name": "Lumi",
        "system": "你是 Lumi，同理型 AI。以溫柔、非評判、短句的反映傾聽與情緒標記來回應。優先肯認、共感與陪伴。用繁體中文回覆，保持溫暖支持的語調。"
    },
    "insight":  {
        "name": "Solin",
        "system": "你是 Solin，洞察型 AI。以蘇格拉底式提問、澄清與重述，幫助使用者重清想法，維持中性、尊重、結構化。用繁體中文回覆。"
    },
    "solution": {
        "name": "Niko", 
        "system": "你是 Niko，解決型 AI。以務實、具體的建議與分步行動為主，給出小目標、工具與下一步，語氣鼓勵但不強迫。用繁體中文回覆。"
    },
    "cognitive": {
        "name": "Clara",
        "system": "你是 Clara，認知型 AI。以 CBT 語氣幫助辨識自動想法、認知偏誤與替代想法，提供簡短表格式步驟與練習。用繁體中文回覆。"
    },
}

def get_system_prompt(bot_type: str) -> str:
    if bot_type in PERSONA_STYLES:
        return PERSONA_STYLES[bot_type]["system"]
    return PERSONA_STYLES["solution"]["system"]  # 預設

# ================= Schemas =================
class HistoryItem(BaseModel):
    role: str
    content: str

class SendPayload(BaseModel):
    bot_type: str = Field(..., pattern="^(empathy|insight|solution|cognitive)$")
    mode: str = Field("text", pattern="^(text|video)$")
    message: str
    history: List[HistoryItem] = []
    demo: bool = False
    video_meta: Optional[Dict[str, Any]] = None

class SendResult(BaseModel):
    ok: bool
    reply: Optional[str] = None
    bot: Optional[Dict[str, Any]] = None
    error: Optional[str] = None

# ================= OpenAI =================
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

# ================= Routes =================
@router.post("/send", response_model=SendResult)
async def send_chat(payload: SendPayload, request: Request, db: Session = Depends(get_db)):
    """聊天端點 - 支援 OpenAI 回覆"""
    
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
            meta={"demo": payload.demo, "video_meta": payload.video_meta}
        )
        db.add(user_message)
        db.commit()
        
        # 2. 準備 OpenAI 請求
        system_prompt = get_system_prompt(payload.bot_type)
        
        # 轉換歷史記錄格式
        messages = []
        for h in payload.history[-10:]:  # 只取最近 10 條
            role = "assistant" if h.role == "assistant" else "user"
            messages.append({"role": role, "content": h.content})
        
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
            meta={"provider": "openai", "model": os.getenv("OPENAI_MODEL", "gpt-4o-mini")}
        )
        db.add(ai_message)
        db.commit()
        
        # 5. 返回結果
        name_map = {"empathy": "Lumi", "insight": "Solin", "solution": "Niko", "cognitive": "Clara"}
        return SendResult(
            ok=True,
            reply=reply_text,
            bot={"type": payload.bot_type, "name": name_map.get(payload.bot_type)},
            error=None
        )
        
    except Exception as e:
        print(f"Chat send error: {e}")
        db.rollback()
        
        # 緊急回覆，確保使用者體驗
        fallback_replies = {
            "empathy": "我在這裡陪著你。此刻最強烈的感受是什麼？",
            "insight": "讓我們一步步來理解這個情況。你覺得最重要的是哪個部分？",
            "solution": "我們可以從一個小步驟開始。你想先處理哪個部分？",
            "cognitive": "讓我們先識別一下剛剛的自動想法。你能描述一下當時心中想到什麼嗎？"
        }
        
        fallback_text = fallback_replies.get(payload.bot_type, fallback_replies["solution"])
        
        # 儲存緊急回覆
        try:
            ai_message = ChatMessage(
                user_id=user_id,
                bot_type=payload.bot_type,
                mode=payload.mode,
                role="ai",
                content=fallback_text,
                meta={"provider": "fallback", "error": str(e)[:200]}
            )
            db.add(ai_message)
            db.commit()
        except Exception:
            pass  # 如果連 fallback 都失敗，就不儲存了
        
        name_map = {"empathy": "Lumi", "insight": "Solin", "solution": "Niko", "cognitive": "Clara"}
        return SendResult(
            ok=True,  # 仍然回傳 ok=True，確保前端正常顯示
            reply=fallback_text,
            bot={"type": payload.bot_type, "name": name_map.get(payload.bot_type)},
            error=f"API temporarily unavailable: {str(e)[:100]}"
        )

# ================= Health Check =================
@router.get("/health/openai")
@router.post("/health/openai")
async def health_openai():
    """OpenAI 健康檢查"""
    model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    key = os.getenv("OPENAI_API_KEY")
    
    info = {
        "model": model,
        "has_key": bool(key),
        "ok": False,
        "error": None
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