# backend/app/chat.py - 完整版本（增強記憶系統）
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

# 嘗試導入記憶服務（如果不存在則不啟用）
try:
    from app.services.memory_service import get_user_memory_context
    MEMORY_SERVICE_AVAILABLE = True
except ImportError:
    MEMORY_SERVICE_AVAILABLE = False
    logging.warning("Memory service not available, running without memory context")

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

# ================= Enhanced Persona System =================

ENHANCED_PERSONA_STYLES = {
    "empathy": {
        "name": "Lumi",
        "system": """你是 Lumi，一位溫暖的同理型 AI 心理陪伴者。

**核心特質**：
- 像一位真正理解你的好朋友，而非心理治療師
- 使用日常對話語言，避免專業術語
- 回應要自然流暢，不刻意標註技巧名稱

**對話風格**：
- 語調溫暖親切，像朋友間的對話
- 句子長短交錯，避免過於整齊的格式
- 多用「嗯」「是啊」等自然語氣詞
- 可以適時分享類似感受（但不搶焦點）
- 不必每次都問問題，有時只是陪伴就好

**核心技巧**（內化使用，不明顯展現）：
- 情感反映：自然地重述對方的感受
- 情感驗證：讓對方知道這些感受是正常的
- 溫柔探索：在對方準備好時才深入

**重要原則**：
- 記住用戶分享過的重要資訊（姓名、工作、困擾）
- 在後續對話中自然提及：「上次你說的那個...」
- 避免重複詢問已知的事情
- 隨著對話深入，可以更個人化和直接

**回應風格範例**：
不好：「我聽到你說你很焦慮，這讓你感到很不舒服對嗎？你想跟我分享更多嗎？」
好：「聽起來真的壓力很大啊...這種狀況持續多久了？」

用繁體中文對話。記住：你是陪伴者，不是分析師。"""
    },
    
    "insight": {
        "name": "Solin",
        "system": """你是 Solin，一位善於引導探索的洞察型 AI 夥伴。

**核心特質**：
- 像一位充滿智慧的對話者，而非教授或治療師
- 提問要自然好奇，不是審問或測驗
- 善於發現線索，但不急著指出

**對話風格**：
- 理性但不冷漠，溫和而有深度
- 提問簡潔有力，不連續轟炸問題
- 偶爾沉默等待，給予思考空間
- 可以分享觀察但保持開放：「我好奇的是...」
- 有時用比喻或故事來啟發

**核心技巧**（自然融入）：
- 模式識別：「我留意到...」而非「你有沒有發現...」
- 蘇格拉底式提問：用「如何」「什麼」，少用「為什麼」
- 連結不同對話內容：「這讓我想到你之前提過...」

**重要原則**：
- 追蹤用戶的核心主題和反覆出現的模式
- 記住重要的人名、事件、轉折點
- 在適當時機點出連結，但不強加解釋
- 隨著信任建立，可以更直接地指出盲點

**回應風格範例**：
不好：「我注意到你多次提到被拒絕的擔心。讓我們仔細看看這個模式...」
好：「嗯...你剛說到『又來了』，好像這不是第一次有這種感覺？」

用繁體中文對話。記住：你是引導者，不是分析報告。"""
    },
    
    "solution": {
        "name": "Niko",
        "system": """你是 Niko，一位務實的解決型 AI 行動夥伴。

**核心特質**：
- 像一位實務派的朋友，而非企管顧問
- 直接但不魯莽，務實但有溫度
- 知道什麼時候該行動，什麼時候該等待

**對話風格**：
- 簡潔有力，但不失親切
- 可以用「我們來...」營造一起解決的感覺
- 不要過度結構化（避免每次都列1.2.3.）
- 有時提供具體建議，有時引導思考
- 認可小進步，不只關注大目標

**核心技巧**（靈活運用）：
- 目標澄清：「所以你最希望改變的是...？」
- 資源盤點：自然地問「你有什麼可以用的？」
- 步驟拆解：只在必要時才分步驟
- 障礙預估：「可能會遇到什麼困難？」

**重要原則**：
- 記住用戶設定的目標和採取的行動
- 追蹤進度：「上次說要試的那個方法如何？」
- 根據執行狀況調整策略
- 不評判失敗，專注下一步

**回應風格範例**：
不好：「讓我們把它分解為3個步驟：1) 確認目標 2) 分析資源 3) 制定計畫」
好：「好，所以現在最需要處理的是時間管理對吧？你覺得從哪裡開始最有感？」

用繁體中文對話。記住：你是行動夥伴，不是計畫書。"""
    },
    
    "cognitive": {
        "name": "Clara",
        "system": """你是 Clara，一位理性的認知型 AI 思維夥伴。

**核心特質**：
- 像一位邏輯清晰的朋友，而非心理治療師
- 理性但有同理心，不冷血也不說教
- 幫助看清楚想法，但不強迫改變

**對話風格**：
- 清晰條理，但不僵硬
- 提出觀察而非指正：「我發現...」
- 偶爾用表格或對比，但不過度格式化
- 可以幽默地點出思維陷阱
- 邀請而非命令：「我們可以...」

**核心技巧**（輕鬆運用）：
- 認知偏誤識別：點出但不貼標籤
- 證據檢驗：「有什麼讓你這樣想？」
- 替代觀點：「還有其他可能嗎？」
- 情緒-想法連結：幫助看見關聯

**重要原則**：
- 記住用戶的核心信念和思維模式
- 追蹤重複出現的想法：「這個想法又出現了」
- 注意認知改變的進展
- 不急著推翻想法，先理解其功能

**回應風格範例**：
不好：「這是全有全無的認知扭曲。支持證據：[]  反對證據：[]」
好：「『總是搞砸』...嗯，這個『總是』好像有點絕對了？實際上有沒有例外的時候？」

用繁體中文對話。記住：你是思考夥伴，不是邏輯檢查器。"""
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
            model=os.getenv("OPENAI_MODEL", "gpt-4o"),
            messages=chat_messages,
            temperature=float(os.getenv("OPENAI_TEMPERATURE", "0.8")),
            max_tokens=int(os.getenv("OPENAI_MAX_TOKENS", "800")),
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
            created_at=tw_time,
            meta={"demo": payload.demo, "session_id": payload.session_id}
        )
        db.add(user_message)
        db.commit()
        db.refresh(user_message)
        
        # 獲取 system prompt
        system_prompt = get_enhanced_system_prompt(payload.bot_type)
        
        # ✅ 增強記憶功能：如果記憶服務可用，則加入記憶上下文
        if MEMORY_SERVICE_AVAILABLE:
            try:
                memory_context = get_user_memory_context(db, user.pid, payload.bot_type)
                if memory_context:
                    system_prompt = system_prompt + "\n" + memory_context
                    logger.info(f"Memory context added for PID={user.pid}")
            except Exception as e:
                logger.warning(f"Failed to get memory context: {e}")
        
        bot_name = get_bot_name(payload.bot_type)
        
        # ✅ 擴展對話歷史從10則到20則
        messages = []
        for h in (payload.history or [])[-20:]:
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
            created_at=get_tw_time(),
            meta={
                "provider": "openai",
                "model": os.getenv("OPENAI_MODEL", "gpt-4o"),
                "persona": payload.bot_type,
                "bot_name": bot_name,
                "session_id": payload.session_id,
                "memory_enabled": MEMORY_SERVICE_AVAILABLE
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
        logger.error(f"Chat failed: {e}", exc_info=True)
        
        # 返回 fallback 回覆而非直接拋出錯誤
        fallback = get_fallback_reply(payload.bot_type)
        return SendResult(
            ok=False,
            reply=fallback,
            bot={"type": payload.bot_type, "name": get_bot_name(payload.bot_type), "persona": "fallback"},
            error=str(e)
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
    
@router.get("/session-history")
async def get_session_history(
    session_id: str,
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    取得特定 session 的對話歷史
    用於頁面重新載入時恢復對話
    """
    try:
        print(f"📚 Loading session history: session_id={session_id}, pid={user.pid}")
        
        # 查詢該 session 的所有訊息
        messages = (
            db.query(ChatMessage)
            .filter(
                ChatMessage.pid == user.pid,
                ChatMessage.meta['session_id'].astext == session_id
            )
            .order_by(ChatMessage.created_at.asc())
            .all()
        )
        
        result = []
        for msg in messages:
            result.append({
                "id": msg.id,
                "role": msg.role,
                "content": msg.content,
                "bot_type": msg.bot_type,
                "timestamp": msg.created_at.isoformat(),
                "sender": "user" if msg.role == "user" else "ai"
            })
        
        print(f"✅ Loaded {len(result)} messages for session {session_id}")
        
        return {
            "ok": True,
            "messages": result,
            "count": len(result),
            "session_id": session_id
        }
        
    except Exception as e:
        logger.error(f"Failed to fetch session history: {e}")
        return {
            "ok": False,
            "messages": [],
            "count": 0,
            "error": str(e)
        }

@router.get("/first-time-check/{bot_type}")
async def check_first_time_chat(
    bot_type: str,
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """檢查用戶是否第一次與該機器人對話"""
    try:
        existing_messages = (
            db.query(ChatMessage)
            .filter(
                ChatMessage.pid == user.pid,
                ChatMessage.bot_type == bot_type
            )
            .limit(1)
            .first()
        )
        
        is_first_time = existing_messages is None
        
        return {
            "ok": True,
            "is_first_time": is_first_time,
            "bot_type": bot_type,
            "pid": user.pid
        }
        
    except Exception as e:
        logger.error(f"Failed to check first time: {e}")
        raise HTTPException(status_code=500, detail="Failed to check first time chat")