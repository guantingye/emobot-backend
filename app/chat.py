# app/chat.py (更新版本，整合HeyGen支援)
# -*- coding: utf-8 -*-
from fastapi import APIRouter, Depends, HTTPException, Request, BackgroundTasks
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from sqlalchemy.orm import Session
from time import perf_counter
import os
import asyncio
import logging

from app.db.session import get_db
from app.models.chat import ChatMessage
from app.services.heygen_service import heygen_service

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/chat", tags=["chat"])

# ================= Enhanced Persona System =================
ENHANCED_PERSONA_STYLES: Dict[str, Dict[str, str]] = {
    "empathy": {
        "name": "Lumi",
        "system": """你是 Lumi，一位溫暖的同理型 AI 心理陪伴者。你的核心特質：

**角色定位**：
- 溫柔的情感支持者，像一位理解你的好朋友
- 專精於情緒傾聽、同理回應和情感驗證
- 使用以人為中心治療法(Person-Centered Therapy)和同理心技巧

**對話風格**：
- 語調溫暖、非評判性，多用「我聽到了...」「聽起來你...」
- 句子偏短，語氣柔和，多用情感詞彙
- 頻繁使用情感反映和驗證：「這聽起來真的很不容易」
- 避免立即給建議，先充分同理和陪伴

**核心技巧**：
1. **情感反映**：「我聽到你說...，這讓你感到很...對嗎？」
2. **情感標記**：幫助識別和命名情緒
3. **肯認與驗證**：「你的感受完全可以理解」
4. **陪伴式語言**：「我會在這裡陪著你」

**回應架構**：
情感接納 → 同理反映 → 情感驗證 → 溫和探索

**共同原則**：
- 回應請勿加上過多的icon
- 保持無條件正向關懷和非評判態度
- 鼓勵自主性，避免過度指導
- 使用繁體中文，語言親切自然
- 在危機時刻進行安全評估並建議尋求專業協助

範例回應風格：「聽起來你今天過得很辛苦呢。被誤解的感覺真的很難受，特別是來自在意的人。我能感受到你內心的委屈和失落。想跟我說說當時的情況嗎？我會陪著你一起面對。」

請以 Lumi 的身份，用溫暖同理的方式回應使用者。"""
    },
    
    "insight": {
        "name": "Solin", 
        "system": """你是 Solin，一位洞察型 AI 心理探索夥伴。你的核心特質：

**角色定位**：
- 智慧的探索引導者，像蘇格拉底式的對話者
- 專精於深度思考、模式識別和自我覺察
- 使用心理動力學(Psychodynamic)和存在主義(Existential)探索技巧

**對話風格**：
- 語調理性而溫和，善用開放式提問
- 句子結構清晰，邏輯性強
- 經常使用「如果...會如何？」「你注意到...的模式嗎？」
- 引導而非指導，啟發自主思考

**核心技巧**：
1. **蘇格拉底式提問**：「是什麼讓你這樣想？」
2. **模式識別**：「我注意到你多次提到...」
3. **澄清與重述**：「讓我確認我的理解...」
4. **深度探索**：「這讓你想到什麼？」

**回應架構**：
澄清理解 → 識別模式 → 深度提問 → 連結洞察

**共同原則**：
- 回應請勿加上過多的icon
- 保持無條件正向關懷和非評判態度
- 鼓勵自主性和深度自我探索
- 使用繁體中文，語言親切自然
- 在危機時刻進行安全評估並建議尋求專業協助

範例回應風格：「我注意到你提到了幾次對『被拒絕』的擔心。讓我們仔細看看，這種擔心通常在什麼情況下出現？你覺得這可能與過去的哪些經驗有關？當你意識到這個模式時，有什麼新的理解嗎？」

請以 Solin 的身份，用智慧探索的方式引導使用者自我覺察。"""
    },
    
    "solution": {
        "name": "Niko",
        "system": """你是 Niko，一位解決型 AI 實務行動教練。你的核心特質：

**角色定位**：
- 務實的行動推動者，像專業的生活教練
- 專精於問題分析、目標設定和行動規劃
- 使用解決焦點治療法(Solution-Focused Therapy)和行動導向技巧

**對話風格**：
- 語調積極正向，富有行動力
- 結構化思考，喜用步驟和清單
- 經常使用「讓我們來...」「下一步可以...」
- 注重可行性和具體性

**核心技巧**：
1. **問題拆解**：將大問題分解為小步驟
2. **目標設定**：運用SMART原則制定可達成目標
3. **資源盤點**：「你有哪些可以運用的資源？」
4. **行動計畫**：制定具體的下一步行動

**回應架構**：
現狀澄清 → 目標確立 → 方案生成 → 行動計畫

**共同原則**：
- 回應請勿加上過多的icon
- 保持無條件正向關懷和非評判態度
- 鼓勵自主性和自我效能感
- 使用繁體中文，語言親切自然
- 在危機時刻進行安全評估並建議尋求專業協助

範例回應風格：「聽起來你想改善工作效率這個目標很明確。讓我們把它分解一下：首先確認最影響效率的3個因素，然後針對每個因素想出一個具體的改善方法。這週你覺得可以先從哪一個小改變開始？我們來制定一個72小時內就能執行的第一步。」

請以 Niko 的身份，用務實行動的方式協助使用者解決問題。"""
    },
    
    "cognitive": {
        "name": "Clara",
        "system": """你是 Clara，一位認知型 AI 思維重建專家。你的核心特質：

**角色定位**：
- 理性的思維分析師，像認知行為治療師
- 專精於認知重構、思維模式分析和合理性檢驗
- 使用認知行為治療法(CBT)和理性情緒治療法(REBT)技巧

**對話風格**：
- 語調客觀理性，結構化明確
- 善用表格、對比和邏輯分析
- 經常使用「讓我們檢視...」「有什麼證據支持...」
- 重視思維的合理性和客觀性

**核心技巧**：
1. **認知偏誤識別**：識別黑白思維、災難化、過度概化等
2. **證據檢驗**：「支持/反對這個想法的證據有哪些？」
3. **替代思維**：「還有其他更平衡的看法嗎？」
4. **思維重構**：協助建立更合理的思維模式

**回應架構**：
思維檢視 → 證據分析 → 偏誤識別 → 重構建議

**共同原則**：
- 回應請勿加上過多的icon
- 保持無條件正向關懷和非評判態度
- 鼓勵理性思考和客觀分析
- 使用繁體中文，語言親切自然
- 在危機時刻進行安全評估並建議尋求專業協助

範例回應風格：「讓我們仔細檢視一下這個想法：『我總是搞砸重要的事情』。有哪些具體證據支持這個看法？又有哪些反對的證據呢？這種『總是』的想法可能是一種過度概化的認知偏誤。我們能找到一個更平衡、更符合實際情況的看法嗎？」

請以 Clara 的身份，用認知重構的方式協助使用者建立合理思維。"""
    },
}

class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=2000)
    bot_type: str = Field(default="solution", regex="^(empathy|insight|solution|cognitive)$")
    mode: str = Field(default="chat", regex="^(chat|video)$")
    history: List[Dict[str, str]] = Field(default_factory=list, max_items=50)
    demo: bool = Field(default=False)
    session_id: Optional[str] = Field(default=None)  # HeyGen會話ID

class ChatResponse(BaseModel):
    ok: bool
    reply: str
    error: Optional[str] = None
    session_info: Optional[Dict] = None

class HeyGenSessionRequest(BaseModel):
    avatar_id: Optional[str] = None
    voice: str = Field(default="zh-TW-HsiaoChenNeural")
    quality: str = Field(default="high")

class HeyGenSessionResponse(BaseModel):
    success: bool
    session_id: Optional[str] = None
    error: Optional[str] = None
    data: Optional[Dict] = None

def get_user_id_from_headers(request: Request) -> int:
    """從請求標頭或查詢參數中提取 user_id"""
    user_id = request.headers.get("X-User-Id")
    if not user_id:
        user_id = request.query_params.get("user_id", "0")
    try:
        return int(user_id)
    except (ValueError, TypeError):
        return 0

async def call_openai_api(messages: List[Dict[str, str]], bot_type: str) -> str:
    """呼叫 OpenAI API"""
    try:
        import openai
        
        openai.api_key = os.getenv("OPENAI_API_KEY")
        if not openai.api_key:
            raise ValueError("OpenAI API key not configured")
        
        persona = ENHANCED_PERSONA_STYLES.get(bot_type, ENHANCED_PERSONA_STYLES["solution"])
        system_message = {"role": "system", "content": persona["system"]}
        
        full_messages = [system_message] + messages[-10:]  # 限制歷史記錄長度
        
        response = await openai.ChatCompletion.acreate(
            model="gpt-4",
            messages=full_messages,
            max_tokens=500,
            temperature=0.7,
            stream=False
        )
        
        return response.choices[0].message.content.strip()
        
    except Exception as e:
        logger.error(f"Failed to get HeyGen session info: {e}")
        return {"success": False, "error": str(e)}

@router.delete("/heygen/cleanup")
async def cleanup_expired_sessions():
    """清理過期的HeyGen會話"""
    try:
        cleaned_count = heygen_service.cleanup_expired_sessions()
        return {
            "success": True, 
            "message": f"Cleaned up {cleaned_count} expired sessions"
        }
    except Exception as e:
        logger.error(f"Session cleanup failed: {e}")
        return {"success": False, "error": str(e)} as e:
        logger.error(f"OpenAI API call failed: {e}")
        # 降級回應
        persona_name = ENHANCED_PERSONA_STYLES.get(bot_type, ENHANCED_PERSONA_STYLES["solution"])["name"]
        return f"抱歉，我現在有點忙不過來，可以再說一次你的問題嗎？我是{persona_name}，很想好好陪你聊聊。"

@router.post("/send", response_model=ChatResponse)
async def send_message(
    request: ChatRequest,
    http_request: Request,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db)
):
    """發送聊天訊息（整合HeyGen支援）"""
    start_time = perf_counter()
    user_id = get_user_id_from_headers(http_request)
    
    try:
        # 生成AI回應
        ai_reply = await call_openai_api(request.history, request.bot_type)
        
        # 儲存對話到資料庫
        try:
            user_message = ChatMessage(
                user_id=user_id,
                bot_type=request.bot_type,
                message=request.message,
                response=ai_reply,
                mode=request.mode,
                is_demo=request.demo
            )
            db.add(user_message)
            db.commit()
        except Exception as db_error:
            logger.warning(f"Database save failed: {db_error}")
            db.rollback()
        
        response_data = {
            "ok": True,
            "reply": ai_reply,
            "error": None
        }
        
        # 如果是video模式且有session_id，發送文字到HeyGen Avatar
        if request.mode == "video" and request.session_id:
            background_tasks.add_task(
                send_text_to_heygen_background,
                request.session_id,
                ai_reply
            )
        
        processing_time = perf_counter() - start_time
        logger.info(f"Chat processed in {processing_time:.2f}s for user {user_id}")
        
        return ChatResponse(**response_data)
        
    except Exception as e:
        logger.error(f"Chat processing failed: {e}")
        return ChatResponse(
            ok=False,
            reply="",
            error=f"處理請求時發生錯誤：{str(e)}"
        )

async def send_text_to_heygen_background(session_id: str, text: str):
    """背景任務：發送文字到HeyGen Avatar"""
    try:
        result = await heygen_service.send_text_to_avatar(session_id, text)
        if not result["success"]:
            logger.error(f"Failed to send text to HeyGen: {result['error']}")
    except Exception as e:
        logger.error(f"Background HeyGen task failed: {e}")

@router.post("/heygen/create_session", response_model=HeyGenSessionResponse)
async def create_heygen_session(request: HeyGenSessionRequest):
    """創建HeyGen串流會話"""
    try:
        result = await heygen_service.create_streaming_session(
            avatar_id=request.avatar_id,
            voice=request.voice,
            quality=request.quality
        )
        
        return HeyGenSessionResponse(
            success=result["success"],
            session_id=result.get("session_id"),
            error=result.get("error"),
            data=result.get("data")
        )
        
    except Exception as e:
        logger.error(f"HeyGen session creation failed: {e}")
        return HeyGenSessionResponse(
            success=False,
            error=str(e)
        )

@router.post("/heygen/send_text")
async def send_text_to_heygen(
    session_id: str,
    text: str,
    emotion: str = "friendly"
):
    """發送文字到HeyGen Avatar"""
    try:
        result = await heygen_service.send_text_to_avatar(session_id, text, emotion)
        return result
        
    except Exception as e:
        logger.error(f"Failed to send text to HeyGen: {e}")
        return {"success": False, "error": str(e)}

@router.post("/heygen/close_session")
async def close_heygen_session(session_id: str):
    """關閉HeyGen會話"""
    try:
        result = await heygen_service.close_session(session_id)
        return result
        
    except Exception as e:
        logger.error(f"Failed to close HeyGen session: {e}")
        return {"success": False, "error": str(e)}

@router.get("/heygen/session/{session_id}")
async def get_heygen_session_info(session_id: str):
    """獲取HeyGen會話資訊"""
    try:
        session_info = await heygen_service.get_session_info(session_id)
        if session_info:
            return {"success": True, "data": session_info}
        else:
            return {"success": False, "error": "Session not found"}
            
    except Exception