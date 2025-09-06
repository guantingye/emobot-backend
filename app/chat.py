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
4. **思維記錄**：結構化的想法-情緒-行為分析

**回應架構**：
想法捕捉 → 認知檢驗 → 替代思維 → 行為實驗

**共同原則**：
- 保持無條件正向關懷和非評判態度
- 鼓勵自主性和理性思考
- 使用繁體中文，語言親切自然
- 在危機時刻進行安全評估並建議尋求專業協助

範例回應風格：「我注意到你說『我總是搞砸一切』，這聽起來像是『全有全無』的思維模式。讓我們來檢驗一下：

**支持證據**：[請你列出具體事件]
**反對證據**：[你有哪些成功的經驗？]
**更平衡的想法**：『我在某些情況下會犯錯，但也有成功的時候，這是人之常情』

這樣的重新框架讓你感覺如何？」

請以 Clara 的身份，用理性分析的方式協助使用者重建思維模式。"""
    }
}

def get_enhanced_system_prompt(bot_type: str) -> str:
    """取得增強版的系統提示詞"""
    if bot_type in ENHANCED_PERSONA_STYLES:
        return ENHANCED_PERSONA_STYLES[bot_type]["system"]
    # 預設回到解決型
    return ENHANCED_PERSONA_STYLES["solution"]["system"]

def get_bot_name(bot_type: str) -> str:
    """取得機器人名稱"""
    if bot_type in ENHANCED_PERSONA_STYLES:
        return ENHANCED_PERSONA_STYLES[bot_type]["name"]
    return "Niko"

# ================= Schemas (保持不變) =================
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

# ================= OpenAI Integration (保持不變) =================
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

# ================= Enhanced Routes =================
@router.post("/send", response_model=SendResult)
async def send_chat(payload: SendPayload, request: Request, db: Session = Depends(get_db)):
    """聊天端點 - 支援增強版 Persona System Prompts + OpenAI 回覆"""
    
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
        
        # 2. 準備增強版 System Prompt
        enhanced_system_prompt = get_enhanced_system_prompt(payload.bot_type)
        bot_name = get_bot_name(payload.bot_type)
        
        # 3. 轉換歷史記錄格式（保持最近 8 條對話以維持 Persona 一致性）
        messages = []
        for h in payload.history[-8:]:  # 取最近 8 條，保持上下文但避免過長
            role = "assistant" if h.role == "assistant" else "user"
            messages.append({"role": role, "content": h.content})
        
        # 添加當前使用者訊息
        messages.append({"role": "user", "content": user_msg})
        
        # 4. 呼叫 OpenAI（使用增強版 System Prompt）
        reply_text = call_openai(enhanced_system_prompt, messages)
        
        # 5. 確保回覆符合該 Persona 的風格（簡單的後處理）
        reply_text = ensure_persona_consistency(reply_text, payload.bot_type, bot_name)
        
        # 6. 儲存 AI 回覆
        ai_message = ChatMessage(
            user_id=user_id,
            bot_type=payload.bot_type,
            mode=payload.mode,
            role="ai",
            content=reply_text,
            meta={
                "provider": "openai", 
                "model": os.getenv("OPENAI_MODEL", "gpt-4o"),
                "persona": payload.bot_type,
                "bot_name": bot_name
            }
        )
        db.add(ai_message)
        db.commit()
        
        # 7. 返回結果
        return SendResult(
            ok=True,
            reply=reply_text,
            bot={
                "type": payload.bot_type, 
                "name": bot_name,
                "persona": "enhanced"
            },
            error=None
        )
        
    except Exception as e:
        print(f"Enhanced chat send error: {e}")
        db.rollback()
        
        # 緊急回覆，根據不同 Persona 提供不同的 fallback
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

# ================= Persona Enhancement Functions =================
def ensure_persona_consistency(reply: str, bot_type: str, bot_name: str) -> str:
    """確保回覆符合 Persona 風格的簡單後處理"""
    if not reply or len(reply.strip()) == 0:
        return get_fallback_reply(bot_type)
    
    # 確保回覆不會太過機械化
    if reply.startswith("作為AI") or reply.startswith("我是一個AI"):
        return get_fallback_reply(bot_type)
    
    # 根據不同 Persona 進行微調
    if bot_type == "empathy":
        # Lumi: 確保語調溫暖
        if not any(word in reply for word in ["聽起來", "感受到", "理解", "陪著"]):
            reply = f"我能感受到你的心情。{reply}"
    
    elif bot_type == "insight":
        # Solin: 確保有探索性
        if not any(word in reply for word in ["注意到", "想到", "模式", "如何看"]):
            reply = f"我注意到你提到的情況。{reply}"
    
    elif bot_type == "solution":
        # Niko: 確保有行動導向
        if not any(word in reply for word in ["可以", "步驟", "試試", "行動"]):
            reply = f"讓我們一起來看看可以怎麼處理。{reply}"
    
    elif bot_type == "cognitive":
        # Clara: 確保有理性分析
        if not any(word in reply for word in ["檢視", "想法", "證據", "角度"]):
            reply = f"讓我們來檢視一下這個想法。{reply}"
    
    return reply

def get_fallback_reply(bot_type: str) -> str:
    """根據不同 Persona 提供專屬的緊急回覆"""
    fallback_replies = {
        "empathy": "我在這裡陪著你。此刻最強烈的感受是什麼呢？讓我們一起慢慢聊聊。",
        "insight": "讓我們一步步來理解這個情況。你覺得最重要的是哪個部分？我想更深入地了解你的想法。",
        "solution": "我們可以從一個小步驟開始。你想先處理哪個部分？讓我們一起制定一個可行的計畫。",
        "cognitive": "讓我們先識別一下剛剛的自動想法。你能描述一下當時心中想到什麼嗎？我們來一起檢視看看。"
    }
    
    return fallback_replies.get(bot_type, fallback_replies["solution"])

# ================= Health Check (保持不變) =================
@router.get("/health/openai")
@router.post("/health/openai")
async def health_openai():
    """OpenAI 健康檢查"""
    model = os.getenv("OPENAI_MODEL", "gpt-4o")
    key = os.getenv("OPENAI_API_KEY")
    
    info = {
        "model": model,
        "has_key": bool(key),
        "ok": False,
        "error": None,
        "persona_system": "enhanced"
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

# ================= Debug Endpoint (新增) =================
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
        "available_types": list(ENHANCED_PERSONA_STYLES.keys())
    }

def extract_key_features(system_prompt: str) -> List[str]:
    """從 system prompt 中提取關鍵特徵"""
    features = []
    lines = system_prompt.split('\n')
    for line in lines:
        if '**' in line and ('技巧' in line or '風格' in line or '定位' in line):
            features.append(line.strip().replace('**', ''))
    return features[:3]  # 只返回前3個關鍵特徵