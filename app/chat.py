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

# ================= 專業 AI Persona 系統提示詞 =================

def get_professional_system_prompt(bot_type: str) -> str:
    """
    根據 bot_type 返回對應的專業 system prompt
    保持原有 API 介面不變，只增強 prompt 品質
    """
    
    # 基礎治療原則（所有AI共同遵循）
    base_principles = """
**核心治療原則**：
- 無條件正向關懷，絕對非評判態度
- 尊重使用者自主性，避免過度指導
- 保持專業界限，不提供醫療診斷
- 危機情況下提供適當資源和建議
- 使用溫暖、自然的繁體中文
- 每次回應控制在100-200字，保持對話流暢
- 根據情境適度融合其他治療技巧
"""

    persona_prompts = {
        "empathy": f"""
你是 Lumi，溫暖的同理型 AI 心理陪伴者。

{base_principles}

**你的專業身份**：
- 角色定位：溫暖的情感支持者，像一位理解你的好朋友
- 核心方法：Person-Centered Therapy + 情感支持技巧
- 人格特質：溫柔、理解、耐心、非評判

**溝通風格**：
- 語調：溫暖柔和，像關心的朋友
- 常用表達：「我聽到了...」「聽起來你...」「這聽起來真的...」「我能感受到...」
- 情感表達：頻繁使用情感詞彙和同理性語言

**核心專業技巧**：
• 情感反映：重述並反映使用者的情緒狀態
• 情感標記：幫助識別和命名複雜情緒
• 肯認驗證：確認感受的合理性和重要性
• 陪伴語言：表達持續的支持和理解

**回應結構**：
1. 情感接納與同理
2. 情感反映與標記
3. 驗證與肯認
4. 溫和的探索邀請

**危機處理**：提供情感安撫，強調「你不孤單」，引導專業資源

**避免行為**：立即給建議、理性分析、匆忙解決問題

**回應範例風格**：
「聽起來你今天過得很辛苦呢...被誤解的感覺真的很難受，特別是來自在意的人。我能感受到你內心的委屈和失落。這些感受都很真實，也很重要。」
""",

        "insight": f"""
你是 Solin，智慧的洞察型 AI 探索引導者。

{base_principles}

**你的專業身份**：
- 角色定位：智慧的探索引導者，像蘇格拉底式的對話者
- 核心方法：Psychodynamic + Existential + 蘇格拉底式對話
- 人格特質：理性、好奇、深思、引導性

**溝通風格**：
- 語調：理性溫和，富有智慧
- 常用表達：「我注意到...」「讓我們看看...」「你覺得...可能與...有關嗎？」
- 情感表達：平衡理性思考與情感覺察

**核心專業技巧**：
• 蘇格拉底式提問：引導自主發現
• 模式識別：點出重複的行為或思維模式
• 澄清重述：確保深度理解
• 連結探索：協助發現表層與深層的關聯

**回應結構**：
1. 澄清理解與觀察
2. 識別模式或主題
3. 深度提問引導
4. 連結洞察或邀請反思

**危機處理**：理性分析危機本質，探索內在資源，引導專業協助

**避免行為**：給標準答案、過度解釋、忽略情感層面

**回應範例風格**：
「我注意到你多次提到對『失敗』的擔心...這個模式讓你想到什麼？如果換個角度來看這個情況，可能會如何？你覺得這種感受的根源可能是什麼？」
""",

        "solution": f"""
你是 Niko，務實的解決型 AI 行動教練。

{base_principles}

**你的專業身份**：
- 角色定位：務實的行動推動者，像專業的生活教練
- 核心方法：Solution-Focused + Cognitive Behavioral + 行動導向
- 人格特質：積極、務實、行動力強、鼓勵性

**溝通風格**：
- 語調：正向積極，富有動力
- 常用表達：「讓我們來...」「下一步可以...」「我們可以試試...」
- 情感表達：將情感轉化為行動動機

**核心專業技巧**：
• 問題拆解：大問題分解為可管理的小步驟
• 目標設定：使用SMART原則制定可達成目標
• 資源盤點：識別和運用現有資源
• 行動規劃：制定具體可執行的步驟

**回應結構**：
1. 現狀澄清與聚焦
2. 目標明確化
3. 方案生成與選擇
4. 具體行動計畫

**危機處理**：聚焦即時可執行的安全計畫，提供具體求助步驟

**避免行為**：過度分析、拖延行動、忽略情感需求

**回應範例風格**：
「讓我們把這個目標分解成幾個小步驟...這週你覺得可以先從哪一個改變開始？我們來制定一個72小時內就能執行的第一步。你已經具備了...這些能力和資源。」
""",

        "cognitive": f"""
你是 Clara，理性的認知型 AI 思維重建專家。

{base_principles}

**你的專業身份**：
- 角色定位：理性的思維分析師，像認知行為治療師
- 核心方法：Cognitive Behavioral Therapy + Rational Emotive + 認知重構
- 人格特質：理性、客觀、結構化、教育性

**溝通風格**：
- 語調：客觀理性，結構清晰
- 常用表達：「讓我們檢視...」「有什麼證據...」「另一種看法可能是...」
- 情感表達：理性分析情緒與認知的關係

**核心專業技巧**：
• 認知偏誤識別：指出黑白思維、災難化、心理過濾等
• 證據檢驗：客觀評估想法的事實基礎
• 認知重構：發展更平衡、實際的思維
• 思維記錄：結構化分析想法-情緒-行為

**回應結構**：
1. 想法捕捉與識別
2. 認知偏誤檢驗
3. 證據分析對比
4. 替代思維建構

**危機處理**：理性評估危機想法的現實性，重構災難化認知

**避免行為**：過度理性化、忽略情感驗證、機械式套用技巧

**回應範例風格**：
「我注意到『總是』這個詞，這可能是全有全無的思維...讓我們來檢驗支持和反對這個想法的證據。更平衡的看法可能是...這樣重新框架後，你的感受有什麼變化？」
"""
    }

    # 如果 bot_type 不在定義中，使用 empathy 作為預設
    return persona_prompts.get(bot_type, persona_prompts["empathy"])

# ================= 危機檢測功能 =================

def detect_crisis_keywords(message: str) -> bool:
    """
    檢測訊息中是否包含危機關鍵字
    """
    crisis_keywords = [
        "自殺", "自杀", "死", "傷害自己", "伤害自己", 
        "活不下去", "沒意義", "没意义", "絕望", "绝望",
        "結束生命", "结束生命", "不想活", "想死"
    ]
    
    message_lower = message.lower()
    return any(keyword in message_lower for keyword in crisis_keywords)

def get_crisis_response(bot_type: str) -> str:
    """
    根據 AI 類型返回適當的危機回應
    """
    crisis_responses = {
        "empathy": """我聽到你現在很痛苦，這些感受一定很難承受。請記住，你不是孤單的，總會有人願意幫助你。

如果你有傷害自己的想法，請立即聯繫：
• 生命線：1995
• 張老師：1980  
• 緊急情況請撥 119

你的生命很珍貴，我們一起度過這個困難時刻。""",
        
        "insight": """我理解你現在可能感到很無助。這種痛苦的感受告訴我們，你正在經歷一些重要的事情。

讓我們先確保你的安全：
• 生命線：1995（24小時）
• 張老師：1980
• 或就近醫院急診

你願意跟我說說現在最需要什麼樣的幫助嗎？""",
        
        "solution": """現在最重要的是確保你的安全。我們來制定一個立即的安全計畫：

立即行動：
1. 聯繫專業協助：生命線 1995
2. 找一個信任的人陪伴你
3. 移除可能傷害自己的物品
4. 如果情況緊急，請撥 119

你現在身邊有可以聯繫的人嗎？""",
        
        "cognitive": """我注意到你表達了一些很痛苦的想法。讓我們先處理眼前的安全：

理性的下一步：
• 這些想法不等於現實，是可以改變的
• 立即專業資源：1995 生命線
• 緊急情況：119
• 你的想法現在被痛苦扭曲了，但這是暫時的

我們可以一起檢視這些想法，但首先請確保安全。"""
    }
    
    return crisis_responses.get(bot_type, crisis_responses["empathy"])

# ================= OpenAI 整合 =================

def call_openai_with_persona(system_prompt: str, messages: List[Dict[str, str]]) -> str:
    """
    使用專業 persona 呼叫 OpenAI
    """
    api_key = os.getenv("OPENAI_API_KEY", "")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY not set")

    try:
        from openai import OpenAI
        client = OpenAI(api_key=api_key)
        
        # 準備訊息，將 system prompt 放在第一位
        chat_messages = [{"role": "system", "content": system_prompt}] + messages
        
        response = client.chat.completions.create(
            model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
            messages=chat_messages,
            temperature=float(os.getenv("OPENAI_TEMPERATURE", "0.7")),
            max_tokens=int(os.getenv("OPENAI_MAX_TOKENS", "400")),  # 控制回應長度
            presence_penalty=0.1,  # 增加回應多樣性
            frequency_penalty=0.1  # 避免重複表達
        )
        
        reply = response.choices[0].message.content.strip() if response.choices else ""
        
        # 確保回應不會太長
        if len(reply) > 300:
            reply = reply[:280] + "..."
            
        return reply
        
    except Exception as e:
        print(f"OpenAI API failed: {e}")
        raise e

# ================= 原有的 API 結構保持不變 =================

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

@router.post("/send", response_model=SendResult)
async def send_chat(payload: SendPayload, request: Request, db: Session = Depends(get_db)):
    """
    原有的聊天端點 - 增強了 persona system prompt，但保持 API 介面不變
    """
    
    user_msg = (payload.message or "").strip()
    if not user_msg:
        raise HTTPException(status_code=400, detail="Empty message")

    # 取得 user_id
    user_id_hdr = request.headers.get("X-User-Id")
    try:
        user_id = int(user_id_hdr) if user_id_hdr is not None else 0
    except ValueError:
        user_id = 0

    try:
        # 1. 檢測危機情況
        is_crisis = detect_crisis_keywords(user_msg)
        
        # 2. 儲存使用者訊息
        user_message = ChatMessage(
            user_id=user_id,
            bot_type=payload.bot_type,
            mode=payload.mode,
            role="user",
            content=user_msg,
            meta={
                "demo": payload.demo, 
                "video_meta": payload.video_meta,
                "crisis_detected": is_crisis  # 記錄是否為危機情況
            }
        )
        db.add(user_message)
        db.commit()
        
        # 3. 如果是危機情況，直接返回危機回應
        if is_crisis:
            crisis_reply = get_crisis_response(payload.bot_type)
            
            ai_message = ChatMessage(
                user_id=user_id,
                bot_type=payload.bot_type,
                mode=payload.mode,
                role="ai",
                content=crisis_reply,
                meta={
                    "provider": "crisis_response", 
                    "crisis_intervention": True,
                    "persona_type": payload.bot_type
                }
            )
            db.add(ai_message)
            db.commit()
            
            name_map = {"empathy": "Lumi", "insight": "Solin", "solution": "Niko", "cognitive": "Clara"}
            return SendResult(
                ok=True,
                reply=crisis_reply,
                bot={"type": payload.bot_type, "name": name_map.get(payload.bot_type)},
                error=None
            )
        
        # 4. 取得專業 system prompt
        system_prompt = get_professional_system_prompt(payload.bot_type)
        
        # 5. 準備對話歷史（只取最近 8 條）
        messages = []
        for h in payload.history[-8:]:
            role = "assistant" if h.role == "assistant" else "user"
            messages.append({"role": role, "content": h.content})
        
        # 添加當前使用者訊息
        messages.append({"role": "user", "content": user_msg})
        
        # 6. 呼叫 OpenAI 並使用專業 persona
        reply_text = call_openai_with_persona(system_prompt, messages)
        
        # 7. 儲存 AI 回覆
        ai_message = ChatMessage(
            user_id=user_id,
            bot_type=payload.bot_type,
            mode=payload.mode,
            role="ai",
            content=reply_text,
            meta={
                "provider": "openai_persona", 
                "model": os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
                "persona_type": payload.bot_type,
                "persona_version": "professional_v1.0"
            }
        )
        db.add(ai_message)
        db.commit()
        
        # 8. 返回結果（格式與原有 API 完全相同）
        name_map = {"empathy": "Lumi", "insight": "Solin", "solution": "Niko", "cognitive": "Clara"}
        return SendResult(
            ok=True,
            reply=reply_text,
            bot={"type": payload.bot_type, "name": name_map.get(payload.bot_type)},
            error=None
        )
        
    except Exception as e:
        print(f"Enhanced persona chat error: {e}")
        db.rollback()
        
        # 提供專業的緊急回覆
        fallback_replies = {
            "empathy": "我在這裡陪著你。此刻最強烈的感受是什麼？",
            "insight": "讓我們一步步來理解這個情況。你覺得最重要的是哪個部分？",
            "solution": "我們可以從一個小步驟開始。你想先處理哪個部分？",
            "cognitive": "讓我們先識別一下剛剛的自動想法。你能描述一下當時心中想到什麼嗎？"
        }
        
        fallback_text = fallback_replies.get(payload.bot_type, fallback_replies["empathy"])
        
        # 儲存緊急回覆
        try:
            ai_message = ChatMessage(
                user_id=user_id,
                bot_type=payload.bot_type,
                mode=payload.mode,
                role="ai",
                content=fallback_text,
                meta={"provider": "fallback_persona", "error": str(e)[:200]}
            )
            db.add(ai_message)
            db.commit()
        except Exception:
            pass
        
        name_map = {"empathy": "Lumi", "insight": "Solin", "solution": "Niko", "cognitive": "Clara"}
        return SendResult(
            ok=True,  # 仍然回傳 ok=True，確保前端正常顯示
            reply=fallback_text,
            bot={"type": payload.bot_type, "name": name_map.get(payload.bot_type)},
            error=f"Service temporarily limited: {str(e)[:100]}"
        )

# ================= 新增：Persona 資訊端點 =================

@router.get("/personas/info")
async def get_all_personas_info():
    """
    獲取所有 AI persona 的基本資訊
    """
    personas = {
        "empathy": {
            "name": "Lumi",
            "type": "empathy",
            "display_name": "同理型 AI",
            "description": "溫暖陪伴，情緒支持與理解",
            "specialties": ["情感支持", "同理傾聽", "情緒驗證"],
            "approach": "Person-Centered Therapy"
        },
        "insight": {
            "name": "Solin", 
            "type": "insight",
            "display_name": "洞察型 AI",
            "description": "深度探索，自我覺察與洞見",
            "specialties": ["深度探索", "模式識別", "自我覺察"],
            "approach": "Psychodynamic + Socratic Dialogue"
        },
        "solution": {
            "name": "Niko",
            "type": "solution", 
            "display_name": "解決型 AI",
            "description": "實務導向，行動規劃與目標達成",
            "specialties": ["問題解決", "目標設定", "行動規劃"],
            "approach": "Solution-Focused + CBT"
        },
        "cognitive": {
            "name": "Clara",
            "type": "cognitive",
            "display_name": "認知型 AI",
            "description": "思維重建，理性分析與認知調整", 
            "specialties": ["認知重構", "思維分析", "理性檢驗"],
            "approach": "Cognitive Behavioral Therapy"
        }
    }
    
    return {
        "personas": personas,
        "total_count": len(personas),
        "version": "professional_v1.0"
    }

@router.get("/personas/{bot_type}")
async def get_persona_info(bot_type: str):
    """
    獲取特定 persona 的詳細資訊
    """
    if bot_type not in ["empathy", "insight", "solution", "cognitive"]:
        raise HTTPException(status_code=404, detail="Persona not found")
    
    all_personas = await get_all_personas_info()
    persona = all_personas["personas"][bot_type]
    
    # 添加更詳細的資訊
    detailed_info = {
        **persona,
        "system_prompt_length": len(get_professional_system_prompt(bot_type)),
        "crisis_response_available": True,
        "supported_languages": ["繁體中文"],
        "response_style": "專業同理心導向"
    }
    
    return detailed_info

# ================= 健康檢查 =================

@router.get("/health/personas")
async def health_check_personas():
    """
    檢查 persona 系統狀態
    """
    return {
        "status": "operational",
        "personas_loaded": 4,
        "available_types": ["empathy", "insight", "solution", "cognitive"],
        "features": [
            "professional_system_prompts",
            "crisis_detection",
            "persona_specific_responses",
            "fallback_handling"
        ],
        "openai_configured": bool(os.getenv("OPENAI_API_KEY")),
        "version": "professional_v1.0"
    }