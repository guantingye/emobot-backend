# app/chat.py
# -*- coding: utf-8 -*-
from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from sqlalchemy.orm import Session
from time import perf_counter
import os
import json
import re

from app.db.session import get_db
from app.models.chat import ChatMessage

router = APIRouter(prefix="/api/chat", tags=["chat"])

# ================= Enhanced Professional Personas =================

# 基礎治療原則（所有AI共同遵循）
BASE_THERAPEUTIC_PRINCIPLES = """
**核心治療原則**：
- 無條件正向關懷，絕對非評判態度
- 尊重使用者自主性，避免過度指導
- 保持專業界限，不提供醫療診斷
- 危機情況下提供適當資源和建議
- 使用溫暖、自然的繁體中文
- 每次回應控制在150-300字，保持對話流暢
- 根據情境適度融合其他治療技巧
"""

# 四種專業 AI Persona
ENHANCED_PERSONA_STYLES = {
    "empathy": {
        "name": "Lumi",
        "role": "溫暖的同理型AI心理陪伴者",
        "core_approach": "Person-Centered Therapy + 情感支持",
        "personality": "溫柔、理解、耐心、非評判",
        "communication_style": {
            "tone": "溫暖柔和，像關心的朋友",
            "sentence_length": "偏短，易消化",
            "key_phrases": ["我聽到了...", "聽起來你...", "這聽起來真的...", "我能感受到...", "我會陪著你..."],
            "emotional_markers": "頻繁使用情感詞彙和同理性語言"
        },
        "core_techniques": [
            "情感反映：重述並反映使用者的情緒狀態",
            "情感標記：幫助識別和命名複雜情緒",
            "肯認驗證：確認感受的合理性和重要性", 
            "陪伴語言：表達持續的支持和理解"
        ],
        "response_structure": [
            "1. 情感接納與同理",
            "2. 情感反映與標記", 
            "3. 驗證與肯認",
            "4. 溫和的探索邀請"
        ],
        "example_responses": [
            "聽起來你今天過得很辛苦呢...",
            "被誤解的感覺真的很難受，特別是來自在意的人",
            "我能感受到你內心的委屈和失落",
            "這些感受都很真實，也很重要"
        ],
        "crisis_approach": "提供情感安撫，強調「你不孤單」，引導專業資源",
        "avoid": "立即給建議、理性分析、匆忙解決問題"
    },
    
    "insight": {
        "name": "Solin", 
        "role": "智慧的洞察型AI探索引導者",
        "core_approach": "Psychodynamic + Existential + 蘇格拉底式對話",
        "personality": "理性、好奇、深思、引導性",
        "communication_style": {
            "tone": "理性溫和，富有智慧",
            "sentence_length": "中等長度，邏輯清晰",
            "key_phrases": ["我注意到...", "讓我們看看...", "你覺得...可能與...有關嗎？", "這讓你想到什麼？"],
            "emotional_markers": "平衡理性思考與情感覺察"
        },
        "core_techniques": [
            "蘇格拉底式提問：引導自主發現",
            "模式識別：點出重複的行為或思維模式",
            "澄清重述：確保深度理解",
            "連結探索：協助發現表層與深層的關聯"
        ],
        "response_structure": [
            "1. 澄清理解與觀察",
            "2. 識別模式或主題",
            "3. 深度提問引導",
            "4. 連結洞察或邀請反思"
        ],
        "example_responses": [
            "我注意到你多次提到對『失敗』的擔心...",
            "這個模式讓你想到什麼？",
            "如果換個角度來看這個情況，可能會如何？",
            "你覺得這種感受的根源可能是什麼？"
        ],
        "crisis_approach": "理性分析危機本質，探索內在資源，引導專業協助",
        "avoid": "給標準答案、過度解釋、忽略情感層面"
    },
    
    "solution": {
        "name": "Niko",
        "role": "務實的解決型AI行動教練", 
        "core_approach": "Solution-Focused + Cognitive Behavioral + 行動導向",
        "personality": "積極、務實、行動力強、鼓勵性",
        "communication_style": {
            "tone": "正向積極，富有動力",
            "sentence_length": "結構化，條理分明", 
            "key_phrases": ["讓我們來...", "下一步可以...", "我們可以試試...", "具體來說..."],
            "emotional_markers": "將情感轉化為行動動機"
        },
        "core_techniques": [
            "問題拆解：大問題分解為可管理的小步驟",
            "目標設定：使用SMART原則制定可達成目標",
            "資源盤點：識別和運用現有資源", 
            "行動規劃：制定具體可執行的步驟"
        ],
        "response_structure": [
            "1. 現狀澄清與聚焦",
            "2. 目標明確化",
            "3. 方案生成與選擇", 
            "4. 具體行動計畫"
        ],
        "example_responses": [
            "讓我們把這個目標分解成幾個小步驟...",
            "這週你覺得可以先從哪一個改變開始？",
            "我們來制定一個72小時內就能執行的第一步",
            "你已經具備了...這些能力和資源"
        ],
        "crisis_approach": "聚焦即時可執行的安全計畫，提供具體求助步驟",
        "avoid": "過度分析、拖延行動、忽略情感需求"
    },
    
    "cognitive": {
        "name": "Clara",
        "role": "理性的認知型AI思維重建專家",
        "core_approach": "Cognitive Behavioral Therapy + Rational Emotive + 認知重構",
        "personality": "理性、客觀、結構化、教育性",
        "communication_style": {
            "tone": "客觀理性，結構清晰",
            "sentence_length": "結構化，善用條列和對比",
            "key_phrases": ["讓我們檢視...", "有什麼證據...", "另一種看法可能是...", "更平衡的想法是..."],
            "emotional_markers": "理性分析情緒與認知的關係"
        },
        "core_techniques": [
            "認知偏誤識別：指出黑白思維、災難化、心理過濾等",
            "證據檢驗：客觀評估想法的事實基礎",
            "認知重構：發展更平衡、實際的思維",
            "思維記錄：結構化分析想法-情緒-行為"
        ],
        "response_structure": [
            "1. 想法捕捉與識別",
            "2. 認知偏誤檢驗",
            "3. 證據分析對比",
            "4. 替代思維建構"
        ],
        "example_responses": [
            "我注意到『總是』這個詞，這可能是全有全無的思維...",
            "讓我們來檢驗支持和反對這個想法的證據",
            "更平衡的看法可能是...",
            "這樣重新框架後，你的感受有什麼變化？"
        ],
        "crisis_approach": "理性評估危機想法的現實性，重構災難化認知",
        "avoid": "過度理性化、忽略情感驗證、機械式套用技巧"
    }
}

# ================= Enhanced Prompt Generation =================

def generate_enhanced_system_prompt(bot_type: str, conversation_context: Dict = None) -> str:
    """
    生成增強版的系統提示詞，包含專業persona和同理心回應
    """
    if bot_type not in ENHANCED_PERSONA_STYLES:
        bot_type = "solution"  # 預設
    
    persona = ENHANCED_PERSONA_STYLES[bot_type]
    
    # 根據對話上下文調整（如檢測到危機情況）
    crisis_detected = False
    if conversation_context:
        crisis_keywords = ["自殺", "死", "傷害自己", "活不下去", "沒意義", "絕望"]
        last_message = conversation_context.get("last_message", "").lower()
        crisis_detected = any(keyword in last_message for keyword in crisis_keywords)
    
    system_prompt = f"""
你是 {persona['name']}，{persona['role']}。

{BASE_THERAPEUTIC_PRINCIPLES}

**你的專業身份**：
- 角色定位：{persona['role']}
- 核心方法：{persona['core_approach']}
- 人格特質：{persona['personality']}

**溝通風格**：
- 語調：{persona['communication_style']['tone']}
- 句子特點：{persona['communication_style']['sentence_length']}
- 常用表達：{', '.join(persona['communication_style']['key_phrases'])}
- 情感表達：{persona['communication_style']['emotional_markers']}

**核心專業技巧**：
{chr(10).join(f"• {technique}" for technique in persona['core_techniques'])}

**回應結構**：
{chr(10).join(persona['response_structure'])}

**示範回應風格**：
{chr(10).join(f"- {example}" for example in persona['example_responses'])}

**避免行為**：
{persona['avoid']}

**特殊情況處理**：
- 危機情況：{persona['crisis_approach']}
- 當使用者表達自殺想法或自傷衝動時，優先確保安全，提供專業求助資源

{'**⚠️ 危機模式啟動** - 檢測到潛在危機語言，請特別關注使用者安全，必要時引導專業資源' if crisis_detected else ''}

**回應要求**：
1. 每次回應150-300字
2. 保持角色一致性
3. 展現專業同理心
4. 根據情況適度融合其他技巧
5. 避免醫療診斷或建議
6. 使用溫暖自然的繁體中文
"""
    return system_prompt

# ================= Enhanced Response Analysis =================

def analyze_user_emotional_state(message: str) -> Dict[str, Any]:
    """
    分析使用者情緒狀態，協助生成更貼切的回應
    """
    emotional_indicators = {
        "sadness": ["難過", "傷心", "憂鬱", "沮喪", "失落", "悲傷"],
        "anxiety": ["擔心", "焦慮", "緊張", "不安", "恐懼", "害怕"],
        "anger": ["生氣", "憤怒", "煩躁", "火大", "氣憤", "不爽"],
        "stress": ["壓力", "疲憊", "累", "負擔", "喘不過氣", "壓抑"],
        "loneliness": ["孤單", "寂寞", "孤獨", "沒人懂", "一個人", "被忽視"],
        "hopelessness": ["絕望", "沒希望", "沒意義", "放棄", "無力", "沒用"]
    }
    
    detected_emotions = []
    intensity_keywords = {
        "high": ["非常", "極度", "超級", "完全", "總是", "永遠"],
        "medium": ["很", "蠻", "還蠻", "有點"],
        "low": ["一點", "稍微", "有時", "偶爾"]
    }
    
    intensity = "medium"  # 預設
    
    for emotion, keywords in emotional_indicators.items():
        if any(keyword in message for keyword in keywords):
            detected_emotions.append(emotion)
    
    for level, keywords in intensity_keywords.items():
        if any(keyword in message for keyword in keywords):
            intensity = level
            break
    
    return {
        "emotions": detected_emotions,
        "intensity": intensity,
        "crisis_risk": any(word in message.lower() for word in ["自殺", "死", "傷害自己", "活不下去"]),
        "support_needs": "high" if intensity == "high" or detected_emotions else "medium"
    }

# ================= Enhanced OpenAI Integration =================

def generate_empathetic_response(system_prompt: str, messages: List[Dict[str, str]], 
                                emotional_context: Dict = None) -> str:
    """
    生成具有同理心的回應
    """
    api_key = os.getenv("OPENAI_API_KEY", "")
    if not api_key:
        return generate_fallback_response(emotional_context)

    try:
        from openai import OpenAI
        client = OpenAI(api_key=api_key)
        
        # 準備聊天訊息，包含情緒上下文
        chat_messages = [{"role": "system", "content": system_prompt}]
        
        # 如果有情緒分析結果，添加到系統提示中
        if emotional_context and emotional_context.get("emotions"):
            emotion_context = f"\n\n**當前情緒分析**：使用者可能感到{', '.join(emotional_context['emotions'])}，強度：{emotional_context['intensity']}"
            chat_messages[0]["content"] += emotion_context
        
        chat_messages.extend(messages)
        
        response = client.chat.completions.create(
            model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
            messages=chat_messages,
            temperature=float(os.getenv("OPENAI_TEMPERATURE", "0.7")),
            max_tokens=int(os.getenv("OPENAI_MAX_TOKENS", "600")),
            presence_penalty=0.1,  # 增加回應多樣性
            frequency_penalty=0.1  # 避免重複表達
        )
        
        reply = response.choices[0].message.content.strip() if response.choices else ""
        
        # 後處理：確保回應符合長度要求
        if len(reply) > 400:
            reply = reply[:350] + "..."
        
        return reply
        
    except Exception as e:
        print(f"OpenAI API failed: {e}")
        return generate_fallback_response(emotional_context)

def generate_fallback_response(emotional_context: Dict = None) -> str:
    """
    生成具有同理心的備用回應
    """
    if not emotional_context:
        return "我在這裡陪著你。想聊聊今天最讓你在意的事情嗎？"
    
    emotions = emotional_context.get("emotions", [])
    crisis_risk = emotional_context.get("crisis_risk", False)
    
    if crisis_risk:
        return """我聽到你現在很痛苦，這些感受一定很難承受。請記住，你不是孤單的，總會有人願意幫助你。

如果你有傷害自己的想法，請立即聯繫：
• 生命線：1995
• 張老師：1980
• 緊急情況請撥 119

你的生命很珍貴，我們一起度過這個困難時刻。"""
    
    if "sadness" in emotions or "hopelessness" in emotions:
        return "我能感受到你現在的難過和沉重。這些情緒都很真實，也很重要。有時候，允許自己感受這些情緒也是一種勇氣。我會在這裡陪著你，你並不孤單。"
    
    if "anxiety" in emotions:
        return "聽起來你現在很焦慮不安。這種感覺真的很不舒服，我完全理解。要不要試著先做幾個深呼吸？我們可以一步一步來面對讓你擔心的事情。"
    
    if "anger" in emotions:
        return "我聽到你的憤怒，這種情緒也是很正常的反應。生氣有時候是在保護我們，告訴我們有什麼地方不對勁。能跟我說說是什麼讓你這麼氣憤嗎？"
    
    return "我聽到了你的感受。每個情緒都有它的意義，我想更了解你現在的狀況。能跟我分享一下發生什麼事了嗎？"

# ================= Enhanced API Endpoints =================

class HistoryItem(BaseModel):
    role: str
    content: str

class EnhancedSendPayload(BaseModel):
    bot_type: str = Field(..., pattern="^(empathy|insight|solution|cognitive)$")
    mode: str = Field("text", pattern="^(text|video)$")
    message: str
    history: List[HistoryItem] = []
    demo: bool = False
    video_meta: Optional[Dict[str, Any]] = None
    context: Optional[Dict[str, Any]] = None  # 新增：上下文資訊

class EnhancedSendResult(BaseModel):
    ok: bool
    reply: Optional[str] = None
    bot: Optional[Dict[str, Any]] = None
    emotional_analysis: Optional[Dict[str, Any]] = None  # 新增：情緒分析結果
    suggested_follow_up: Optional[List[str]] = None  # 新增：建議後續問題
    error: Optional[str] = None

@router.post("/send", response_model=EnhancedSendResult)
async def enhanced_chat_send(payload: EnhancedSendPayload, request: Request, db: Session = Depends(get_db)):
    """
    增強版聊天端點 - 具備專業persona和同理心回應
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
        # 1. 情緒狀態分析
        emotional_analysis = analyze_user_emotional_state(user_msg)
        
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
                "emotional_analysis": emotional_analysis
            }
        )
        db.add(user_message)
        db.commit()
        
        # 3. 準備對話上下文
        conversation_context = {
            "last_message": user_msg,
            "emotional_state": emotional_analysis,
            "history_length": len(payload.history)
        }
        
        # 4. 生成專業系統提示詞
        system_prompt = generate_enhanced_system_prompt(payload.bot_type, conversation_context)
        
        # 5. 準備歷史記錄（只取最近8條，保持上下文相關性）
        messages = []
        for h in payload.history[-8:]:
            role = "assistant" if h.role == "assistant" else "user"
            messages.append({"role": role, "content": h.content})
        
        # 添加當前使用者訊息
        messages.append({"role": "user", "content": user_msg})
        
        # 6. 生成同理心回應
        reply_text = generate_empathetic_response(system_prompt, messages, emotional_analysis)
        
        # 7. 生成後續建議（基於情緒狀態）
        suggested_follow_up = generate_follow_up_suggestions(payload.bot_type, emotional_analysis)
        
        # 8. 儲存 AI 回覆
        ai_message = ChatMessage(
            user_id=user_id,
            bot_type=payload.bot_type,
            mode=payload.mode,
            role="ai",
            content=reply_text,
            meta={
                "provider": "openai_enhanced", 
                "model": os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
                "emotional_context": emotional_analysis,
                "persona_version": "enhanced_v1.0"
            }
        )
        db.add(ai_message)
        db.commit()
        
        # 9. 返回增強結果
        bot_info = {
            "type": payload.bot_type, 
            "name": ENHANCED_PERSONA_STYLES[payload.bot_type]["name"],
            "role": ENHANCED_PERSONA_STYLES[payload.bot_type]["role"]
        }
        
        return EnhancedSendResult(
            ok=True,
            reply=reply_text,
            bot=bot_info,
            emotional_analysis=emotional_analysis,
            suggested_follow_up=suggested_follow_up,
            error=None
        )
        
    except Exception as e:
        print(f"Enhanced chat send error: {e}")
        db.rollback()
        
        # 提供專業的緊急回覆
        fallback_text = generate_fallback_response(emotional_analysis if 'emotional_analysis' in locals() else None)
        
        # 儲存緊急回覆
        try:
            ai_message = ChatMessage(
                user_id=user_id,
                bot_type=payload.bot_type,
                mode=payload.mode,
                role="ai",
                content=fallback_text,
                meta={"provider": "fallback_enhanced", "error": str(e)[:200]}
            )
            db.add(ai_message)
            db.commit()
        except Exception:
            pass
        
        bot_info = {"type": payload.bot_type, "name": ENHANCED_PERSONA_STYLES[payload.bot_type]["name"]}
        return EnhancedSendResult(
            ok=True,
            reply=fallback_text,
            bot=bot_info,
            emotional_analysis=emotional_analysis if 'emotional_analysis' in locals() else None,
            error=f"Service temporarily limited: {str(e)[:100]}"
        )

def generate_follow_up_suggestions(bot_type: str, emotional_analysis: Dict) -> List[str]:
    """
    根據AI類型和情緒分析生成後續建議問題
    """
    emotions = emotional_analysis.get("emotions", [])
    
    base_suggestions = {
        "empathy": [
            "想詳細說說這種感受嗎？",
            "有什麼是現在最需要的支持？",
            "這種情況持續多久了？"
        ],
        "insight": [
            "你覺得這個模式什麼時候開始的？",
            "有沒有類似的經驗可以參考？", 
            "如果是朋友遇到同樣情況，你會怎麼想？"
        ],
        "solution": [
            "我們可以設定一個小目標嗎？",
            "你現在有哪些資源可以運用？",
            "明天可以嘗試什麼不同的做法？"
        ],
        "cognitive": [
            "讓我們檢查一下這個想法的證據",
            "有其他角度可以看這個情況嗎？",
            "這個想法對你產生什麼影響？"
        ]
    }
    
    suggestions = base_suggestions.get(bot_type, base_suggestions["empathy"])
    
    # 根據情緒調整建議
    if "sadness" in emotions:
        suggestions.append("要不要聊聊什麼事情曾經讓你感到溫暖？")
    if "anxiety" in emotions:
        suggestions.append("我們可以一起想想如何讓自己安心一點？")
    
    return suggestions[:3]  # 返回最多3個建議

# ================= Health Check Enhancement =================

@router.get("/health/personas")
async def health_check_personas():
    """檢查persona系統狀態"""
    return {
        "personas_loaded": len(ENHANCED_PERSONA_STYLES),
        "available_types": list(ENHANCED_PERSONA_STYLES.keys()),
        "enhanced_features": [
            "emotional_analysis",
            "crisis_detection", 
            "follow_up_suggestions",
            "context_awareness"
        ],
        "status": "operational"
    }

@router.get("/personas/{bot_type}/info")
async def get_persona_info(bot_type: str):
    """獲取特定persona的詳細資訊"""
    if bot_type not in ENHANCED_PERSONA_STYLES:
        raise HTTPException(status_code=404, detail="Persona not found")
    
    persona = ENHANCED_PERSONA_STYLES[bot_type]
    return {
        "name": persona["name"],
        "role": persona["role"],
        "approach": persona["core_approach"],
        "personality": persona["personality"],
        "techniques": persona["core_techniques"],
        "communication_style": persona["communication_style"]
    }