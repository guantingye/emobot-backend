# app/routers/avatar_animation.py - 頭像動畫API路由
import os
import asyncio
import base64
import tempfile
import json
import re
from typing import Optional, Dict, Any, List
from datetime import datetime

from fastapi import APIRouter, HTTPException, BackgroundTasks
from pydantic import BaseModel
import edge_tts

router = APIRouter()

# ================= Pydantic Models =================

class AvatarAnimationRequest(BaseModel):
    text: str
    bot_type: Optional[str] = "solution"
    animation_style: Optional[str] = "normal"
    voice_id: Optional[str] = None

class AvatarAnimationResponse(BaseModel):
    success: bool
    audio_base64: Optional[str] = None
    animation_data: Optional[Dict[str, Any]] = None
    duration: Optional[float] = None
    error: Optional[str] = None

# ================= 動畫風格配置 =================

ANIMATION_STYLES = {
    "empathy": {
        "name": "Lumi",
        "voice": "zh-TW-HsiaoChenNeural",
        "rate": "0.9",
        "style": {
            "mouth_intensity": 0.8,
            "blink_frequency": 0.7,
            "head_movement": 0.6,
            "emotion": "gentle"
        }
    },
    "insight": {
        "name": "Solin", 
        "voice": "zh-TW-YunJheNeural",
        "rate": "0.95",
        "style": {
            "mouth_intensity": 0.7,
            "blink_frequency": 0.5,
            "head_movement": 0.4,
            "emotion": "thoughtful"
        }
    },
    "solution": {
        "name": "Niko",
        "voice": "zh-TW-HsiaoChenNeural",
        "rate": "1.0",
        "style": {
            "mouth_intensity": 0.9,
            "blink_frequency": 0.6,
            "head_movement": 0.8,
            "emotion": "confident"
        }
    },
    "cognitive": {
        "name": "Clara",
        "voice": "zh-TW-YunJheNeural",
        "rate": "1.05",
        "style": {
            "mouth_intensity": 0.6,
            "blink_frequency": 0.4,
            "head_movement": 0.3,
            "emotion": "calm"
        }
    }
}

# ================= 語音合成與動畫生成 =================

async def generate_speech_and_animation(text: str, bot_type: str) -> Dict[str, Any]:
    """生成語音和動畫數據"""
    
    # 清理文字，移除特殊符號但保留基本標點
    clean_text = re.sub(r'[^\w\s，。！？、：；「」『』（）]', '', text)
    if not clean_text.strip():
        clean_text = "很高興和你聊天"
    
    # 獲取機器人風格配置
    style_config = ANIMATION_STYLES.get(bot_type, ANIMATION_STYLES["solution"])
    voice_id = style_config["voice"]
    rate = style_config["rate"]
    animation_style = style_config["style"]
    
    try:
        # 使用 Edge-TTS 生成語音
        communicate = edge_tts.Communicate(clean_text, voice_id, rate=f"+{float(rate)-1.0:.0%}")
        
        # 臨時檔案處理
        with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as temp_audio:
            temp_audio_path = temp_audio.name
            
        # 生成語音檔案
        await communicate.save(temp_audio_path)
        
        # 讀取音頻檔案並轉換為 base64
        with open(temp_audio_path, "rb") as audio_file:
            audio_data = audio_file.read()
            audio_base64 = base64.b64encode(audio_data).decode('utf-8')
        
        # 清理臨時檔案
        os.unlink(temp_audio_path)
        
        # 生成動畫數據
        animation_data = generate_animation_timeline(clean_text, animation_style)
        
        return {
            "success": True,
            "audio_base64": f"data:audio/mp3;base64,{audio_base64}",
            "animation_data": animation_data,
            "duration": animation_data.get("total_duration", 3.0)
        }
        
    except Exception as e:
        print(f"語音合成失敗: {e}")
        # 返回靜默動畫作為後備方案
        fallback_animation = generate_fallback_animation(clean_text, animation_style)
        return {
            "success": False,
            "audio_base64": None,
            "animation_data": fallback_animation,
            "duration": fallback_animation.get("total_duration", 3.0),
            "error": f"語音合成失敗，使用靜默模式: {str(e)}"
        }

def generate_animation_timeline(text: str, style: Dict[str, Any]) -> Dict[str, Any]:
    """根據文字內容生成動畫時間軸"""
    
    # 基本參數
    chars = len(text)
    words = len(text.split())
    base_duration = max(2.0, chars * 0.15)  # 基礎持續時間
    
    # 從風格配置中獲取參數
    mouth_intensity = style.get("mouth_intensity", 0.8)
    blink_frequency = style.get("blink_frequency", 0.6)
    head_movement = style.get("head_movement", 0.5)
    emotion = style.get("emotion", "neutral")
    
    timeline = []
    current_time = 0.0
    
    # 生成嘴部動畫（模擬音素）
    mouth_frames = []
    for i, char in enumerate(text):
        if char in '，。！？、：；':
            # 標點符號處停頓
            mouth_frames.append({
                "time": current_time,
                "mouth_openness": 0.0,
                "type": "pause"
            })
            current_time += 0.3
        elif char.strip():
            # 一般字符的嘴型變化
            openness = mouth_intensity * (0.3 + 0.4 * (i % 3) / 2)
            mouth_frames.append({
                "time": current_time,
                "mouth_openness": openness,
                "type": "phoneme"
            })
            current_time += 0.12
    
    # 生成眨眼動畫
    blink_frames = []
    blink_interval = 2.0 / max(1, blink_frequency)
    blink_time = blink_interval
    while blink_time < base_duration:
        blink_frames.extend([
            {"time": blink_time, "eye_state": "closing"},
            {"time": blink_time + 0.1, "eye_state": "closed"},
            {"time": blink_time + 0.2, "eye_state": "opening"},
            {"time": blink_time + 0.3, "eye_state": "open"}
        ])
        blink_time += blink_interval
    
    # 生成頭部微動
    head_frames = []
    if head_movement > 0.3:
        head_time = 0.0
        while head_time < base_duration:
            # 輕微的頭部搖擺
            x_offset = head_movement * 2 * (0.5 - (head_time % 4) / 4)
            y_offset = head_movement * 1 * (0.5 - (head_time % 6) / 6)
            head_frames.append({
                "time": head_time,
                "head_x": x_offset,
                "head_y": y_offset
            })
            head_time += 0.5
    
    return {
        "total_duration": max(base_duration, current_time),
        "mouth_animation": mouth_frames,
        "blink_animation": blink_frames,
        "head_animation": head_frames,
        "style_config": {
            "emotion": emotion,
            "intensity": mouth_intensity
        },
        "metadata": {
            "text_length": chars,
            "word_count": words,
            "generated_at": datetime.utcnow().isoformat()
        }
    }

def generate_fallback_animation(text: str, style: Dict[str, Any]) -> Dict[str, Any]:
    """生成後備靜默動畫（當語音合成失敗時）"""
    chars = len(text)
    duration = max(3.0, chars * 0.2)
    
    # 簡化的動畫，只有眨眼和輕微頭部動作
    return {
        "total_duration": duration,
        "mouth_animation": [
            {"time": 0.0, "mouth_openness": 0.0, "type": "silent"}
        ],
        "blink_animation": [
            {"time": 1.0, "eye_state": "closing"},
            {"time": 1.1, "eye_state": "closed"},
            {"time": 1.2, "eye_state": "opening"},
            {"time": 1.3, "eye_state": "open"},
            {"time": 3.0, "eye_state": "closing"},
            {"time": 3.1, "eye_state": "closed"},
            {"time": 3.2, "eye_state": "opening"},
            {"time": 3.3, "eye_state": "open"}
        ],
        "head_animation": [
            {"time": 0.0, "head_x": 0, "head_y": 0},
            {"time": duration, "head_x": 0, "head_y": 0}
        ],
        "style_config": {"emotion": "calm", "intensity": 0.0},
        "metadata": {"fallback_mode": True}
    }

# ================= API 端點 =================

@router.post("/animate", response_model=AvatarAnimationResponse)
async def create_avatar_animation(
    request: AvatarAnimationRequest,
    background_tasks: BackgroundTasks
):
    """為機器人頭像生成說話動畫"""
    
    if not request.text or not request.text.strip():
        raise HTTPException(status_code=400, detail="文字內容不能為空")
    
    # 限制文字長度
    text = request.text.strip()
    if len(text) > 500:
        text = text[:500] + "..."
    
    try:
        result = await generate_speech_and_animation(text, request.bot_type)
        
        return AvatarAnimationResponse(
            success=result["success"],
            audio_base64=result.get("audio_base64"),
            animation_data=result.get("animation_data"),
            duration=result.get("duration"),
            error=result.get("error")
        )
        
    except Exception as e:
        print(f"動畫生成失敗: {e}")
        raise HTTPException(status_code=500, detail=f"動畫生成失敗: {str(e)}")

@router.get("/health")
async def health_check():
    """動畫系統健康檢查"""
    try:
        # 測試 Edge-TTS 是否可用
        test_text = "測試"
        voices = await edge_tts.list_voices()
        available_voices = [v for v in voices if v['Locale'].startswith('zh-TW')]
        
        return {
            "status": "healthy",
            "edge_tts_available": True,
            "available_voices": len(available_voices),
            "supported_bots": list(ANIMATION_STYLES.keys()),
            "test_completed": True
        }
    except Exception as e:
        return {
            "status": "error",
            "error": str(e),
            "edge_tts_available": False
        }

@router.get("/styles")
async def get_animation_styles():
    """獲取可用的動畫風格"""
    return {
        "styles": ANIMATION_STYLES,
        "available_bots": list(ANIMATION_STYLES.keys())
    }