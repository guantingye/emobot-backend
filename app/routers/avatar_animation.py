# app/routers/avatar_animation.py - 完整修復版，處理Edge-TTS 403錯誤
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
import logging

# 設置日誌
logger = logging.getLogger(__name__)

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
    """生成語音和動畫數據 - 修復版，處理多種語音合成方案"""
    
    # 清理文字，移除特殊符號但保留基本標點
    clean_text = re.sub(r'[^\w\s，。！？、：；「」『』（）]', '', text)
    if not clean_text.strip():
        clean_text = "很高興和你聊天"
    
    # 獲取機器人風格配置
    style_config = ANIMATION_STYLES.get(bot_type, ANIMATION_STYLES["solution"])
    voice_id = style_config["voice"]
    rate = style_config["rate"]
    animation_style = style_config["style"]
    
    # 生成動畫數據（無論語音是否成功都要生成）
    animation_data = generate_animation_timeline(clean_text, animation_style)
    
    # 嘗試多種語音合成方案
    audio_base64 = await try_multiple_tts_providers(clean_text, voice_id, rate)
    
    return {
        "success": bool(audio_base64),
        "audio_base64": audio_base64,
        "animation_data": animation_data,
        "duration": animation_data.get("total_duration", 3.0),
        "error": None if audio_base64 else "語音合成失敗，使用靜默模式"
    }

async def try_multiple_tts_providers(text: str, voice_id: str, rate: str) -> Optional[str]:
    """嘗試多種TTS提供者"""
    
    # 方案1：Edge-TTS（第一優先）
    audio_data = await try_edge_tts(text, voice_id, rate)
    if audio_data:
        return f"data:audio/mp3;base64,{audio_data}"
    
    # 方案2：系統TTS（Windows/macOS）
    audio_data = await try_system_tts(text)
    if audio_data:
        return f"data:audio/wav;base64,{audio_data}"
    
    # 方案3：Google TTS（需要API key）
    audio_data = await try_google_tts(text, voice_id)
    if audio_data:
        return f"data:audio/mp3;base64,{audio_data}"
    
    # 所有方案都失敗
    logger.warning("所有TTS提供者都失敗")
    return None

async def try_edge_tts(text: str, voice_id: str, rate: str) -> Optional[str]:
    """嘗試Edge-TTS，處理403錯誤"""
    try:
        import edge_tts
        
        # 使用更保守的參數避免403
        communicate = edge_tts.Communicate(
            text, 
            voice_id,
            rate=f"+{max(-50, min(50, int((float(rate) - 1.0) * 100)))}%"  # 限制在-50%到+50%範圍
        )
        
        # 創建臨時文件
        with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as temp_audio:
            temp_audio_path = temp_audio.name
            
        # 添加重試邏輯
        for attempt in range(3):
            try:
                await communicate.save(temp_audio_path)
                
                # 讀取音頻文件並轉換為base64
                with open(temp_audio_path, "rb") as audio_file:
                    audio_data = audio_file.read()
                    
                # 檢查文件是否有效（大於1KB）
                if len(audio_data) > 1024:
                    return base64.b64encode(audio_data).decode('utf-8')
                else:
                    logger.warning(f"Edge-TTS生成的文件太小: {len(audio_data)} bytes")
                    
            except Exception as e:
                logger.warning(f"Edge-TTS嘗試 {attempt + 1}/3 失敗: {e}")
                if attempt < 2:
                    await asyncio.sleep(1)  # 重試前等待
                    
            finally:
                # 清理臨時文件
                try:
                    os.unlink(temp_audio_path)
                except:
                    pass
                    
        return None
        
    except ImportError:
        logger.error("edge-tts 套件未安裝")
        return None
    except Exception as e:
        logger.error(f"Edge-TTS失敗: {e}")
        return None

async def try_system_tts(text: str) -> Optional[str]:
    """嘗試系統內建TTS"""
    try:
        import subprocess
        import platform
        
        system = platform.system()
        
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_audio:
            temp_audio_path = temp_audio.name
            
        if system == "Darwin":  # macOS
            # 使用macOS的say命令
            cmd = ["say", "-v", "Mei-Jia", "-o", temp_audio_path, text]
            result = subprocess.run(cmd, capture_output=True, timeout=30)
            
        elif system == "Windows":
            # 使用Windows SAPI
            powershell_script = f'''
            Add-Type -AssemblyName System.Speech
            $synth = New-Object System.Speech.Synthesis.SpeechSynthesizer
            $synth.SetOutputToWaveFile("{temp_audio_path}")
            $synth.Speak("{text}")
            $synth.Dispose()
            '''
            result = subprocess.run(
                ["powershell", "-Command", powershell_script], 
                capture_output=True, timeout=30
            )
        else:
            # Linux - 嘗試espeak
            cmd = ["espeak", "-v", "zh", "-s", "150", "-w", temp_audio_path, text]
            result = subprocess.run(cmd, capture_output=True, timeout=30)
            
        if result.returncode == 0 and os.path.exists(temp_audio_path):
            with open(temp_audio_path, "rb") as audio_file:
                audio_data = audio_file.read()
                if len(audio_data) > 1024:
                    return base64.b64encode(audio_data).decode('utf-8')
                    
    except Exception as e:
        logger.warning(f"系統TTS失敗: {e}")
    finally:
        try:
            os.unlink(temp_audio_path)
        except:
            pass
            
    return None

async def try_google_tts(text: str, voice_id: str) -> Optional[str]:
    """嘗試Google Cloud TTS（需要API key）"""
    try:
        google_api_key = os.getenv("GOOGLE_TTS_API_KEY")
        if not google_api_key:
            return None
            
        import aiohttp
        
        # Google TTS API請求
        url = f"https://texttospeech.googleapis.com/v1/text:synthesize?key={google_api_key}"
        
        # 將Edge-TTS語音ID映射到Google語音
        voice_mapping = {
            "zh-TW-HsiaoChenNeural": "zh-TW-Wavenet-A",
            "zh-TW-YunJheNeural": "zh-TW-Wavenet-B"
        }
        google_voice = voice_mapping.get(voice_id, "zh-TW-Wavenet-A")
        
        payload = {
            "input": {"text": text},
            "voice": {
                "languageCode": "zh-TW",
                "name": google_voice
            },
            "audioConfig": {
                "audioEncoding": "MP3",
                "speakingRate": 1.0
            }
        }
        
        async with aiohttp.ClientSession() as session:
            async with session.post(url, json=payload, timeout=30) as response:
                if response.status == 200:
                    result = await response.json()
                    return result.get("audioContent")
                    
    except Exception as e:
        logger.warning(f"Google TTS失敗: {e}")
        
    return None

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
            # 標點符號停頓
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
            "generated_at": datetime.utcnow().isoformat(),
            "tts_attempted": True
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
        "metadata": {"fallback_mode": True, "no_audio": True}
    }

# ================= API 端點 =================

@router.post("/animate", response_model=AvatarAnimationResponse)
async def create_avatar_animation(
    request: AvatarAnimationRequest,
    background_tasks: BackgroundTasks
):
    """為機器人頭像生成說話動畫 - 修復版，強化錯誤處理"""
    
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
        logger.error(f"動畫生成失敗: {e}")
        
        # 生成純動畫（無語音）作為後備
        try:
            style_config = ANIMATION_STYLES.get(request.bot_type, ANIMATION_STYLES["solution"])
            fallback_animation = generate_fallback_animation(text, style_config["style"])
            
            return AvatarAnimationResponse(
                success=False,
                audio_base64=None,
                animation_data=fallback_animation,
                duration=fallback_animation.get("total_duration", 3.0),
                error=f"動畫生成失敗，使用靜默模式: {str(e)}"
            )
        except Exception as fallback_error:
            logger.error(f"後備動畫也失敗: {fallback_error}")
            raise HTTPException(status_code=500, detail=f"動畫系統暫時無法使用: {str(e)}")

@router.get("/health")
async def health_check():
    """動畫系統健康檢查 - 修復版，檢測多個TTS提供者"""
    
    status_info = {
        "status": "unknown",
        "providers": {},
        "supported_bots": list(ANIMATION_STYLES.keys()),
        "test_completed": False
    }
    
    # 測試Edge-TTS
    edge_tts_status = await test_edge_tts()
    status_info["providers"]["edge_tts"] = edge_tts_status
    
    # 測試系統TTS
    system_tts_status = await test_system_tts()
    status_info["providers"]["system_tts"] = system_tts_status
    
    # 測試Google TTS
    google_tts_status = await test_google_tts()
    status_info["providers"]["google_tts"] = google_tts_status
    
    # 計算整體狀態
    available_providers = sum(1 for p in status_info["providers"].values() if p.get("available", False))
    
    if available_providers > 0:
        status_info["status"] = "healthy"
    elif any(p.get("partially_available", False) for p in status_info["providers"].values()):
        status_info["status"] = "degraded"
    else:
        status_info["status"] = "error"
    
    status_info["available_providers"] = available_providers
    status_info["test_completed"] = True
    
    return status_info

async def test_edge_tts() -> Dict[str, Any]:
    """測試Edge-TTS可用性"""
    try:
        import edge_tts
        
        # 嘗試獲取語音列表
        voices = await edge_tts.list_voices()
        available_voices = [v for v in voices if v['Locale'].startswith('zh-TW')]
        
        # 嘗試簡單合成
        test_text = "測試"
        communicate = edge_tts.Communicate(test_text, "zh-TW-HsiaoChenNeural")
        
        with tempfile.NamedTemporaryFile(suffix=".mp3", delete=True) as temp_file:
            await asyncio.wait_for(communicate.save(temp_file.name), timeout=10)
            
            # 檢查文件大小
            file_size = os.path.getsize(temp_file.name)
            
            return {
                "available": file_size > 100,
                "available_voices": len(available_voices),
                "test_file_size": file_size,
                "error": None if file_size > 100 else "Generated file too small"
            }
            
    except asyncio.TimeoutError:
        return {
            "available": False,
            "error": "Request timeout - possible rate limiting"
        }
    except Exception as e:
        error_msg = str(e)
        is_403 = "403" in error_msg or "Invalid response status" in error_msg
        
        return {
            "available": False,
            "partially_available": not is_403,  # 非403錯誤可能是暫時性的
            "error": error_msg,
            "likely_cause": "Rate limiting or regional restrictions" if is_403 else "Network or service issue"
        }

async def test_system_tts() -> Dict[str, Any]:
    """測試系統TTS可用性"""
    try:
        import platform
        import subprocess
        
        system = platform.system()
        
        if system == "Darwin":  # macOS
            # 檢查say命令
            result = subprocess.run(["which", "say"], capture_output=True)
            available = result.returncode == 0
            
        elif system == "Windows":
            # 檢查PowerShell和SAPI
            ps_test = subprocess.run(
                ["powershell", "-Command", "Add-Type -AssemblyName System.Speech; Write-Output 'OK'"],
                capture_output=True, timeout=10
            )
            available = ps_test.returncode == 0
            
        elif system == "Linux":
            # 檢查espeak
            result = subprocess.run(["which", "espeak"], capture_output=True)
            available = result.returncode == 0
            
        else:
            available = False
            
        return {
            "available": available,
            "system": system,
            "error": None if available else f"TTS not available on {system}"
        }
        
    except Exception as e:
        return {
            "available": False,
            "error": str(e)
        }

async def test_google_tts() -> Dict[str, Any]:
    """測試Google TTS可用性"""
    try:
        google_api_key = os.getenv("GOOGLE_TTS_API_KEY")
        
        if not google_api_key:
            return {
                "available": False,
                "error": "GOOGLE_TTS_API_KEY not configured"
            }
        
        import aiohttp
        
        # 測試API連接
        url = f"https://texttospeech.googleapis.com/v1/voices?key={google_api_key}"
        
        async with aiohttp.ClientSession() as session:
            async with session.get(url, timeout=10) as response:
                if response.status == 200:
                    voices_data = await response.json()
                    zh_tw_voices = [
                        v for v in voices_data.get("voices", []) 
                        if v.get("languageCodes", [{}])[0].startswith("zh-TW")
                    ]
                    
                    return {
                        "available": True,
                        "available_voices": len(zh_tw_voices),
                        "error": None
                    }
                else:
                    return {
                        "available": False,
                        "error": f"API returned status {response.status}"
                    }
                    
    except Exception as e:
        return {
            "available": False,
            "error": str(e)
        }

@router.get("/styles")
async def get_animation_styles():
    """獲取可用的動畫風格"""
    return {
        "styles": ANIMATION_STYLES,
        "available_bots": list(ANIMATION_STYLES.keys()),
        "description": "頭像動畫風格配置，包含語音和動作參數"
    }

# ================= 背景任務：語音緩存 =================

@router.post("/preload")
async def preload_common_phrases(background_tasks: BackgroundTasks):
    """預載常用短語的語音（背景任務）"""
    
    common_phrases = [
        "你好，我是你的AI夥伴",
        "今天想聊什麼呢？",
        "我在這裡陪著你",
        "讓我們一起思考一下",
        "謝謝你和我分享"
    ]
    
    for bot_type in ANIMATION_STYLES.keys():
        for phrase in common_phrases:
            background_tasks.add_task(preload_phrase, phrase, bot_type)
    
    return {
        "message": "開始預載常用短語",
        "phrases_count": len(common_phrases),
        "bot_types": len(ANIMATION_STYLES)
    }

async def preload_phrase(text: str, bot_type: str):
    """預載單個短語（背景執行）"""
    try:
        await generate_speech_and_animation(text, bot_type)
        logger.info(f"預載完成: {bot_type} - {text[:20]}...")
    except Exception as e:
        logger.warning(f"預載失敗: {bot_type} - {text[:20]}... - {e}")

# ================= 輔助工具 =================

@router.get("/test")
async def test_animation_generation():
    """測試動畫生成系統"""
    test_text = "這是一個測試語句，用來檢查動畫生成系統是否正常運作。"
    
    results = {}
    
    for bot_type in ANIMATION_STYLES.keys():
        try:
            result = await generate_speech_and_animation(test_text, bot_type)
            results[bot_type] = {
                "success": result["success"],
                "has_audio": bool(result.get("audio_base64")),
                "has_animation": bool(result.get("animation_data")),
                "duration": result.get("duration"),
                "error": result.get("error")
            }
        except Exception as e:
            results[bot_type] = {
                "success": False,
                "error": str(e)
            }
    
    overall_success = any(r.get("success", False) for r in results.values())
    
    return {
        "overall_success": overall_success,
        "test_text": test_text,
        "results": results,
        "timestamp": datetime.utcnow().isoformat()
    }