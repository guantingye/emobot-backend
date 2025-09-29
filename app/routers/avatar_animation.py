# app/routers/avatar_animation.py - 完整修復版（OpenAI優先 + Edge/Google備援）
import os
import asyncio
import base64
import tempfile
import json
import re
from typing import Optional, Dict, Any, List, Tuple
from datetime import datetime

from fastapi import APIRouter, HTTPException, BackgroundTasks
from pydantic import BaseModel
import logging

logger = logging.getLogger(__name__)
router = APIRouter()

# ================= Pydantic Models =================

class AvatarAnimationRequest(BaseModel):
    text: str
    bot_type: Optional[str] = "solution"
    animation_style: Optional[str] = "normal"
    voice_id: Optional[str] = None  # 可覆寫預設 voice

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
        "voice": "alloy",  # OpenAI voice (primary)
        "edge_voice": "zh-TW-HsiaoChenNeural",
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
        "voice": "alloy",
        "edge_voice": "zh-TW-YunJheNeural",
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
        "voice": "alloy",
        "edge_voice": "zh-TW-HsiaoChenNeural",
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
        "voice": "alloy",
        "edge_voice": "zh-TW-YunJheNeural",
        "rate": "1.05",
        "style": {
            "mouth_intensity": 0.6,
            "blink_frequency": 0.4,
            "head_movement": 0.3,
            "emotion": "calm"
        }
    }
}

# ================== Provider Helpers ==================

def _env_bool(name: str, default: bool = False) -> bool:
    v = os.getenv(name)
    if v is None:
        return default
    return str(v).lower() in ("1", "true", "yes", "y", "on")

def _safe_b64(data: bytes, mime: str = "audio/mp3") -> str:
    return f"data:{mime};base64,{base64.b64encode(data).decode('utf-8')}"

# ------------------ OpenAI TTS (primary) ------------------

async def _openai_tts_bytes(text: str, voice: str = "alloy", fmt: str = "mp3") -> Optional[bytes]:
    """
    以 OpenAI TTS 產生音訊（bytes）。失敗回 None。
    """
    api_key = os.getenv("OPENAI_API_KEY", "").strip()
    if not api_key:
        return None

    # 允許透過環境變數覆寫
    model = os.getenv("OPENAI_TTS_MODEL", "gpt-4o-mini-tts").strip() or "gpt-4o-mini-tts"
    fmt = (fmt or os.getenv("OPENAI_TTS_FORMAT", "mp3")).lower()
    if fmt not in ("mp3", "wav", "pcm"):
        fmt = "mp3"
    voice = voice or os.getenv("OPENAI_TTS_VOICE", "alloy")

    # OpenAI 官方 SDK 為同步；用 thread 以免阻塞事件迴圈
    async def _do() -> Optional[bytes]:
        try:
            from openai import OpenAI
            client = OpenAI(api_key=api_key)
            resp = client.audio.speech.create(
                model=model,
                voice=voice,
                input=text,
                format=fmt,
            )
            # 嘗試從多種欄位取 bytes（不同 SDK 版可能差異）
            if hasattr(resp, "read"):
                return resp.read()
            content = getattr(resp, "content", None)
            if isinstance(content, (bytes, bytearray)):
                return bytes(content)

            # 迭代 chunk
            chunks = []
            for ch in resp:
                if isinstance(ch, (bytes, bytearray)):
                    chunks.append(bytes(ch))
                elif hasattr(ch, "data"):
                    chunks.append(ch.data)
            if chunks:
                return b"".join(chunks)
        except Exception as e:
            logger.warning(f"[OpenAI TTS] Exception: {e}")
        return None

    return await asyncio.to_thread(_do)

# ------------------ Edge TTS (optional fallback) ------------------

async def _edge_tts_bytes(text: str, voice: str, rate: str) -> Optional[bytes]:
    """
    使用 edge-tts 產生 MP3。Render/雲端常見 403，此為備援。
    """
    if not _env_bool("EDGE_TTS_ENABLED", False):
        return None
    try:
        import edge_tts  # type: ignore
    except Exception:
        logger.info("edge-tts 未安裝或不可用")
        return None

    # 將 rate(0.9~1.1等)轉為 +N% 並限制範圍
    try:
        pct = int((float(rate) - 1.0) * 100)
    except Exception:
        pct = 0
    pct = max(-50, min(50, pct))
    rate_str = f"+{pct}%"

    # 重試 3 次，處理 403/網路問題
    for attempt in range(3):
        try:
            communicate = edge_tts.Communicate(text, voice, rate=rate_str)
            with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as tmp:
                tmp_path = tmp.name

            try:
                await asyncio.wait_for(communicate.save(tmp_path), timeout=30)
                if os.path.exists(tmp_path) and os.path.getsize(tmp_path) > 1024:
                    with open(tmp_path, "rb") as f:
                        return f.read()
                else:
                    logger.warning(f"Edge-TTS 產出檔案太小或不存在 (attempt={attempt+1})")
            finally:
                try:
                    os.unlink(tmp_path)
                except Exception:
                    pass

        except asyncio.TimeoutError:
            logger.warning(f"Edge-TTS 超時 (attempt={attempt+1})")
        except Exception as e:
            # 常見：403 Invalid response status
            emsg = str(e)
            if "403" in emsg or "Invalid response status" in emsg:
                logger.warning(f"Edge-TTS 403/RL (attempt={attempt+1}): {emsg}")
            else:
                logger.warning(f"Edge-TTS 錯誤 (attempt={attempt+1}): {emsg}")

        await asyncio.sleep(1)

    return None

# ------------------ Google TTS (optional fallback) ------------------

async def _google_tts_b64(text: str, voice_hint: str) -> Optional[str]:
    """
    使用 Google Cloud TTS 回傳 base64（API 直接回傳base64）。
    """
    google_api_key = os.getenv("GOOGLE_TTS_API_KEY", "").strip()
    if not google_api_key:
        return None
    try:
        import aiohttp
    except Exception:
        logger.info("aiohttp 未安裝，略過 Google TTS")
        return None

    # Edge voice -> Google voice 粗略映射
    voice_map = {
        "zh-TW-HsiaoChenNeural": "zh-TW-Wavenet-A",
        "zh-TW-YunJheNeural": "zh-TW-Wavenet-B",
    }
    google_voice = voice_map.get(voice_hint, "zh-TW-Wavenet-A")

    url = f"https://texttospeech.googleapis.com/v1/text:synthesize?key={google_api_key}"
    payload = {
        "input": {"text": text},
        "voice": {"languageCode": "zh-TW", "name": google_voice},
        "audioConfig": {"audioEncoding": "MP3", "speakingRate": 1.0},
    }

    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(url, json=payload, timeout=30) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    return data.get("audioContent")
                else:
                    logger.warning(f"Google TTS 狀態碼: {resp.status}")
    except Exception as e:
        logger.warning(f"Google TTS 失敗: {e}")
    return None

# ================= 語音合成與動畫生成 =================

async def generate_speech_and_animation(text: str, bot_type: str) -> Dict[str, Any]:
    """
    生成語音與動畫：
      1) OpenAI TTS（首選）
      2) Edge-TTS（可選，EDGE_TTS_ENABLED=1）
      3) Google TTS（需 GOOGLE_TTS_API_KEY）
    """
    clean_text = re.sub(r'[^\w\s\u4e00-\u9fff，。！？、：；「」『』（）]', '', text)  # 保留中英數與常用標點
    if not clean_text.strip():
        clean_text = "很高興和你聊天"

    style_conf = ANIMATION_STYLES.get(bot_type, ANIMATION_STYLES["solution"])
    # OpenAI voice 優先（可由請求覆寫），Edge voice 僅作備援時使用
    openai_voice = style_conf.get("voice", "alloy")
    edge_voice = style_conf.get("edge_voice", "zh-TW-HsiaoChenNeural")
    rate = style_conf.get("rate", "1.0")
    anim_style = style_conf["style"]

    # 先做動畫（無論語音是否成功）
    animation_data = generate_animation_timeline(clean_text, anim_style)

    # 1) OpenAI TTS
    audio_bytes = await _openai_tts_bytes(clean_text, voice=openai_voice, fmt=os.getenv("OPENAI_TTS_FORMAT", "mp3"))
    if audio_bytes:
        animation_data.setdefault("meta", {})["provider"] = "openai"
        return {
            "success": True,
            "audio_base64": _safe_b64(audio_bytes, mime="audio/mp3"),
            "animation_data": animation_data,
            "duration": animation_data.get("total_duration", 3.0),
            "error": None
        }

    # 2) Edge-TTS（可選）
    edge_bytes = await _edge_tts_bytes(clean_text, voice=edge_voice, rate=rate)
    if edge_bytes:
        animation_data.setdefault("meta", {})["provider"] = "edge-tts"
        return {
            "success": True,
            "audio_base64": _safe_b64(edge_bytes, mime="audio/mp3"),
            "animation_data": animation_data,
            "duration": animation_data.get("total_duration", 3.0),
            "error": None
        }

    # 3) Google TTS（可選）
    google_b64 = await _google_tts_b64(clean_text, voice_hint=edge_voice)
    if google_b64:
        animation_data.setdefault("meta", {})["provider"] = "google-tts"
        return {
            "success": True,
            "audio_base64": f"data:audio/mp3;base64,{google_b64}",
            "animation_data": animation_data,
            "duration": animation_data.get("total_duration", 3.0),
            "error": None
        }

    # 全部失敗：靜默模式
    animation_data.setdefault("meta", {})["provider"] = "none"
    return {
        "success": False,
        "audio_base64": None,
        "animation_data": animation_data,
        "duration": animation_data.get("total_duration", 3.0),
        "error": "All TTS providers failed (OpenAI/Edge/Google)."
    }

# 舊函式名（供 /api/debug/avatar-test 調用）
# 注意：main.py 會從這裡 import
async def generate_speech_and_animation_old(text: str, bot_type: str) -> Dict[str, Any]:
    return await generate_speech_and_animation(text, bot_type)

# ================= API 端點 =================

@router.post("/animate", response_model=AvatarAnimationResponse)
async def create_avatar_animation(
    request: AvatarAnimationRequest,
    background_tasks: BackgroundTasks
):
    """
    生成頭像語音 + 動畫（與前端既有協定相容）
    """
    if not request.text or not request.text.strip():
        raise HTTPException(status_code=400, detail="文字內容不能為空")

    # 限制文字長度（保守限 500）
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
        try:
            style_conf = ANIMATION_STYLES.get(request.bot_type, ANIMATION_STYLES["solution"])
            fallback_animation = generate_fallback_animation(text, style_conf["style"])
            return AvatarAnimationResponse(
                success=False,
                audio_base64=None,
                animation_data=fallback_animation,
                duration=fallback_animation.get("total_duration", 3.0),
                error=f"動畫生成失敗，使用靜默模式: {str(e)}"
            )
        except Exception as fe:
            logger.error(f"後備動畫也失敗: {fe}")
            raise HTTPException(status_code=500, detail="動畫系統暫時無法使用")

@router.get("/health")
async def health_check():
    """
    動畫/TTS 提供者健康檢查
      - openai: 僅檢查環境變數是否存在（避免產生成本）
      - edge-tts: 檢查套件 & EDGE_TTS_ENABLED
      - google-tts: 嘗試列出 voices（需要 GOOGLE_TTS_API_KEY）
    """
    info: Dict[str, Any] = {
        "status": "unknown",
        "providers": {},
        "supported_bots": list(ANIMATION_STYLES.keys()),
        "test_completed": True
    }

    # OpenAI
    openai_ok = bool(os.getenv("OPENAI_API_KEY"))
    info["providers"]["openai"] = {"available": openai_ok, "error": None if openai_ok else "OPENAI_API_KEY not set"}

    # Edge
    try:
        import edge_tts  # type: ignore
        edge_enabled = _env_bool("EDGE_TTS_ENABLED", False)
        info["providers"]["edge_tts"] = {
            "available": edge_enabled,
            "error": None if edge_enabled else "EDGE_TTS_ENABLED=1 以啟用（雲端可能403）"
        }
    except Exception:
        info["providers"]["edge_tts"] = {"available": False, "error": "edge-tts not installed"}

    # Google
    google_key = os.getenv("GOOGLE_TTS_API_KEY")
    if google_key:
        # 輕量探測：列舉 voices
        try:
            import aiohttp  # type: ignore
            url = f"https://texttospeech.googleapis.com/v1/voices?key={google_key}"
            async with aiohttp.ClientSession() as session:
                async with session.get(url, timeout=10) as resp:
                    info["providers"]["google_tts"] = {
                        "available": resp.status == 200,
                        "status_code": resp.status,
                        "error": None if resp.status == 200 else f"status={resp.status}"
                    }
        except Exception as e:
            info["providers"]["google_tts"] = {"available": False, "error": str(e)}
    else:
        info["providers"]["google_tts"] = {"available": False, "error": "GOOGLE_TTS_API_KEY not set"}

    # 總結
    if any(p.get("available") for p in info["providers"].values()):
        info["status"] = "healthy"
    else:
        info["status"] = "error"

    return info

@router.get("/styles")
async def get_animation_styles():
    """獲取可用的動畫風格"""
    return {
        "styles": ANIMATION_STYLES,
        "available_bots": list(ANIMATION_STYLES.keys()),
        "description": "頭像動畫風格配置，包含語音與動作參數"
    }

@router.post("/preload")
async def preload_common_phrases(background_tasks: BackgroundTasks):
    """
    預載常用短語（背景任務）
    """
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
    return {"message": "開始預載常用短語", "phrases_count": len(common_phrases), "bot_types": len(ANIMATION_STYLES)}

async def preload_phrase(text: str, bot_type: str):
    try:
        await generate_speech_and_animation(text, bot_type)
        logger.info(f"預載完成: {bot_type} - {text[:20]}...")
    except Exception as e:
        logger.warning(f"預載失敗: {bot_type} - {text[:20]}... - {e}")

@router.get("/test")
async def test_animation_generation():
    """
    輕量自測：每個 bot 走一遍流程（注意：若開啟 OpenAI 會產生成本）
    """
    test_text = "這是一個測試語句，用來檢查動畫生成系統是否正常運作。"
    results: Dict[str, Any] = {}
    for bot_type in ANIMATION_STYLES.keys():
        try:
            result = await generate_speech_and_animation(test_text, bot_type)
            results[bot_type] = {
                "success": result["success"],
                "has_audio": bool(result.get("audio_base64")),
                "has_animation": bool(result.get("animation_data")),
                "provider": (result.get("animation_data", {}) or {}).get("meta", {}).get("provider"),
                "duration": result.get("duration"),
                "error": result.get("error")
            }
        except Exception as e:
            results[bot_type] = {"success": False, "error": str(e)}
    overall_success = any(r.get("success", False) for r in results.values())
    return {"overall_success": overall_success, "test_text": test_text, "results": results, "timestamp": datetime.utcnow().isoformat()}

# ================= 動畫生成 =================

def generate_animation_timeline(text: str, style: Dict[str, Any]) -> Dict[str, Any]:
    """
    根據文字生成嘴型、眨眼與頭部微動時間軸（與前端相容）
    """
    chars = len(text)
    words = len(text.split())
    base_duration = max(2.0, min(12.0, chars * 0.12))  # 上限12s避免過長

    mouth_intensity = style.get("mouth_intensity", 0.8)
    blink_frequency = style.get("blink_frequency", 0.6)
    head_movement = style.get("head_movement", 0.5)
    emotion = style.get("emotion", "neutral")

    timeline_mouth = []
    t = 0.0
    for i, ch in enumerate(text):
        if ch in '，。！？、：；':
            timeline_mouth.append({"time": t, "mouth_openness": 0.0, "type": "pause"})
            t += 0.28
        elif ch.strip():
            openness = round(mouth_intensity * (0.35 + 0.35 * ((i % 3) / 2.0)), 3)
            timeline_mouth.append({"time": t, "mouth_openness": openness, "type": "phoneme"})
            t += 0.11
        if t > base_duration:
            break

    # 眨眼
    timeline_blink = []
    if blink_frequency > 0:
        interval = max(1.6, 2.4 / blink_frequency)
        bt = interval
        while bt < base_duration:
            timeline_blink.extend([
                {"time": round(bt, 2), "eye_state": "closing"},
                {"time": round(bt + 0.08, 2), "eye_state": "closed"},
                {"time": round(bt + 0.16, 2), "eye_state": "opening"},
                {"time": round(bt + 0.24, 2), "eye_state": "open"},
            ])
            bt += interval

    # 頭部微動
    timeline_head = []
    if head_movement > 0.2:
        ht = 0.0
        while ht < base_duration:
            x = round(head_movement * 2 * (0.5 - ((ht % 4) / 4)), 3)
            y = round(head_movement * 1 * (0.5 - ((ht % 6) / 6)), 3)
            timeline_head.append({"time": round(ht, 2), "head_x": x, "head_y": y})
            ht += 0.5

    return {
        "total_duration": max(base_duration, t),
        "mouth_animation": timeline_mouth,
        "blink_animation": timeline_blink,
        "head_animation": timeline_head,
        "style_config": {"emotion": emotion, "intensity": mouth_intensity},
        "meta": {"provider": "pending"},
        "metadata": {
            "text_length": chars,
            "word_count": words,
            "generated_at": datetime.utcnow().isoformat(),
            "tts_attempted": True
        }
    }

def generate_fallback_animation(text: str, style: Dict[str, Any]) -> Dict[str, Any]:
    """語音失敗時的靜默動畫"""
    chars = len(text)
    duration = max(3.0, min(10.0, chars * 0.15))
    return {
        "total_duration": duration,
        "mouth_animation": [{"time": 0.0, "mouth_openness": 0.0, "type": "silent"}],
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
        "meta": {"provider": "none"},
        "metadata": {"fallback_mode": True, "no_audio": True}
    }
