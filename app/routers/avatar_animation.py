# app/routers/avatar_animation.py - 強化版（OpenAI優先 + HTTP備援 + 詳細診斷）
import os
import asyncio
import base64
import tempfile
import re
from typing import Optional, Dict, Any, Tuple
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
    "empathy":  {"name": "Lumi",   "voice": "alloy", "edge_voice": "zh-TW-HsiaoChenNeural", "rate": "0.9",
                 "style": {"mouth_intensity": 0.8, "blink_frequency": 0.7, "head_movement": 0.6, "emotion": "gentle"}},
    "insight":  {"name": "Solin",  "voice": "alloy", "edge_voice": "zh-TW-YunJheNeural",    "rate": "0.95",
                 "style": {"mouth_intensity": 0.7, "blink_frequency": 0.5, "head_movement": 0.4, "emotion": "thoughtful"}},
    "solution": {"name": "Niko",   "voice": "alloy", "edge_voice": "zh-TW-HsiaoChenNeural", "rate": "1.0",
                 "style": {"mouth_intensity": 0.9, "blink_frequency": 0.6, "head_movement": 0.8, "emotion": "confident"}},
    "cognitive":{"name": "Clara",  "voice": "alloy", "edge_voice": "zh-TW-YunJheNeural",    "rate": "1.05",
                 "style": {"mouth_intensity": 0.6, "blink_frequency": 0.4, "head_movement": 0.3, "emotion": "calm"}},
}

# ================== 小工具 ==================

def _env_bool(name: str, default: bool = False) -> bool:
    v = os.getenv(name)
    if v is None:
        return default
    return str(v).lower() in ("1", "true", "yes", "y", "on")

def _safe_b64(data: bytes, mime: str = "audio/mp3") -> str:
    return f"data:{mime};base64,{base64.b64encode(data).decode('utf-8')}"

# ================== OpenAI TTS（主用，雙路徑） ==================

async def _openai_tts_bytes(text: str, voice: str = "alloy", fmt: str = "mp3") -> Tuple[Optional[bytes], Optional[str]]:
    """
    以 OpenAI TTS 產生音訊（bytes）。失敗回 (None, error_msg)。
    1) 嘗試 SDK streaming_response
    2) 失敗則用 aiohttp 直呼 REST API（/v1/audio/speech）
    """
    api_key = os.getenv("OPENAI_API_KEY", "").strip()
    if not api_key:
        return None, "OPENAI_API_KEY not set"

    model = os.getenv("OPENAI_TTS_MODEL", "gpt-4o-mini-tts").strip() or "gpt-4o-mini-tts"
    fmt = (fmt or os.getenv("OPENAI_TTS_FORMAT", "mp3")).lower()
    if fmt not in ("mp3", "wav", "pcm"):
        fmt = "mp3"
    voice = voice or os.getenv("OPENAI_TTS_VOICE", "alloy")
    base_url = os.getenv("OPENAI_API_BASE", "https://api.openai.com").rstrip("/")

    last_err = None

    # 1) 官方 SDK with_streaming_response
    def _do_sdk() -> Tuple[Optional[bytes], Optional[str]]:
        try:
            from openai import OpenAI
            client = OpenAI(api_key=api_key, base_url=base_url)

            with tempfile.NamedTemporaryFile(suffix=f".{fmt}", delete=False) as tmp:
                tmp_path = tmp.name

            try:
                with client.audio.speech.with_streaming_response.create(
                    model=model,
                    voice=voice,
                    input=text,
                    format=fmt,
                ) as resp:
                    resp.stream_to_file(tmp_path)

                if os.path.exists(tmp_path) and os.path.getsize(tmp_path) > 1024:
                    with open(tmp_path, "rb") as f:
                        return f.read(), None
                return None, "OpenAI SDK produced empty/too-small file"
            finally:
                try:
                    os.unlink(tmp_path)
                except Exception:
                    pass

        except Exception as e:
            return None, f"OpenAI SDK error: {e}"

    data, err = await asyncio.to_thread(_do_sdk)
    if data:
        return data, None
    last_err = err

    # 2) 直接 HTTP（aiohttp）
    try:
        import aiohttp
        url = f"{base_url}/v1/audio/speech"
        payload = {"model": model, "voice": voice, "input": text, "format": fmt}
        headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
        timeout = aiohttp.ClientTimeout(total=60)

        async with aiohttp.ClientSession(timeout=timeout) as session:
            async with session.post(url, json=payload, headers=headers) as resp:
                if resp.status == 200:
                    bin_data = await resp.read()
                    if bin_data and len(bin_data) > 1024:
                        return bin_data, None
                    return None, "OpenAI HTTP returned too small/empty body"
                else:
                    body = await resp.text()
                    return None, f"OpenAI HTTP {resp.status}: {body[:200]}"
    except Exception as e:
        last_err = f"OpenAI HTTP error: {e}"

    return None, last_err or "OpenAI TTS failed"

# ================== Edge TTS（可選備援） ==================

async def _edge_tts_bytes(text: str, voice: str, rate: str) -> Tuple[Optional[bytes], Optional[str]]:
    if not _env_bool("EDGE_TTS_ENABLED", False):
        return None, "EDGE_TTS not enabled"
    try:
        import edge_tts  # type: ignore
    except Exception:
        return None, "edge-tts not installed"

    try:
        pct = int((float(rate) - 1.0) * 100)
    except Exception:
        pct = 0
    pct = max(-50, min(50, pct))
    rate_str = f"+{pct}%"

    last_err = None
    for attempt in range(3):
        try:
            communicate = edge_tts.Communicate(text, voice, rate=rate_str)
            with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as tmp:
                tmp_path = tmp.name

            try:
                await asyncio.wait_for(communicate.save(tmp_path), timeout=30)
                if os.path.exists(tmp_path) and os.path.getsize(tmp_path) > 1024:
                    with open(tmp_path, "rb") as f:
                        return f.read(), None
                last_err = "Edge produced empty/too-small file"
            finally:
                try:
                    os.unlink(tmp_path)
                except Exception:
                    pass

        except asyncio.TimeoutError:
            last_err = "Edge timeout (rate limit or network)"
        except Exception as e:
            emsg = str(e)
            if "403" in emsg or "Invalid response status" in emsg:
                last_err = f"Edge 403/RL: {emsg}"
            else:
                last_err = f"Edge error: {emsg}"
        await asyncio.sleep(1)

    return None, last_err or "Edge TTS failed"

# ================== Google TTS（可選備援） ==================

async def _google_tts_b64(text: str, voice_hint: str) -> Tuple[Optional[str], Optional[str]]:
    google_api_key = os.getenv("GOOGLE_TTS_API_KEY", "").strip()
    if not google_api_key:
        return None, "GOOGLE_TTS_API_KEY not set"
    try:
        import aiohttp
    except Exception:
        return None, "aiohttp not installed"

    voice_map = {"zh-TW-HsiaoChenNeural": "zh-TW-Wavenet-A", "zh-TW-YunJheNeural": "zh-TW-Wavenet-B"}
    google_voice = voice_map.get(voice_hint, "zh-TW-Wavenet-A")

    url = f"https://texttospeech.googleapis.com/v1/text:synthesize?key={google_api_key}"
    payload = {
        "input": {"text": text},
        "voice": {"languageCode": "zh-TW", "name": google_voice},
        "audioConfig": {"audioEncoding": "MP3", "speakingRate": 1.0},
    }

    try:
        timeout = aiohttp.ClientTimeout(total=30)
        async with aiohttp.ClientSession(timeout=timeout) as session:
            async with session.post(url, json=payload) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    b64 = data.get("audioContent")
                    if b64:
                        return b64, None
                    return None, "Google returned empty audioContent"
                else:
                    body = await resp.text()
                    return None, f"Google HTTP {resp.status}: {body[:200]}"
    except Exception as e:
        return None, f"Google error: {e}"

# ================= 語音合成與動畫生成 =================

async def generate_speech_and_animation(text: str, bot_type: str) -> Dict[str, Any]:
    """
    生成語音與動畫：
      1) OpenAI TTS（首選：SDK → HTTP）
      2) Edge-TTS（可選）
      3) Google TTS（可選）
    """
    debug = _env_bool("DEBUG_TTS", False)

    clean_text = re.sub(r'[^\w\s\u4e00-\u9fff，。！？、：；「」『』（）]', '', text)
    if not clean_text.strip():
        clean_text = "很高興和你聊天"

    conf = ANIMATION_STYLES.get(bot_type, ANIMATION_STYLES["solution"])
    openai_voice = conf.get("voice", "alloy")
    edge_voice = conf.get("edge_voice", "zh-TW-HsiaoChenNeural")
    rate = conf.get("rate", "1.0")
    anim_style = conf["style"]

    animation_data = generate_animation_timeline(clean_text, anim_style)

    # 1) OpenAI
    audio_bytes, err = await _openai_tts_bytes(clean_text, voice=openai_voice, fmt=os.getenv("OPENAI_TTS_FORMAT", "mp3"))
    if audio_bytes:
        animation_data.setdefault("meta", {})["provider"] = "openai"
        return {"success": True, "audio_base64": _safe_b64(audio_bytes, "audio/mp3"),
                "animation_data": animation_data, "duration": animation_data.get("total_duration", 3.0), "error": None}

    last_error = f"OpenAI failed: {err}" if err else "OpenAI failed"

    # 2) Edge
    edge_bytes, edge_err = await _edge_tts_bytes(clean_text, voice=edge_voice, rate=rate)
    if edge_bytes:
        animation_data.setdefault("meta", {})["provider"] = "edge-tts"
        return {"success": True, "audio_base64": _safe_b64(edge_bytes, "audio/mp3"),
                "animation_data": animation_data, "duration": animation_data.get("total_duration", 3.0), "error": None}
    if edge_err:
        last_error += f" | {edge_err}"

    # 3) Google
    google_b64, g_err = await _google_tts_b64(clean_text, voice_hint=edge_voice)
    if google_b64:
        animation_data.setdefault("meta", {})["provider"] = "google-tts"
        return {"success": True, "audio_base64": f"data:audio/mp3;base64,{google_b64}",
                "animation_data": animation_data, "duration": animation_data.get("total_duration", 3.0), "error": None}
    if g_err:
        last_error += f" | {g_err}"

    # 全部失敗
    animation_data.setdefault("meta", {})["provider"] = "none"
    public_err = "All TTS providers failed (OpenAI/Edge/Google)."
    if debug and last_error:
        public_err += f" detail: {last_error}"
    return {"success": False, "audio_base64": None, "animation_data": animation_data,
            "duration": animation_data.get("total_duration", 3.0), "error": public_err}

# 舊名相容（main.py 的 debug 端點可能調用）
async def generate_speech_and_animation_old(text: str, bot_type: str) -> Dict[str, Any]:
    return await generate_speech_and_animation(text, bot_type)

# ================= API 端點 =================

@router.post("/animate", response_model=AvatarAnimationResponse)
async def create_avatar_animation(request: AvatarAnimationRequest, background_tasks: BackgroundTasks):
    if not request.text or not request.text.strip():
        raise HTTPException(status_code=400, detail="文字內容不能為空")

    text = request.text.strip()
    if len(text) > 500:
        text = text[:500] + "..."

    try:
        result = await generate_speech_and_animation(text, request.bot_type)
        return AvatarAnimationResponse(**result)
    except Exception as e:
        logger.error(f"動畫生成失敗: {e}")
        try:
            conf = ANIMATION_STYLES.get(request.bot_type, ANIMATION_STYLES["solution"])
            fallback_animation = generate_fallback_animation(text, conf["style"])
            return AvatarAnimationResponse(
                success=False, audio_base64=None,
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
    提供者健康檢查：
      - openai: 環境變數 + base_url ping
      - edge-tts: 是否啟用/安裝
      - google-tts: voices 探測（若有 key）
    """
    info: Dict[str, Any] = {"status": "unknown", "providers": {}, "supported_bots": list(ANIMATION_STYLES.keys())}
    base_url = os.getenv("OPENAI_API_BASE", "https://api.openai.com").rstrip("/")

    # OpenAI
    openai_ok = bool(os.getenv("OPENAI_API_KEY"))
    ping_ok = False
    ping_err = None
    if openai_ok:
        try:
            import aiohttp
            async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=6)) as s:
                async with s.get(base_url, headers={"Authorization": f"Bearer {os.getenv('OPENAI_API_KEY')}"}) as r:
                    ping_ok = (200 <= r.status < 500)  # 2xx/3xx/4xx 都代表 DNS/TLS/路由可達
        except Exception as e:
            ping_err = str(e)
    info["providers"]["openai"] = {"env": openai_ok, "ping": ping_ok, "base": base_url, "error": ping_err}

    # Edge
    try:
        import edge_tts  # type: ignore
        edge_enabled = _env_bool("EDGE_TTS_ENABLED", False)
        info["providers"]["edge_tts"] = {"enabled": edge_enabled, "installed": True}
    except Exception:
        info["providers"]["edge_tts"] = {"enabled": _env_bool("EDGE_TTS_ENABLED", False), "installed": False}

    # Google
    google_key = os.getenv("GOOGLE_TTS_API_KEY")
    if google_key:
        try:
            import aiohttp
            url = f"https://texttospeech.googleapis.com/v1/voices?key={google_key}"
            async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=8)) as session:
                async with session.get(url) as resp:
                    info["providers"]["google_tts"] = {"key": True, "http": resp.status}
        except Exception as e:
            info["providers"]["google_tts"] = {"key": True, "http": None, "error": str(e)}
    else:
        info["providers"]["google_tts"] = {"key": False}

    info["status"] = "healthy" if (info["providers"]["openai"]["env"] and (info["providers"]["openai"]["ping"])) else "degraded"
    return info

@router.get("/styles")
async def get_animation_styles():
    return {"styles": ANIMATION_STYLES, "available_bots": list(ANIMATION_STYLES.keys()),
            "description": "頭像動畫風格配置，包含語音與動作參數"}

@router.post("/preload")
async def preload_common_phrases(background_tasks: BackgroundTasks):
    common_phrases = ["你好，我是你的AI夥伴", "今天想聊什麼呢？", "我在這裡陪著你", "讓我們一起思考一下", "謝謝你和我分享"]
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
    chars = len(text)
    words = len(text.split())
    base_duration = max(2.0, min(12.0, chars * 0.12))  # 上限12s

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
        "metadata": {"text_length": chars, "word_count": words, "generated_at": datetime.utcnow().isoformat(), "tts_attempted": True}
    }

def generate_fallback_animation(text: str, style: Dict[str, Any]) -> Dict[str, Any]:
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
