# app/routers/avatar_animation.py - OpenAI TTS 分段版（無 Edge/Google；含超時保護與長文分段）
import os
import asyncio
import base64
import tempfile
import re
from typing import Optional, Dict, Any, Tuple, List
from datetime import datetime

from fastapi import APIRouter, HTTPException, BackgroundTasks
from pydantic import BaseModel
import logging

logger = logging.getLogger(__name__)
router = APIRouter()

# ================== 工具：環境變數 ==================
def _env_bool(name: str, default: bool = False) -> bool:
    v = os.getenv(name)
    if v is None:
        return default
    return str(v).lower() in ("1", "true", "yes", "y", "on")

def _env_int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, "").strip() or default)
    except Exception:
        return default

def _safe_b64(data: bytes, mime: str = "audio/mp3") -> str:
    return f"data:{mime};base64,{base64.b64encode(data).decode('utf-8')}"

# ================== 超時設定（秒） ==================
OPENAI_HTTP_TIMEOUT = _env_int("OPENAI_TTS_HTTP_TIMEOUT", 10)  # 單段 TTS 請求
TOTAL_ROUTE_TIMEOUT = _env_int("AVATAR_TTS_TIMEOUT", 30)       # /animate 整體超時
MAX_CHARS = _env_int("OPENAI_TTS_MAX_CHARS", 200)              # 分段長度
DEBUG_TTS = _env_bool("DEBUG_TTS", False)

# ================== 動畫風格 ==================
ANIMATION_STYLES = {
    "empathy":  {"name": "Lumi",   "voice": "alloy",
                 "style": {"mouth_intensity": 0.8, "blink_frequency": 0.7, "head_movement": 0.6, "emotion": "gentle"}},
    "insight":  {"name": "Solin",  "voice": "alloy",
                 "style": {"mouth_intensity": 0.7, "blink_frequency": 0.5, "head_movement": 0.4, "emotion": "thoughtful"}},
    "solution": {"name": "Niko",   "voice": "alloy",
                 "style": {"mouth_intensity": 0.9, "blink_frequency": 0.6, "head_movement": 0.8, "emotion": "confident"}},
    "cognitive":{"name": "Clara",  "voice": "alloy",
                 "style": {"mouth_intensity": 0.6, "blink_frequency": 0.4, "head_movement": 0.3, "emotion": "calm"}},
}

# ================== Pydantic models ==================
class AvatarAnimationRequest(BaseModel):
    text: str
    bot_type: Optional[str] = "solution"
    animation_style: Optional[str] = "normal"
    voice_id: Optional[str] = None  # 若要覆寫預設 voice

class AvatarAnimationResponse(BaseModel):
    success: bool
    audio_base64: Optional[str] = None
    audio_segments: Optional[List[str]] = None  # 多段 audio（data URL）
    animation_data: Optional[Dict[str, Any]] = None
    duration: Optional[float] = None
    error: Optional[str] = None

# ================== 文字分段 ==================
def _chunk_text_by_sentence(text: str, max_len: int = 200) -> List[str]:
    """
    依標點優先切段，超長句再硬切；確保每段長度不超過 max_len。
    """
    sents = re.split(r'(?<=[。！？!?])\s*', text)
    chunks: List[str] = []
    cur = ""
    for s in sents:
        if not s:
            continue
        if len(cur) + len(s) <= max_len:
            cur += s
        else:
            if cur:
                chunks.append(cur)
            if len(s) <= max_len:
                cur = s
            else:
                # 對超長單句再細切
                start = 0
                while start < len(s):
                    part = s[start:start+max_len]
                    if len(part) == max_len:
                        chunks.append(part)
                    else:
                        cur = part
                    start += max_len
    if cur:
        chunks.append(cur)
    return chunks

# ================== OpenAI TTS（HTTP 直呼） ==================
async def _openai_tts_bytes(text: str, voice: str = "alloy", fmt: str = "mp3") -> Tuple[Optional[bytes], Optional[str]]:
    """
    以 OpenAI TTS 產生音訊（bytes）。失敗回 (None, error_msg)。
    走 /v1/audio/speech HTTP API，加總超時。
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

    try:
        import aiohttp
    except Exception as e:
        return None, f"aiohttp missing: {e}"

    url = f"{base_url}/v1/audio/speech"
    payload = {"model": model, "voice": voice, "input": text, "format": fmt}
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}

    try:
        timeout = aiohttp.ClientTimeout(total=max(3, OPENAI_HTTP_TIMEOUT))
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
    except asyncio.TimeoutError:
        return None, f"OpenAI HTTP timeout ({OPENAI_HTTP_TIMEOUT}s)"
    except Exception as e:
        return None, f"OpenAI HTTP error: {e}"

# ================== 動畫生成 ==================
def generate_animation_timeline(text: str, style: Dict[str, Any]) -> Dict[str, Any]:
    """
    依文字長度與風格產生基礎動畫時間軸（嘴型/眨眼/頭部微動）
    """
    chars = len(text)
    words = len(text.split())
    base_duration = max(2.0, min(12.0, chars * 0.12))  # 上限 12 秒，避免過長

    mouth_intensity = style.get("mouth_intensity", 0.8)
    blink_frequency = style.get("blink_frequency", 0.6)
    head_movement = style.get("head_movement", 0.5)
    emotion = style.get("emotion", "neutral")

    mouth_frames = []
    t = 0.0
    for i, ch in enumerate(text):
        if ch in '，。！？、：；':
            mouth_frames.append({"time": t, "mouth_openness": 0.0, "type": "pause"})
            t += 0.28
        elif ch.strip():
            openness = round(mouth_intensity * (0.35 + 0.35 * ((i % 3) / 2.0)), 3)
            mouth_frames.append({"time": t, "mouth_openness": openness, "type": "phoneme"})
            t += 0.11
        if t > base_duration:
            break

    blink_frames = []
    if blink_frequency > 0:
        interval = max(1.6, 2.4 / blink_frequency)
        bt = interval
        while bt < base_duration:
            blink_frames.extend([
                {"time": round(bt, 2), "eye_state": "closing"},
                {"time": round(bt + 0.08, 2), "eye_state": "closed"},
                {"time": round(bt + 0.16, 2), "eye_state": "opening"},
                {"time": round(bt + 0.24, 2), "eye_state": "open"},
            ])
            bt += interval

    head_frames = []
    if head_movement > 0.2:
        ht = 0.0
        while ht < base_duration:
            x = round(head_movement * 2 * (0.5 - ((ht % 4) / 4)), 3)
            y = round(head_movement * 1 * (0.5 - ((ht % 6) / 6)), 3)
            head_frames.append({"time": round(ht, 2), "head_x": x, "head_y": y})
            ht += 0.5

    return {
        "total_duration": max(base_duration, t),
        "mouth_animation": mouth_frames,
        "blink_animation": blink_frames,
        "head_animation": head_frames,
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
    """
    純靜默後備動畫（避免前端 pending）
    """
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

# ================== 合成 + 動畫主流程（支援分段） ==================
async def generate_speech_and_animation(text: str, bot_type: str) -> Dict[str, Any]:
    clean_text = re.sub(r'[^\w\s\u4e00-\u9fff，。！？、：；「」『』（）]', '', text)
    if not clean_text.strip():
        clean_text = "很高興和你聊天"

    conf = ANIMATION_STYLES.get(bot_type, ANIMATION_STYLES["solution"])
    openai_voice = conf.get("voice", "alloy")
    anim_style = conf["style"]

    # 預先產生一份動畫（若分段會重建拼接）
    animation_data = generate_animation_timeline(clean_text, anim_style)

    text_len = len(clean_text)
    is_segmented = text_len > MAX_CHARS

    last_error = None

    if is_segmented:
        parts = _chunk_text_by_sentence(clean_text, max_len=MAX_CHARS)
        audio_segments: List[str] = []
        # 用於拼接動畫時間軸
        timeline_mouth_all: List[Dict[str, Any]] = []
        timeline_blink_all: List[Dict[str, Any]] = []
        timeline_head_all: List[Dict[str, Any]] = []
        time_offset = 0.0

        for idx, part in enumerate(parts):
            audio_bytes, err = await _openai_tts_bytes(part, voice=openai_voice, fmt=os.getenv("OPENAI_TTS_FORMAT", "mp3"))
            if not audio_bytes:
                last_error = f"OpenAI failed on segment {idx+1}/{len(parts)}: {err}"
                break
            audio_segments.append(_safe_b64(audio_bytes, "audio/mp3"))

            seg_anim = generate_animation_timeline(part, anim_style)
            # 偏移 mouth
            for f in seg_anim["mouth_animation"]:
                f["time"] += time_offset
                timeline_mouth_all.append(f)
            # 偏移 blink
            for f in seg_anim["blink_animation"]:
                f["time"] += time_offset
                timeline_blink_all.append(f)
            # 偏移 head
            for f in seg_anim["head_animation"]:
                f["time"] += time_offset
                timeline_head_all.append(f)

            time_offset += seg_anim.get("total_duration", 3.0)

        if audio_segments and len(audio_segments) == len(parts):
            animation_data["mouth_animation"] = timeline_mouth_all
            animation_data["blink_animation"] = timeline_blink_all
            animation_data["head_animation"] = timeline_head_all
            animation_data["total_duration"] = time_offset
            animation_data.setdefault("meta", {})["provider"] = "openai"
            return {
                "success": True,
                "audio_base64": None,
                "audio_segments": audio_segments,
                "animation_data": animation_data,
                "duration": time_offset,
                "error": None
            }
        else:
            # 分段失敗 → 嘗試單段全文（保底）
            audio_bytes, err = await _openai_tts_bytes(clean_text, voice=openai_voice, fmt=os.getenv("OPENAI_TTS_FORMAT", "mp3"))
            if audio_bytes:
                animation_data.setdefault("meta", {})["provider"] = "openai"
                return {
                    "success": True,
                    "audio_base64": _safe_b64(audio_bytes, "audio/mp3"),
                    "animation_data": animation_data,
                    "duration": animation_data.get("total_duration", 3.0),
                    "error": None
                }
            last_error = last_error or f"OpenAI single-shot failed: {err}"

    else:
        # 短文本 → 直接單段
        audio_bytes, err = await _openai_tts_bytes(clean_text, voice=openai_voice, fmt=os.getenv("OPENAI_TTS_FORMAT", "mp3"))
        if audio_bytes:
            animation_data.setdefault("meta", {})["provider"] = "openai"
            return {
                "success": True,
                "audio_base64": _safe_b64(audio_bytes, "audio/mp3"),
                "animation_data": animation_data,
                "duration": animation_data.get("total_duration", 3.0),
                "error": None
            }
        last_error = f"OpenAI failed: {err}" if err else "OpenAI failed"

    # 仍失敗 → 回靜默動畫
    animation_data.setdefault("meta", {})["provider"] = "none"
    public_err = "OpenAI TTS failed."
    if DEBUG_TTS and last_error:
        public_err += f" detail: {last_error}"
    return {
        "success": False,
        "audio_base64": None,
        "audio_segments": None,
        "animation_data": animation_data,
        "duration": animation_data.get("total_duration", 3.0),
        "error": public_err
    }

# 舊名相容
async def generate_speech_and_animation_old(text: str, bot_type: str) -> Dict[str, Any]:
    return await generate_speech_and_animation(text, bot_type)

# ================== API 端點 ==================
@router.post("/animate", response_model=AvatarAnimationResponse)
async def create_avatar_animation(request: AvatarAnimationRequest, background_tasks: BackgroundTasks):
    if not request.text or not request.text.strip():
        raise HTTPException(status_code=400, detail="文字內容不能為空")

    text = request.text.strip()
    if len(text) > 1000:
        text = text[:1000] + "..."

    try:
        result = await asyncio.wait_for(
            generate_speech_and_animation(text, request.bot_type),
            timeout=max(5, TOTAL_ROUTE_TIMEOUT)
        )
        return AvatarAnimationResponse(**result)
    except asyncio.TimeoutError:
        # 超時 → 回靜默動畫，避免前端 pending
        conf = ANIMATION_STYLES.get(request.bot_type, ANIMATION_STYLES["solution"])
        fallback_animation = generate_fallback_animation(text, conf["style"])
        err = f"TTS overall timeout ({TOTAL_ROUTE_TIMEOUT}s)"
        if DEBUG_TTS:
            err += " | openai may be slow or unreachable"
        return AvatarAnimationResponse(
            success=False,
            audio_base64=None,
            audio_segments=None,
            animation_data=fallback_animation,
            duration=fallback_animation.get("total_duration", 3.0),
            error=err
        )
    except Exception as e:
        logger.error(f"動畫生成失敗: {e}")
        conf = ANIMATION_STYLES.get(request.bot_type, ANIMATION_STYLES["solution"])
        fallback_animation = generate_fallback_animation(text, conf["style"])
        return AvatarAnimationResponse(
            success=False,
            audio_base64=None,
            audio_segments=None,
            animation_data=fallback_animation,
            duration=fallback_animation.get("total_duration", 3.0),
            error=f"動畫生成失敗，使用靜默模式: {str(e)}"
        )

@router.get("/health")
async def health_check():
    """
    僅檢測 OpenAI TTS：環境變數、基本連線性、超時設定回報
    """
    info: Dict[str, Any] = {"status": "unknown", "providers": {}, "supported_bots": list(ANIMATION_STYLES.keys())}
    base_url = os.getenv("OPENAI_API_BASE", "https://api.openai.com").rstrip("/")
    openai_ok = bool(os.getenv("OPENAI_API_KEY"))
    ping_ok = False
    ping_err = None

    if openai_ok:
        try:
            import aiohttp
            async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=6)) as s:
                async with s.get(base_url, headers={"Authorization": f"Bearer {os.getenv('OPENAI_API_KEY')}"}) as r:
                    ping_ok = (200 <= r.status < 500)
        except Exception as e:
            ping_err = str(e)

    info["providers"]["openai"] = {
        "env": openai_ok,
        "ping": ping_ok,
        "base": base_url,
        "error": ping_err,
        "http_timeout": OPENAI_HTTP_TIMEOUT,
        "route_timeout": TOTAL_ROUTE_TIMEOUT,
        "max_chars": MAX_CHARS,
    }
    info["status"] = "healthy" if (openai_ok and ping_ok) else ("degraded" if openai_ok else "error")
    return info

@router.get("/styles")
async def get_animation_styles():
    return {
        "styles": ANIMATION_STYLES,
        "available_bots": list(ANIMATION_STYLES.keys()),
        "description": "頭像動畫風格配置，包含語音與動作參數"
    }

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
    try:
        result = await asyncio.wait_for(generate_speech_and_animation(test_text, "solution"), timeout=max(5, TOTAL_ROUTE_TIMEOUT))
        results["solution"] = {
            "success": result["success"],
            "has_audio": bool(result.get("audio_base64") or result.get("audio_segments")),
            "has_animation": bool(result.get("animation_data")),
            "provider": (result.get("animation_data", {}) or {}).get("meta", {}).get("provider"),
            "duration": result.get("duration"),
            "error": result.get("error")
        }
    except asyncio.TimeoutError:
        results["solution"] = {"success": False, "error": f"overall timeout ({TOTAL_ROUTE_TIMEOUT}s)"}
    except Exception as e:
        results["solution"] = {"success": False, "error": str(e)}

    overall_success = any(r.get("success", False) for r in results.values())
    return {"overall_success": overall_success, "test_text": test_text, "results": results, "timestamp": datetime.utcnow().isoformat()}
