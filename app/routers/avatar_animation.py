# app/routers/avatar_animation.py
import os
import asyncio
import base64
import re
from typing import Optional, Dict, Any, List
from datetime import datetime

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
import logging

logger = logging.getLogger(__name__)
router = APIRouter()

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
    meta: Optional[Dict[str, Any]] = None

BOT_VOICES = {
    "empathy": "nova",
    "insight": "shimmer",
    "solution": "alloy",
    "cognitive": "echo"
}

PERSONA_STYLES = {
    "empathy": {
        "name": "Lumi",
        "voice": "nova",
        "speaking_rate": 0.95,
        "pause_factor": 1.25,
        "energy": 0.75,
        "color": {"start": "#FFB6C1", "end": "#FF8FB1"}
    },
    "insight": {
        "name": "Solin",
        "voice": "shimmer",
        "speaking_rate": 1.0,
        "pause_factor": 1.15,
        "energy": 0.85,
        "color": {"start": "#7AC2DD", "end": "#5A8CF2"}
    },
    "solution": {
        "name": "Niko",
        "voice": "alloy",
        "speaking_rate": 1.05,
        "pause_factor": 1.05,
        "energy": 0.95,
        "color": {"start": "#3AA87A", "end": "#9AE6B4"}
    },
    "cognitive": {
        "name": "Clara",
        "voice": "echo",
        "speaking_rate": 1.0,
        "pause_factor": 1.15,
        "energy": 0.88,
        "color": {"start": "#7A4DC8", "end": "#B794F4"}
    },
}

@router.post("/animate", response_model=AvatarAnimationResponse)
async def create_avatar_animation(request: AvatarAnimationRequest):
    if not request.text or not request.text.strip():
        raise HTTPException(status_code=400, detail="文字內容不能為空")

    # 標準化並記錄 bot_type
    bot_type = (request.bot_type or "solution").strip().lower()
    logger.info(f"🎯 [TTS Request] bot_type='{bot_type}', text_preview='{request.text[:50]}...'")
    
    # 獲取對應風格設定
    style = PERSONA_STYLES.get(bot_type, PERSONA_STYLES["solution"])
    selected_voice = style["voice"]
    
    logger.info(f"🎤 [Voice Mapping] bot_type='{bot_type}' → voice='{selected_voice}'")

    text = sanitize_text(request.text.strip())
    if len(text) > 1600:
        text = text[:1600] + "..."

    animation = generate_animation_timeline(
        text=text,
        speaking_rate=style["speaking_rate"],
        energy=style["energy"],
        pause_factor=style["pause_factor"]
    )

    mp3_b64 = None
    try:
        mp3_b64 = await tts_openai_chunked(
            text=text,
            speaking_rate=style["speaking_rate"],
            pause_factor=style["pause_factor"],
            voice=selected_voice
        )
        logger.info(f"✅ [TTS Success] voice='{selected_voice}', audio_size={len(mp3_b64) if mp3_b64 else 0}")
    except Exception as e:
        logger.exception(f"❌ [TTS Failed] voice='{selected_voice}', error={str(e)}")

    ok = mp3_b64 is not None
    
    # 構建完整的 meta 資訊
    meta_info = {
        "provider": "openai",
        "model": "tts-1",
        "bot_type": bot_type,
        "voice": selected_voice,
        "speaking_rate": style["speaking_rate"],
        "pause_factor": style["pause_factor"],
        "energy": style["energy"],
        "color": style["color"],
        "generated_at": datetime.utcnow().isoformat() + "Z",
        "text_length": len(text)
    }
    
    logger.info(f"📦 [Response Meta] {meta_info}")
    
    return AvatarAnimationResponse(
        success=ok,
        audio_base64=f"data:audio/mp3;base64,{mp3_b64}" if ok else None,
        animation_data=animation,
        duration=animation.get("total_duration", 3.0),
        error=None if ok else "TTS 生成失敗,僅顯示動畫(靜音)。",
        meta=meta_info
    )

@router.get("/health")
async def health_check():
    openai_key = bool(os.getenv("OPENAI_API_KEY"))
    return {
        "status": "healthy" if openai_key else "error",
        "providers": {"openai": {"env": openai_key}},
        "supported_bots": list(PERSONA_STYLES.keys()),
        "voice_mapping": {k: v["voice"] for k, v in PERSONA_STYLES.items()}
    }

@router.get("/styles")
async def get_styles():
    return {"styles": PERSONA_STYLES}

def sanitize_text(text: str) -> str:
    return re.sub(r"[^\u4e00-\u9fffA-Za-z0-9,。!?、:;「」『』()…\s]", "", text)

def split_text_for_tts(text: str, max_len: int = 200) -> List[str]:
    parts = re.split(r"([。!?])", text)
    sentences = []
    for i in range(0, len(parts), 2):
        seg = parts[i]
        if not seg.strip():
            continue
        end = parts[i + 1] if i + 1 < len(parts) else ""
        sentences.append((seg + end).strip())

    chunks = []
    buf = ""
    for s in sentences:
        if len(buf) + len(s) <= max_len:
            buf += (s if not buf else "" + s)
        else:
            if buf:
                chunks.append(buf)
            if len(s) <= max_len:
                buf = s
            else:
                for j in range(0, len(s), max_len):
                    chunks.append(s[j: j + max_len])
                buf = ""
    if buf:
        chunks.append(buf)
    return chunks

async def tts_openai_chunked(text: str, speaking_rate: float, pause_factor: float, voice: str) -> Optional[str]:
    api_key = os.getenv("OPENAI_API_KEY", "")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY not set")

    chunks = split_text_for_tts(text, max_len=200)
    if not chunks:
        return None

    from openai import OpenAI
    client = OpenAI(api_key=api_key)

    logger.info(f"🎤 [TTS Start] voice='{voice}', chunks={len(chunks)}, text_preview='{text[:50]}...'")

    async def synth_one(t: str, chunk_idx: int) -> bytes:
        def _do() -> bytes:
            logger.info(f"🎵 [Chunk {chunk_idx+1}/{len(chunks)}] voice='{voice}', text='{t[:30]}...'")
            res = client.audio.speech.create(
                model="tts-1",
                voice=voice,
                input=t,
                response_format="mp3",
                speed=1.0
            )
            return res.content
        return await asyncio.to_thread(_do)

    tasks = [synth_one(ck, idx) for idx, ck in enumerate(chunks)]
    results = await asyncio.gather(*tasks, return_exceptions=True)

    out_bytes = bytearray()
    success_count = 0
    for idx, result in enumerate(results):
        if isinstance(result, Exception):
            logger.warning(f"⚠️ Chunk {idx} failed: {result}")
            continue
        if not result or len(result) < 500:
            logger.warning(f"⚠️ Chunk {idx} too small: {len(result) if result else 0}")
            continue
        out_bytes.extend(result)
        success_count += 1
        if idx < len(results) - 1:
            await asyncio.sleep(0.05 * pause_factor)

    logger.info(f"✅ [TTS Complete] voice='{voice}', success={success_count}/{len(chunks)}, total_bytes={len(out_bytes)}")

    if len(out_bytes) < 800:
        return None
    return base64.b64encode(bytes(out_bytes)).decode("utf-8")

def generate_animation_timeline(text: str, speaking_rate: float, energy: float, pause_factor: float) -> Dict[str, Any]:
    char_time = 0.11 / max(0.6, speaking_rate)
    base_duration = max(2.0, min(24.0, len(text) * char_time))

    mouth_frames = []
    t = 0.0
    for i, ch in enumerate(text):
        if ch in ",、:;":
            mouth_frames.append({"time": t, "mouth_openness": 0.05, "type": "pause"})
            t += 0.12 * pause_factor
        elif ch in "。!?":
            mouth_frames.append({"time": t, "mouth_openness": 0.02, "type": "pause"})
            t += 0.20 * pause_factor
        elif ch.strip():
            openness = 0.3 + 0.6 * ((i % 4) / 3.0)
            mouth_frames.append({"time": t, "mouth_openness": float(openness)})
            t += 0.09

    blink_frames = []
    blink_interval = max(1.5, 2.2 - energy * 0.5)
    bt = blink_interval
    while bt < base_duration + 1.0:
        blink_frames.extend([
            {"time": bt, "eye_state": "closing"},
            {"time": bt + 0.08, "eye_state": "closed"},
            {"time": bt + 0.15, "eye_state": "opening"},
            {"time": bt + 0.22, "eye_state": "open"},
        ])
        bt += blink_interval

    head_frames = []
    ht = 0.0
    amp = 0.5 + energy * 0.5
    while ht < base_duration + 0.5:
        head_frames.append({
            "time": ht,
            "head_x": round(amp * 0.5 * (0.5 - (ht % 3) / 3), 3),
            "head_y": round(amp * 0.3 * (0.5 - (ht % 5) / 5), 3),
        })
        ht += 0.42

    total = max(base_duration, mouth_frames[-1]["time"] + 0.5 if mouth_frames else 2.5)
    return {
        "total_duration": float(total),
        "mouth_animation": mouth_frames,
        "blink_animation": blink_frames,
        "head_animation": head_frames,
        "metadata": {
            "text_length": len(text),
            "generated_at": datetime.utcnow().isoformat() + "Z",
            "speaking_rate": speaking_rate,
            "pause_factor": pause_factor
        }
    }