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

PERSONA_STYLES: Dict[str, dict] = {
    "empathy": {   # Lumi（平和的亞洲年輕女性）
        "name": "Lumi",
        "voice": "nova",         # ← 改為 nova
        "speaking_rate": 0.93,   # 更慢更柔
        "pause_factor": 1.40,    # 停頓更長
        "energy": 0.80,          # 更溫和
        "color": {"start": "#FFB6C1", "end": "#FF8FB1"},
    },
    "insight": {   # Solin（維持）
        "name": "Solin",
        "voice": "nova",
        "speaking_rate": 0.98,
        "pause_factor": 1.30,
        "energy": 0.77,
        "color": {"start": "#7AC2DD", "end": "#5A8CF2"},
    },
    "solution": {  # Niko（維持）
        "name": "Niko",
        "voice": "nova",
        "speaking_rate": 1.00,
        "pause_factor": 1.20,
        "energy": 0.97,
        "color": {"start": "#3AA87A", "end": "#9AE6B4"},
    },
    "cognitive": { # Clara（更年輕、清亮的亞洲女性）
        "name": "Clara",
        "voice": "nova",         # ← 改為 nova
        "speaking_rate": 1.02,   # 稍快更清亮
        "pause_factor": 1.22,    # 停頓略短
        "energy": 0.92,          # 更有精神但不尖
        "color": {"start": "#7A4DC8", "end": "#B794F4"},
    },
}

@router.post("/animate", response_model=AvatarAnimationResponse)
async def create_avatar_animation(request: AvatarAnimationRequest):
    if not request.text or not request.text.strip():
        raise HTTPException(status_code=400, detail="文字內容不能為空")

    bot_type = (request.bot_type or "solution").strip().lower()
    logger.info(f"[TTS Request] bot_type='{bot_type}', text_preview='{request.text[:50]}...'")
    
    style = PERSONA_STYLES.get(bot_type, PERSONA_STYLES["solution"])
    selected_voice = style["voice"]
    
    logger.info(f"[Voice Mapping] bot_type='{bot_type}' -> voice='{selected_voice}'")

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
        logger.info(f"[TTS Success] voice='{selected_voice}', audio_size={len(mp3_b64) if mp3_b64 else 0}")
    except Exception as e:
        logger.exception(f"[TTS Failed] voice='{selected_voice}', error={str(e)}")

    ok = mp3_b64 is not None
    
    meta_info = {
        "provider": "openai",
        "model": "tts-1-hd",
        "bot_type": bot_type,
        "voice": selected_voice,
        "speaking_rate": style["speaking_rate"],
        "pause_factor": style["pause_factor"],
        "energy": style["energy"],
        "color": style["color"],
        "generated_at": datetime.utcnow().isoformat() + "Z",
        "text_length": len(text)
    }
    
    logger.info(f"[Response Meta] {meta_info}")
    
    return AvatarAnimationResponse(
        success=ok,
        audio_base64=f"data:audio/mp3;base64,{mp3_b64}" if ok else None,
        animation_data=animation,
        duration=animation.get("total_duration", 3.0),
        error=None if ok else "TTS生成失敗,僅顯示動畫(靜音)。",
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
    return re.sub(r"[^\u4e00-\u9fffA-Za-z0-9,。!?、:;「」『』()\s]", "", text)

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

    logger.info(f"[TTS Start] voice='{voice}', chunks={len(chunks)}, text_preview='{text[:50]}...'")

    speed = min(1.8, max(0.5, speaking_rate))

    async def synth_one(t: str, chunk_idx: int) -> bytes:
        def _do() -> bytes:
            logger.info(f"[Chunk {chunk_idx+1}/{len(chunks)}] voice='{voice}', text='{t[:30]}...', speed={speed}")
            res = client.audio.speech.create(
                model="tts-1-hd",
                voice=voice,
                input=t,
                response_format="mp3",
                speed=speed
            )
            return res.content
        return await asyncio.to_thread(_do)

    tasks = [synth_one(ck, idx) for idx, ck in enumerate(chunks)]
    results = await asyncio.gather(*tasks, return_exceptions=True)

    out_bytes = bytearray()
    success_count = 0
    for idx, result in enumerate(results):
        if isinstance(result, Exception):
            logger.warning(f"Chunk {idx} failed: {result}")
            continue
        if not result or len(result) < 500:
            logger.warning(f"Chunk {idx} too small: {len(result) if result else 0}")
            continue
        out_bytes.extend(result)
        success_count += 1
        if idx < len(results) - 1:
            await asyncio.sleep(0.04 * pause_factor)

    logger.info(f"[TTS Complete] voice='{voice}', success={success_count}/{len(chunks)}, total_bytes={len(out_bytes)}")

    if len(out_bytes) < 800:
        return None
    return base64.b64encode(bytes(out_bytes)).decode("utf-8")

def generate_animation_timeline(text: str, speaking_rate: float, energy: float, pause_factor: float) -> Dict[str, Any]:
    char_time = 0.10 / max(0.6, speaking_rate)
    base_duration = max(2.0, min(24.0, len(text) * char_time))

    mouth_frames = []
    t = 0.0
    for i, ch in enumerate(text):
        if ch in ",、:;":
            mouth_frames.append({"time": t, "mouth_openness": 0.05, "type": "pause"})
            t += 0.10 * pause_factor
        elif ch in "。!?":
            mouth_frames.append({"time": t, "mouth_openness": 0.1, "type": "stop"})
            t += 0.20 * pause_factor
        else:
            open_val = 0.4 + (0.3 * energy) if ch not in " \n" else 0.05
            mouth_frames.append({"time": t, "mouth_openness": open_val, "type": "talk"})
            t += char_time

    total_duration = max(base_duration, t + 0.5)

    return {
        "total_duration": total_duration,
        "mouth_frames": mouth_frames,
        "energy": energy,
        "speaking_rate": speaking_rate,
        "pause_factor": pause_factor
    }