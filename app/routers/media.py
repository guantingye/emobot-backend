# routers/media.py
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, HttpUrl
from services.tts import synthesize_tts
from services.lipsync import generate_lipsync_video
import uuid
import os

router = APIRouter(prefix="/api/media", tags=["media"])

class LipSyncIn(BaseModel):
    text: str
    speaker: str = "female_zh"
    image_url: HttpUrl

@router.post("/lipsync")
def lipsync(payload: LipSyncIn):
    if not payload.text.strip():
        raise HTTPException(status_code=400, detail="text is required")

    # 輸出檔名
    base_id = str(uuid.uuid4())
    out_dir = os.getenv("MEDIA_OUT_DIR", "static/media")
    os.makedirs(out_dir, exist_ok=True)

    wav_path = os.path.join(out_dir, f"{base_id}.wav")
    mp4_path = os.path.join(out_dir, f"{base_id}.mp4")

    # 1) 產生 TTS（WAV）
    synthesize_tts(
        text=payload.text.strip(),
        wav_out=wav_path,
        speaker=payload.speaker,
        use_piper=bool(int(os.getenv("USE_PIPER", "0"))),
        piper_model=os.getenv("PIPER_MODEL", "zh-TW-WeiChung-Hsiao-NEON"),
    )

    # 2) 產生嘴型影片（MP4）
    generate_lipsync_video(
        image_url=str(payload.image_url),
        audio_path=wav_path,
        mp4_out=mp4_path
    )

    # 3) 回傳可公開存取的網址
    # 假設你已在 FastAPI 啟用 StaticFiles 對應 /static
    public_url_base = os.getenv("PUBLIC_BASE_URL", "").rstrip("/")
    if not public_url_base:
        # 後備：相對路徑
        video_url = f"/static/media/{os.path.basename(mp4_path)}"
    else:
        video_url = f"{public_url_base}/static/media/{os.path.basename(mp4_path)}"

    return {"ok": True, "video_url": video_url}
