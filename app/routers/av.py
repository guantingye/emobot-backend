# app/routers/av.py
from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
import os, uuid, asyncio, io, math, base64, tempfile
import numpy as np
import aiohttp
import librosa
from PIL import Image, ImageDraw
from moviepy.editor import ImageClip, AudioFileClip, CompositeVideoClip
import edge_tts

router = APIRouter(prefix="/api/av", tags=["av"])

# 準備靜態目錄（Render 可寫 /tmp）
STATIC_ROOT = os.environ.get("AV_STATIC_DIR", "/tmp/av")
AUDIO_DIR = os.path.join(STATIC_ROOT, "audio")
VIDEO_DIR = os.path.join(STATIC_ROOT, "video")
os.makedirs(AUDIO_DIR, exist_ok=True)
os.makedirs(VIDEO_DIR, exist_ok=True)

# 供 main.py 掛載靜態（在下方提供 main.py 片段）
def get_static_mount():
    return ("/static/av", StaticFiles(directory=STATIC_ROOT), "av_static")

class UtterReq(BaseModel):
    text: str = Field(..., min_length=1, max_length=600)
    # 可傳 URL 或 base64 圖（優先 image_url）
    image_url: str | None = None
    image_base64: str | None = None  # "data:image/png;base64,...." 也可
    # 相對座標 [x, y, w, h]，0~1；不傳則用預設嘴型區域（圖片下緣中間）
    mouth_box: list[float] | None = None
    # TTS 聲音（Edge-TTS 名稱）；預設台灣女聲
    voice: str | None = "zh-TW-HsiaoYuNeural"
    # 影片 FPS
    fps: int = 30

async def _download_bytes(url: str) -> bytes:
    async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=20)) as sess:
        async with sess.get(url) as r:
            if r.status != 200:
                raise HTTPException(400, f"fetch image failed: {r.status}")
            return await r.read()

def _decode_base64(data_uri: str) -> bytes:
    try:
        if "," in data_uri:
            data_uri = data_uri.split(",", 1)[1]
        return base64.b64decode(data_uri)
    except Exception:
        raise HTTPException(400, "invalid base64 image")

async def _tts_to_mp3(text: str, voice: str, out_path: str):
    tts = edge_tts.Communicate(text, voice=voice)
    await tts.save(out_path)

def _rms_envelope(audio_path: str, sr: int = 22050, hop: int = 512) -> tuple[np.ndarray, float]:
    # 讀音訊並計算 RMS 包絡線（0~1）
    y, sr = librosa.load(audio_path, sr=sr, mono=True)
    rms = librosa.feature.rms(y=y, frame_length=2048, hop_length=hop).flatten()
    if rms.max() > 0:
        rms = rms / rms.max()
    times = librosa.frames_to_time(np.arange(len(rms)), sr=sr, hop_length=hop)
    return rms, float(times[-1] if len(times) else len(y) / sr)

def _make_viseme_video(img_bytes: bytes, audio_path: str, out_path: str, mouth_box: list[float] | None, fps: int):
    # 載圖
    base = Image.open(io.BytesIO(img_bytes)).convert("RGBA")
    W, H = base.size

    # 預設嘴型區域：圖片下緣中間 30% 寬 × 14% 高
    if not mouth_box or len(mouth_box) != 4:
        mx, my, mw, mh = 0.35, 0.72, 0.30, 0.14
    else:
        mx, my, mw, mh = [float(max(0.0, min(1.0, v))) for v in mouth_box]

    # 轉實際像素
    Mx, My = int(mx * W), int(my * H)
    Mw, Mh = int(mw * W), int(mh * H)

    # 音量包絡線
    rms, dur = _rms_envelope(audio_path)
    if dur <= 0:
        dur = AudioFileClip(audio_path).duration

    # 以 FPS 對齊每一幀的開口比例
    # 線性插值，把 rms 映射到 [min_open, max_open]
    min_open, max_open = 0.1, 1.0
    hop_time = dur / max(1, len(rms))
    def open_ratio_at(t: float) -> float:
        idx = int(t / hop_time)
        idx = max(0, min(len(rms) - 1, idx))
        # 讓張嘴幅度晃動更自然：加一點平滑與非線性
        r = float(rms[idx]) ** 0.65
        return min_open + (max_open - min_open) * r

    base_np = np.array(base)  # (H,W,4)

    def make_frame(t: float):
        img = Image.fromarray(base_np.copy())
        draw = ImageDraw.Draw(img, "RGBA")
        # 嘴為橢圓：寬固定、高根據音量變化
        mouth_h = max(2, int(Mh * open_ratio_at(t)))
        cx, cy = Mx + Mw // 2, My + Mh // 2
        # 橢圓區域（上下以 mouth_h/2 動態張嘴）
        bbox = [cx - Mw // 2, cy - mouth_h // 2, cx + Mw // 2, cy + mouth_h // 2]
        # 畫嘴：深色外緣 + 內部半透明
        draw.ellipse(bbox, fill=(20, 20, 20, 170), outline=(0, 0, 0, 220), width=2)
        return np.array(img.convert("RGB"))

    img_clip = ImageClip(np.array(base.convert("RGB"))).set_duration(dur)
    dyn_clip = img_clip.fl_image(lambda f: f)  # 佔位，實際用 make_frame 產生
    dyn_clip = dyn_clip.set_make_frame(make_frame).set_duration(dur)
    audio_clip = AudioFileClip(audio_path)

    final = CompositeVideoClip([dyn_clip]).set_audio(audio_clip)
    final.write_videofile(out_path, fps=fps, codec="libx264", audio_codec="aac", verbose=False, logger=None)
    final.close()
    audio_clip.close()
    img_clip.close()

@router.post("/utter")
async def utter(req: UtterReq):
    if not req.text.strip():
        raise HTTPException(400, "empty text")

    vid_id = str(uuid.uuid4())
    mp3_path = os.path.join(AUDIO_DIR, f"{vid_id}.mp3")
    mp4_path = os.path.join(VIDEO_DIR, f"{vid_id}.mp4")

    # 1) TTS
    await _tts_to_mp3(req.text.strip(), req.voice or "zh-TW-HsiaoYuNeural", mp3_path)

    # 2) 取得圖片 bytes
    img_bytes = None
    if req.image_url:
        img_bytes = await _download_bytes(req.image_url)
    elif req.image_base64:
        img_bytes = _decode_base64(req.image_base64)
    else:
        # 沒傳圖時：產生一張簡單漸層頭像
        W, H = 900, 1200
        fallback = Image.new("RGBA", (W, H), (245, 247, 250, 255))
        d = ImageDraw.Draw(fallback)
        d.rectangle([0, H*0.6, W, H], fill=(230, 235, 245, 255))
        bio = io.BytesIO()
        fallback.save(bio, format="PNG")
        img_bytes = bio.getvalue()

    # 3) 影片生成（同步嘴型）
    loop = asyncio.get_event_loop()
    await loop.run_in_executor(None, _make_viseme_video, img_bytes, mp3_path, mp4_path, req.mouth_box, req.fps)

    video_url = f"/static/av/video/{os.path.basename(mp4_path)}"
    audio_url = f"/static/av/audio/{os.path.basename(mp3_path)}"
    return JSONResponse({"ok": True, "video_url": video_url, "audio_url": audio_url})
