# app/routers/av.py
from fastapi import APIRouter, HTTPException, Response
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
import os, uuid, asyncio, io, base64
import numpy as np
import aiohttp
from PIL import Image, ImageDraw
from moviepy.editor import ImageClip, AudioFileClip, CompositeVideoClip
import edge_tts

router = APIRouter(prefix="/api/av", tags=["av"])

# ====== 靜態檔路徑 ======
STATIC_ROOT = os.environ.get("AV_STATIC_DIR", "/tmp/av")
AUDIO_DIR = os.path.join(STATIC_ROOT, "audio")
VIDEO_DIR = os.path.join(STATIC_ROOT, "video")
os.makedirs(AUDIO_DIR, exist_ok=True)
os.makedirs(VIDEO_DIR, exist_ok=True)

def get_static_mount():
    # 在 main.py 以 app.mount("/static/av", ...) 掛載
    return ("/static/av", StaticFiles(directory=STATIC_ROOT), "av_static")

# ====== 請求模型 ======
class UtterReq(BaseModel):
    text: str = Field(..., min_length=1, max_length=600)
    image_url: str | None = None        # 可用 http(s) 或相對路徑（/avatars/*.png）
    image_base64: str | None = None     # data:image/png;base64,....
    mouth_box: list[float] | None = None   # [x,y,w,h] in 0~1
    voice: str | None = "zh-TW-HsiaoYuNeural"
    fps: int = 30

# ====== 工具函數 ======
async def _download_bytes(url: str) -> bytes:
    async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=25)) as sess:
        async with sess.get(url) as r:
            if r.status != 200:
                raise HTTPException(400, f"fetch image failed: {r.status}")
            return await r.read()

def _absolutize_frontend_url(possibly_relative: str) -> str:
    """
    將 /xxx 這種相對路徑補成 https://your-frontend/xxx
    來源：環境變數 FRONTEND_URL（例： https://emobot-plus.vercel.app）
    """
    if not possibly_relative:
        return possibly_relative
    if possibly_relative.startswith("http://") or possibly_relative.startswith("https://"):
        return possibly_relative
    base = os.environ.get("FRONTEND_URL", "").strip().rstrip("/")
    if not base:
        # 沒有 FRONTEND_URL 時，回傳原字串給呼叫端以便出錯時回報
        return possibly_relative
    if not possibly_relative.startswith("/"):
        possibly_relative = "/" + possibly_relative
    return base + possibly_relative

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

def _rms_envelope_via_moviepy(audio_path: str, sample_rate: int = 22050, frame_hop: int = 512):
    """
    使用 moviepy(to_soundarray) 讀取音訊並計算 RMS 包絡線。
    避免 librosa 對 mp3 的系統依賴問題（Render 無 ffmpeg 也能靠 imageio-ffmpeg 自帶）。
    """
    clip = AudioFileClip(audio_path)
    dur = float(clip.duration) if clip.duration is not None else 0.0
    # 以 sample_rate 取樣成單聲道波形（值域約 -1~1）
    arr = clip.to_soundarray(fps=sample_rate)  # shape: (N, 1|2)
    if arr.ndim == 2 and arr.shape[1] > 1:
        y = arr.mean(axis=1)
    else:
        y = arr.reshape(-1)
    clip.close()

    y = y.astype(np.float32)
    N = y.shape[0]
    if N == 0:
        return np.zeros(1, dtype=np.float32), dur or 0.0

    # 設定 frame 大小：以 hop 的 4 倍當作窗長
    frame_len = int(frame_hop * 4)
    if frame_len <= 0:
        frame_len = 2048
    hop = int(frame_hop)

    frames = []
    for start in range(0, N, hop):
        end = min(start + frame_len, N)
        seg = y[start:end]
        if seg.size == 0:
            frames.append(0.0)
            continue
        rms = float(np.sqrt(np.mean(seg * seg)) + 1e-8)
        frames.append(rms)
    rms = np.array(frames, dtype=np.float32)

    # 正規化到 0~1
    m = float(rms.max()) if rms.size else 0.0
    if m > 0:
        rms = rms / m

    # 若無法從 clip.duration 取得時間，退回用 N/sample_rate
    if not dur or dur <= 0:
        dur = float(N) / float(sample_rate)

    return rms, dur

def _make_viseme_video(img_bytes: bytes, audio_path: str, out_path: str, mouth_box: list[float] | None, fps: int):
    base = Image.open(io.BytesIO(img_bytes)).convert("RGBA")
    W, H = base.size

    # 預設嘴型區域（圖片下緣中間）
    if not mouth_box or len(mouth_box) != 4:
        mx, my, mw, mh = 0.35, 0.72, 0.30, 0.14
    else:
        mx, my, mw, mh = [float(max(0.0, min(1.0, v))) for v in mouth_box]
    Mx, My = int(mx * W), int(my * H)
    Mw, Mh = int(mw * W), int(mh * H)

    # 以 moviepy 計算 RMS 包絡
    rms, dur = _rms_envelope_via_moviepy(audio_path)
    if dur <= 0:
        dur = AudioFileClip(audio_path).duration or 0.01

    min_open, max_open = 0.1, 1.0
    hop_time = dur / max(1, len(rms))

    def open_ratio_at(t: float) -> float:
        idx = int(t / hop_time)
        idx = max(0, min(len(rms) - 1, idx))
        r = float(rms[idx]) ** 0.65
        return min_open + (max_open - min_open) * r

    base_np = np.array(base)

    def make_frame(t: float):
        img = Image.fromarray(base_np.copy())
        draw = ImageDraw.Draw(img, "RGBA")
        mouth_h = max(2, int(Mh * open_ratio_at(t)))
        cx, cy = Mx + Mw // 2, My + Mh // 2
        bbox = [cx - Mw // 2, cy - mouth_h // 2, cx + Mw // 2, cy + mouth_h // 2]
        draw.ellipse(bbox, fill=(20, 20, 20, 170), outline=(0, 0, 0, 220), width=2)
        return np.array(img.convert("RGB"))

    img_clip = ImageClip(np.array(base.convert("RGB"))).set_duration(dur)
    dyn_clip = img_clip.fl_image(lambda f: f).set_make_frame(make_frame).set_duration(dur)
    audio_clip = AudioFileClip(audio_path)

    final = CompositeVideoClip([dyn_clip]).set_audio(audio_clip)
    final.write_videofile(out_path, fps=fps, codec="libx264", audio_codec="aac", verbose=False, logger=None)
    final.close()
    audio_clip.close()
    img_clip.close()

# ====== CORS 預檢（加強穩定；CORSMiddleware 通常已處理）======
@router.options("/utter")
async def options_utter():
    return Response(status_code=204)

# ====== 主要 API ======
@router.post("/utter")
async def utter(req: UtterReq):
    try:
        text = (req.text or "").strip()
        if not text:
            raise HTTPException(400, "empty text")

        vid_id = str(uuid.uuid4())
        mp3_path = os.path.join(AUDIO_DIR, f"{vid_id}.mp3")
        mp4_path = os.path.join(VIDEO_DIR, f"{vid_id}.mp4")

        # 1) TTS → mp3
        await _tts_to_mp3(text, req.voice or "zh-TW-HsiaoYuNeural", mp3_path)

        # 2) 取得圖片 bytes（支援相對路徑自動補 FRONTEND_URL）
        img_bytes: bytes | None = None
        if req.image_url:
            img_url = _absolutize_frontend_url(req.image_url)
            try:
                img_bytes = await _download_bytes(img_url)
            except HTTPException as he:
                # 如果是相對路徑但 FRONTEND_URL 未設，或抓不到 → 回 400 並帶明確錯誤
                raise HTTPException(400, f"fetch image failed: {he.detail}")
            except Exception as e:
                raise HTTPException(400, f"fetch image error: {str(e)[:120]}")
        elif req.image_base64:
            img_bytes = _decode_base64(req.image_base64)
        else:
            # fallback：簡單底圖
            W, H = 900, 1200
            fallback = Image.new("RGBA", (W, H), (245, 247, 250, 255))
            d = ImageDraw.Draw(fallback)
            d.rectangle([0, int(H*0.62), W, H], fill=(230, 235, 245, 255))
            bio = io.BytesIO()
            fallback.save(bio, format="PNG")
            img_bytes = bio.getvalue()

        # 3) 產生影片（同步嘴型）
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, _make_viseme_video, img_bytes, mp3_path, mp4_path, req.mouth_box, req.fps)

        video_url = f"/static/av/video/{os.path.basename(mp4_path)}"
        audio_url = f"/static/av/audio/{os.path.basename(mp3_path)}"
        return JSONResponse({"ok": True, "video_url": video_url, "audio_url": audio_url})

    except HTTPException:
        raise
    except Exception as e:
        # 攔住未知錯誤，避免 500 洩漏
        return JSONResponse({"ok": False, "error": f"utter_failed: {str(e)[:200]}"}, status_code=400)
