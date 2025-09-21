# app/routers/av.py
from fastapi import APIRouter, HTTPException, Request, Response
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
import os, uuid, asyncio, io, base64, traceback
import numpy as np
import aiohttp
from PIL import Image, ImageDraw
from moviepy.editor import ImageClip, AudioFileClip, CompositeVideoClip

router = APIRouter(prefix="/api/av", tags=["av"])

# ====== 靜態檔路徑 ======
STATIC_ROOT = os.environ.get("AV_STATIC_DIR", "/tmp/av")
AUDIO_DIR = os.path.join(STATIC_ROOT, "audio")
VIDEO_DIR = os.path.join(STATIC_ROOT, "video")
os.makedirs(AUDIO_DIR, exist_ok=True)
os.makedirs(VIDEO_DIR, exist_ok=True)

def get_static_mount():
    return ("/static/av", StaticFiles(directory=STATIC_ROOT), "av_static")

# ====== 請求模型 ======
class UtterReq(BaseModel):
    text: str = Field(..., min_length=1, max_length=600)
    image_url: str | None = None
    image_base64: str | None = None
    mouth_box: list[float] | None = None   # [x,y,w,h] in 0~1
    voice: str | None = "zh-TW-HsiaoYuNeural"
    fps: int = 30

# ====== 工具函數 ======
def _to_absolute_url(possible_url: str | None) -> str | None:
    """把 /static/media/... 這類相對路徑補成絕對網址（用 FRONTEND_URL 當 base）"""
    if not possible_url:
        return None
    u = possible_url.strip()
    if not u:
        return None
    if u.startswith("http://") or u.startswith("https://") or u.startswith("data:"):
        return u
    base = os.environ.get("FRONTEND_URL", "").strip().rstrip("/")
    if not base:
        # 後備：你也可以改成你的正式站
        base = "https://emobot-plus.vercel.app"
    if not u.startswith("/"):
        u = "/" + u
    return f"{base}{u}"

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
    import edge_tts  # 延後載入，避免啟動期相依延遲
    tts = edge_tts.Communicate(text, voice=voice)
    await tts.save(out_path)

def _rms_envelope_via_moviepy(audio_path: str, target_sr: int = 16000, hop: int = 512):
    """
    使用 moviepy 直接讀取 mp3 為 numpy，避免 librosa/soundfile 在 Render 缺 libsndfile。
    回傳 (rms[0..1], duration_seconds)
    """
    clip = AudioFileClip(audio_path)
    dur = float(clip.duration)
    # 取樣：to_soundarray(fps) -> shape (N, 2) 或 (N,)
    arr = clip.to_soundarray(fps=target_sr)  # [-1,1] 浮點
    clip.close()
    if arr.ndim == 2:
        arr = arr.mean(axis=1)  # 轉單聲道
    arr = arr.astype("float32")

    # 分幀計算 RMS
    n = len(arr)
    if n == 0:
        return np.zeros(1, dtype="float32"), dur
    frame = 2048
    hop = int(hop)
    if hop <= 0:
        hop = 512
    starts = np.arange(0, max(1, n - frame), hop, dtype=int)
    rms_list = []
    for s in starts:
        e = min(s + frame, n)
        seg = arr[s:e]
        if len(seg) == 0:
            rms_list.append(0.0)
        else:
            rms_list.append(float(np.sqrt(np.mean(seg * seg))))
    rms = np.array(rms_list, dtype="float32")
    if rms.size == 0:
        rms = np.array([0.0], dtype="float32")
    m = rms.max()
    if m > 0:
        rms = rms / m
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

    # 取音量包絡線（改用 moviepy，避開 librosa 相依）
    rms, dur = _rms_envelope_via_moviepy(audio_path)
    if dur <= 0:
        dur = 1.0

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

# ====== 預檢（有些代理會需要；CORSMiddleware 通常已處理）======
@router.options("/utter")
async def options_utter():
    return Response(status_code=204)

# ====== 主要 API ======
@router.post("/utter")
async def utter(req: UtterReq):
    try:
        text = req.text.strip()
        if not text:
            raise HTTPException(400, "empty text")

        vid_id = str(uuid.uuid4())
        mp3_path = os.path.join(AUDIO_DIR, f"{vid_id}.mp3")
        mp4_path = os.path.join(VIDEO_DIR, f"{vid_id}.mp4")

        # 1) TTS
        await _tts_to_mp3(text, req.voice or "zh-TW-HsiaoYuNeural", mp3_path)

        # 2) 取得圖片
        img_bytes: bytes | None = None
        if req.image_base64:
            img_bytes = _decode_base64(req.image_base64)
        elif req.image_url:
            abs_url = _to_absolute_url(req.image_url)
            try:
                img_bytes = await _download_bytes(abs_url)
            except Exception as e:
                # 抓圖失敗就降級為 fallback，不再 500
                img_bytes = None

        if img_bytes is None:
            # fallback：簡單底圖
            W, H = 900, 1200
            fallback = Image.new("RGBA", (W, H), (245, 247, 250, 255))
            d = ImageDraw.Draw(fallback)
            d.rectangle([0, int(H*0.62), W, H], fill=(230, 235, 245, 255))
            bio = io.BytesIO()
            fallback.save(bio, format="PNG")
            img_bytes = bio.getvalue()

        # 3) 產生影片
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, _make_viseme_video, img_bytes, mp3_path, mp4_path, req.mouth_box, req.fps)

        video_url = f"/static/av/video/{os.path.basename(mp4_path)}"
        audio_url = f"/static/av/audio/{os.path.basename(mp3_path)}"
        return JSONResponse({"ok": True, "video_url": video_url, "audio_url": audio_url})
    except HTTPException as he:
        # 保持語意化錯誤
        return JSONResponse({"ok": False, "error": str(he.detail)}, status_code=he.status_code)
    except Exception as e:
        # 回傳可讀錯誤訊息，避免 500 毫無資訊
        err = f"{e.__class__.__name__}: {e}"
        print("AV/utter error:", err)
        traceback.print_exc()
        return JSONResponse({"ok": False, "error": err}, status_code=500)
