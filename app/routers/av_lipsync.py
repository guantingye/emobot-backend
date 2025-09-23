# app/routers/av_lipsync.py
import os, uuid, math, asyncio, aiohttp, aiofiles
from fastapi import APIRouter, HTTPException
from fastapi import Depends
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from pathlib import Path
import numpy as np
from pydub import AudioSegment
import cv2
import imageio.v3 as iio
import edge_tts

MEDIA_DIR = Path(os.getenv("MEDIA_DIR", "static/media")).resolve()
MEDIA_DIR.mkdir(parents=True, exist_ok=True)

router = APIRouter(prefix="/api/av", tags=["av"])

class TalkIn(BaseModel):
    text: str = Field(..., min_length=1, max_length=2000)
    avatar_url: str = Field(..., description="用戶當前機器人圖片 URL")
    voice: str = Field(default="zh-TW-HsiaoChenNeural")  # edge-tts 免金鑰語音
    speed: str = Field(default="+0%")  # 語速可微調

async def tts_to_wav(text: str, voice: str, speed: str, out_wav: Path):
    # 產出 wav（edge-tts 先輸出 mp3 再轉 wav）
    mp3_path = out_wav.with_suffix(".mp3")
    communicate = edge_tts.Communicate(text, voice=voice, rate=speed)
    await communicate.save(str(mp3_path))
    # 轉 wav（16k mono）
    audio = AudioSegment.from_file(mp3_path)
    audio = audio.set_channels(1).set_frame_rate(16000)
    audio.export(out_wav, format="wav")
    mp3_path.unlink(missing_ok=True)
    return out_wav

async def download_image(url: str, out_path: Path):
    async with aiohttp.ClientSession() as sess:
        async with sess.get(url) as r:
            if r.status != 200:
                raise HTTPException(400, f"下載 avatar 失敗: {r.status}")
            data = await r.read()
    async with aiofiles.open(out_path, "wb") as f:
        await f.write(data)
    return out_path

def generate_talking_video_cpu(image_path: Path, wav_path: Path, out_mp4: Path, fps: int = 30):
    """
    Phase 1：CPU 友善的「近似嘴型動畫」：
    - 讀入單張頭像，估計嘴部區域（圖中間偏下 35%~75% 高度帶），
    - 依音量能量對該區域做輕微位移/伸縮，產生說話節奏感。
    - 效果穩定低延遲，無需 GPU / 大模型。
    - 未來開啟 Wav2Lip 時會改用真·唇形同步（見下方 TODO）。
    """
    # 讀音訊並計算每幀的音量 RMS
    audio = AudioSegment.from_wav(wav_path)
    frame_ms = 1000 / fps
    n_frames = int(math.ceil(len(audio) / frame_ms))
    # 正規化音量 -> 0~1
    rms_series = []
    for i in range(n_frames):
        seg = audio[i*frame_ms:(i+1)*frame_ms]
        if len(seg) == 0:
            rms_series.append(0.0); continue
        rms = seg.rms
        rms_series.append(rms)
    rms_arr = np.array(rms_series, dtype=np.float32)
    if rms_arr.max() > 0:
        rms_arr = rms_arr / rms_arr.max()
    else:
        rms_arr[:] = 0

    # 準備影格
    img = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if img is None:
        raise HTTPException(400, "無法讀取 avatar 圖片")
    h, w = img.shape[:2]
    base = img.copy()

    # 嘴部 ROI：以中下方區域近似（保守不失真）
    y1, y2 = int(h*0.55), int(h*0.85)
    x1, x2 = int(w*0.28), int(w*0.72)
    mouth_h = y2 - y1

    frames = []
    for amp in rms_arr:
        # 根據音量做 0~8 像素的垂直開闔與 0~2% 的縮放
        shift = int(amp * 8)            # 下巴微張
        scale = 1.0 + amp * 0.02        # 輕微放大
        canvas = base.copy()

        # mouth 區塊位移
        mouth = base[y1:y2, x1:x2].copy()
        M = np.float32([[1, 0, 0], [0, 1, shift]])
        mouth_shifted = cv2.warpAffine(mouth, M, (x2-x1, y2-y1), borderMode=cv2.BORDER_REFLECT)
        canvas[y1:y2, x1:x2] = mouth_shifted

        # 輕微整體縮放（模擬說話帶動）
        M2 = cv2.getRotationMatrix2D((w/2, h/2), 0, scale)
        canvas = cv2.warpAffine(canvas, M2, (w, h), borderMode=cv2.BORDER_REFLECT)
        frames.append(cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB))

    # 以 imageio-ffmpeg 輸出 mp4 並混音
    # 先輸出純影像 mp4
    iio.imwrite(out_mp4, frames, fps=fps, codec="libx264", quality=7)
    # 以 ffmpeg 合成音訊（imageio 會使用系統 ffmpeg）
    # 重新寫入：讀回 video，疊加 audio
    tmp_mp4 = out_mp4.with_suffix(".tmp.mp4")
    os.rename(out_mp4, tmp_mp4)
    # ffmpeg 指令：-shortest 以較短者結束，避免黑屏尾巴
    os.system(f'ffmpeg -y -i "{tmp_mp4}" -i "{wav_path}" -c:v copy -c:a aac -shortest "{out_mp4}" -loglevel error')
    tmp_mp4.unlink(missing_ok=True)
    return out_mp4, len(audio)

async def ensure_wav2lip_and_run(image_path: Path, wav_path: Path, out_mp4: Path):
    """
    Phase 2（選配）：真·唇形同步。需要下載 Wav2Lip 權重與依賴，CPU 可跑但較慢。
    為了保持本文精簡，這裡僅示意：若偵測到模型檔即走 Wav2Lip，否則 fallback 至 Phase 1。
    """
    model_flag = os.getenv("WAV2LIP_ENABLED", "false").lower() == "true"
    model_file = Path(os.getenv("WAV2LIP_MODEL", "models/wav2lip_gan.pth"))
    if model_flag and model_file.exists():
        # TODO: 這裡可換成你專案內的 Wav2Lip 推論包裝呼叫。
        # 例如：subprocess.run(["python","infer_wav2lip.py","--face",image_path,"--audio",wav_path,"--outfile",out_mp4])
        # 為讓你先能上線，先回落到 Phase 1。
        pass
    # 使用 Phase 1 方案
    return generate_talking_video_cpu(image_path, wav_path, out_mp4)

@router.post("/talk")
async def talk_once(payload: TalkIn):
    text = payload.text.strip()
    if not text:
        raise HTTPException(400, "text 不可為空")

    uid = uuid.uuid4().hex[:8]
    img_path = MEDIA_DIR / f"avatar_{uid}.png"
    wav_path = MEDIA_DIR / f"tts_{uid}.wav"
    mp4_path = MEDIA_DIR / f"clip_{uid}.mp4"

    # 1) 抓圖
    await download_image(payload.avatar_url, img_path)
    # 2) 文字轉語音
    await tts_to_wav(text, payload.voice, payload.speed, wav_path)
    # 3) 產生影片（CPU 近似嘴型 / 或 Wav2Lip）
    mp4_path, duration_ms = await ensure_wav2lip_and_run(img_path, wav_path, mp4_path)

    video_url = f"/static/media/{mp4_path.name}"
    audio_url = f"/static/media/{wav_path.name}"

    return JSONResponse({
        "ok": True,
        "video_url": video_url,
        "audio_url": audio_url,
        "duration_ms": duration_ms
    })
