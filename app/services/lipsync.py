# services/lipsync.py
import os
import cv2
import tempfile
import numpy as np
import requests
from moviepy.editor import VideoFileClip, AudioFileClip
from pathlib import Path

# Wave2Lip 推論（使用官方/社群封裝）
# 這裡示意性地呼叫一個封裝函式 run_wav2lip(image_path, audio_path, out_mp4)
# 你可以把官方 inference 程式碼整合進來，或用現成封裝（如 pip install Wav2Lip）
from wav2lip_infer import run_wav2lip  # ← 請在同專案放入一個簡單封裝或直接引官方 inference

def _download(url: str, dest_path: str):
    r = requests.get(url, timeout=60)
    r.raise_for_status()
    with open(dest_path, "wb") as f:
        f.write(r.content)

def generate_lipsync_video(image_url: str, audio_path: str, mp4_out: str):
    os.makedirs(os.path.dirname(mp4_out), exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp:
        img_path = os.path.join(tmp, "avatar.jpg")
        _download(image_url, img_path)

        # 1) Wave2Lip 產出（含嘴型）的 MP4 (無聲 / 有聲皆可，這裡直接帶入聲音）
        run_wav2lip(image_path=img_path, audio_path=audio_path, out_path=mp4_out)

        if not os.path.exists(mp4_out):
            raise RuntimeError("Wave2Lip failed to create output video")

        # 可選：二次封裝或壓縮（moviepy 重新封裝避免編碼相容問題）
        # 這段通常可省略，除非你的環境需要特定容器設定
        try:
            v = VideoFileClip(mp4_out)
            a = AudioFileClip(audio_path)
            final = v.set_audio(a)
            tmp_out = os.path.join(tmp, "final.mp4")
            final.write_videofile(tmp_out, fps=25, codec="libx264", audio_codec="aac", verbose=False, logger=None)
            os.replace(tmp_out, mp4_out)
        except Exception:
            # 若 moviepy 在環境失敗，保留原檔
            pass
