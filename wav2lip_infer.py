# wav2lip_infer.py
import os
import sys
import subprocess

def run_wav2lip(image_path: str, audio_path: str, out_path: str):
    """
    以子行程方式呼叫官方 Wav2Lip 的 inference.py。
    需求：
      - third_party/Wav2Lip/ 目錄存在
      - 目錄下有 Wav2Lip.pth 權重
    CPU/GPU：若部署有 CUDA，官方 inference 會自動用 GPU；沒有就走 CPU。
    """
    repo_dir = os.path.abspath(os.getenv("WAV2LIP_DIR", "third_party/Wav2Lip"))
    ckpt = os.path.join(repo_dir, "Wav2Lip.pth")
    infer = os.path.join(repo_dir, "inference.py")

    if not os.path.exists(infer):
        raise RuntimeError(f"inference.py not found at {infer}")
    if not os.path.exists(ckpt):
        raise RuntimeError(f"Wav2Lip.pth not found at {ckpt}")

    # 注意：pads / resize_factor 可按你的頭像素材做微調；這裡先給穩定值
    cmd = [
        sys.executable, infer,
        "--checkpoint_path", ckpt,
        "--face", image_path,
        "--audio", audio_path,
        "--outfile", out_path,
        "--pads", "0", "10", "0", "0",
        "--resize_factor", "2",
        "--nosmooth"
    ]

    p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if p.returncode != 0 or not os.path.exists(out_path):
        err = p.stderr.decode("utf-8", "ignore")
        raise RuntimeError(f"Wav2Lip inference failed: {err[:300]}")
