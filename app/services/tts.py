# services/tts.py
import os
import wave
import contextlib
import pyttsx3
import subprocess

def synthesize_tts(text: str, wav_out: str,
                   speaker: str = "female_zh",
                   use_piper: bool = False,
                   piper_model: str = "zh-TW-WeiChung-Hsiao-NEON"):
    """
    預設 pyttsx3（最穩、無外網依賴）；若 USE_PIPER=1 則使用 Piper CLI（需放好可執行檔與模型）。
    """
    os.makedirs(os.path.dirname(wav_out), exist_ok=True)

    if use_piper:
        model_dir = os.getenv("PIPER_MODEL_DIR", "piper_models")
        model_path = os.path.join(model_dir, f"{piper_model}.onnx")
        config_path = os.path.join(model_dir, f"{piper_model}.json")
        if not (os.path.exists(model_path) and os.path.exists(config_path)):
            raise RuntimeError("Piper 模型未找到：請把 .onnx 與 .json 放到 PIPER_MODEL_DIR")
        cmd = ["piper", "--model", model_path, "--config", config_path, "--output_file", wav_out]
        p = subprocess.run(cmd, input=text.encode("utf-8"),
                           stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        if p.returncode != 0 or not os.path.exists(wav_out):
            raise RuntimeError(f"Piper 產生語音失敗：{p.stderr.decode('utf-8','ignore')[:300]}")
        return

    # ----------------- pyttsx3 -----------------
    engine = pyttsx3.init()
    # 嘗試選用中文 voice（不同平台 voice 參數不同，以下是容錯寫法）
    chosen = None
    for v in engine.getProperty("voices"):
        name = (getattr(v, "name", "") or "").lower()
        langs = ",".join(getattr(v, "languages", []) or []).lower()
        if "zh" in langs or "chinese" in name or "中文" in name:
            chosen = v.id
            break
    if chosen:
        engine.setProperty("voice", chosen)
    engine.setProperty("rate", 170)
    engine.save_to_file(text, wav_out)
    engine.runAndWait()

    if not os.path.exists(wav_out):
        raise RuntimeError("pyttsx3 產生語音失敗")

    # 確認不是空檔
    with contextlib.closing(wave.open(wav_out, "r")) as wf:
        if wf.getnframes() == 0:
            raise RuntimeError("TTS 輸出為空音訊")
