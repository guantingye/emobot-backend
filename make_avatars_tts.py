# make_avatars_tts.py
# 1) pip install --upgrade openai
# 2) 設定環境變數 OPENAI_API_KEY
# 3) python make_avatars_tts.py

import os
import base64
import inspect
from pathlib import Path
from openai import OpenAI

MODEL = "gpt-4o-mini-tts"
OUTDIR = Path("tts_out")

TEXTS = {
    "Lumi_fable": "嗨，我是 Lumi。擅長以溫和提問與情緒標記，先幫你把心裡的感受釐清。我們會一起辨認此刻最強烈的情緒與需求，練習接住自己，而不是急著解決。若你正面臨孤獨、失落、自我懷疑、關係拉扯或分手復原，我會做你可依靠的同行者。從一口深呼吸開始，把速度放慢；有我在，情緒是可以被看見、也能被照顧的。",
    "Solin_alloy": "你好，我是 Solin。我會帶你回望生命片段，從關係脈絡與反覆出現的模式，找出情緒背後真正的需求。透過精準而溫柔的回饋，我們把卡住的位置說清楚，讓改變有方向。若你對自我價值困惑、承載創傷記憶、在人際裡一再受傷，或想理解夢與空缺，我會陪你把「為什麼」看見。當你能理解自己，選擇就不再只是反應，而是有意識的前行。",
    "Niko_nova": "嗨，我是 Niko。和我合作會很務實：先釐清目標，再拆解阻礙，設定可以快速開始的小步行動。每一步都可追蹤、可調整。面對職場壓力、溝通衝突、時間管理、決策卡關，我會用清單、優先序與回饋週期幫你前進。先從最小可行的改變開始，把可控的事做好；當行動啟動，動能就會回來，你也會更靠近想要的生活。",
    "Clara_shimmer": "你好，我是 Clara。我擅長以認知行為取向，幫你辨識自動思考與認知偏誤，找出證據，重建更貼近現實與價值的觀點，並搭配行為實驗與日常練習穩定情緒。若你常陷入焦慮、負面自我對話、完美主義或拖延，我會提供清晰表單、逐步指引與回家作業。當你能看見念頭如何影響情緒與行為，就能把腦中的結解開來，重新找回掌控感。",
}

VOICE = {
    "Lumi_fable": "fable",
    "Solin_alloy": "alloy",
    "Niko_nova": "nova",
    "Clara_shimmer": "shimmer",
}

def _write_bytes(data: bytes, out_path: Path) -> None:
    OUTDIR.mkdir(parents=True, exist_ok=True)
    with open(out_path, "wb") as f:
        f.write(data)

def _gentle_kwargs(speech_create_func) -> dict:
    """
    讓聲線更溫柔：若 SDK 支援，就帶入更慢的語速。
    - 目前常見參數：speed（浮點數，1.0=原速）
    - 若該參數不存在，會自動忽略以確保相容。
    """
    sig = inspect.signature(speech_create_func)
    extra = {}
    if "speed" in sig.parameters:
        extra["speed"] = 0.92  # 微慢、更柔和
    return extra

def synthesize(client: OpenAI, voice: str, text: str, out_path: Path) -> None:
    """
    相容各版本 openai-python：
    - 不用 streaming
    - 偵測 'format' 或 'response_format'
    - 若支援 'speed'，自動帶入 0.92（更溫柔）
    """
    create_func = client.audio.speech.create
    sig = inspect.signature(create_func)

    kwargs = {
        "model": MODEL,
        "voice": voice,
        "input": text,
        **_gentle_kwargs(create_func),
    }
    if "format" in sig.parameters:
        kwargs["format"] = "mp3"
    elif "response_format" in sig.parameters:
        kwargs["response_format"] = "mp3"

    resp = create_func(**kwargs)

    # 盡量涵蓋不同回傳型態
    if hasattr(resp, "read"):
        data = resp.read()
    elif hasattr(resp, "content"):
        data = resp.content
    elif isinstance(resp, (bytes, bytearray)):
        data = bytes(resp)
    elif isinstance(resp, dict):
        if "audio" in resp:
            data = base64.b64decode(resp["audio"])
        elif "data" in resp and isinstance(resp["data"], (bytes, bytearray)):
            data = bytes(resp["data"])
        else:
            raise RuntimeError("Unsupported response dict shape.")
    else:
        raise RuntimeError(f"Unsupported response type: {type(resp)}")

    _write_bytes(data, out_path)

def main() -> None:
    if not os.getenv("OPENAI_API_KEY"):
        raise SystemExit("🔑 請先設定環境變數 OPENAI_API_KEY")
    client = OpenAI()
    for key, text in TEXTS.items():
        out_file = OUTDIR / f"{key}.mp3"
        synthesize(client, VOICE[key], text, out_file)
        print(f"✅ Saved: {out_file}")

if __name__ == "__main__":
    main()
