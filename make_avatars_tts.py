# make_avatars_tts.py
# 1) pip install --upgrade openai python-dotenv
# 2) 設定 OPENAI_API_KEY 或使用 --key / --env-file 參數
# 3) python make_avatars_tts.py [--key sk-xxxx] [--env-file .env]

import os
import re
import base64
import inspect
import argparse
from pathlib import Path
from typing import Dict
from dotenv import load_dotenv, find_dotenv
from openai import OpenAI

MODEL = "gpt-4o-mini-tts"
OUTDIR = Path("tts_out")

# === Persona 聲線設定 ===
# Niko/Solin 維持原設定；Clara/Lumi 改為 nova，並微調節奏與停頓以貼合目標聲感
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

# === 影片開場白腳本（可自行替換） ===
SCRIPTS: Dict[str, str] = {
    "empathy": (
        "嗨，我是 Lumi。擅長以溫和提問與情緒標記，先幫你把心裡的感受釐清。"
        "我們會一起辨認此刻最強烈的情緒與需求，練習接住自己，而不是急著解決。"
        "若你正面臨孤獨、失落、自我懷疑、關係拉扯或分手復原，我會做你可依靠的同行者。"
        "從一口深呼吸開始，把速度放慢；有我在，情緒是可以被看見、也能被照顧的。"
    ),
    "insight": (
        "你好，我是 Solin。我會帶你回望生命片段，從關係脈絡與反覆出現的模式，找出情緒背後真正的需求。"
        "透過精準而溫柔的回饋，我們把卡住的位置說清楚，讓改變有方向。"
        "若你對自我價值困惑、承載創傷記憶、在人際裡一再受傷，或想理解夢與空缺，我會陪你把「為什麼」看見。"
        "當你能理解自己，選擇就不再只是反應，而是有意識的前行。"
    ),
    "solution": (
        "嗨，我是 Niko。和我合作會很務實：先釐清目標，再拆解阻礙，設定可以快速開始的小步行動。"
        "每一步都可追蹤、可調整。"
        "面對職場壓力、溝通衝突、時間管理、決策卡關，我會用清單、優先序與回饋週期幫你前進。"
        "先從最小可行的改變開始，把可控的事做好；當行動啟動，動能就會回來，你也會更靠近想要的生活。"
    ),
    "cognitive": (
        "你好，我是 Clara。我擅長以認知行為取向，幫你辨識自動思考與認知偏誤，找出證據，重建更貼近現實與價值的觀點，"
        "並搭配行為實驗與日常練習穩定情緒。"
        "若你常陷入焦慮、負面自我對話、完美主義或拖延，我會提供清晰表單、逐步指引與回家作業。"
        "當你能看見念頭如何影響情緒與行為，就能把腦中的結解開來，重新找回掌控感。"
    ),
}

# ---------- Prosody helpers ----------
PUNCS_SENTENCE = "。！？"
PUNCS_CLAUSE = "，、；："

def _apply_prosody(text: str, pause_factor: float, energy: float) -> str:
    # 以標點插入空白/省略號，模擬更柔或更俐落的停頓節奏
    clause_pad = 1 if pause_factor < 1.2 else (2 if pause_factor < 1.35 else 3)
    sent_pad = clause_pad + 1
    def pad_spaces(n: int) -> str: return " " * max(0, n)
    def repl_sentence(m):
        ch = m.group(0)
        suffix = "…" if pause_factor >= 1.25 else ""
        return f"{ch}{suffix}{pad_spaces(sent_pad)}"
    def repl_clause(m):
        ch = m.group(0)
        suffix = "…" if pause_factor >= 1.35 or energy <= 0.85 else ""
        return f"{ch}{suffix}{pad_spaces(clause_pad)}"
    text = re.sub(f"[{PUNCS_SENTENCE}]", repl_sentence, text)
    text = re.sub(f"[{PUNCS_CLAUSE}]", repl_clause, text)
    text = re.sub(r"([。！？])([^ \n])", r"\1  \2", text)
    return text

# ---------- OpenAI helpers ----------
def _choose_audio_kwargs(speech_create_func, prefer_format="mp3", speed: float = None) -> dict:
    # 相容不同 SDK 版本：format / response_format / speed
    sig = inspect.signature(speech_create_func)
    kwargs = {}
    if "format" in sig.parameters:
        kwargs["format"] = prefer_format
    elif "response_format" in sig.parameters:
        kwargs["response_format"] = prefer_format
    if speed is not None and "speed" in sig.parameters:
        kwargs["speed"] = float(speed)
    return kwargs

def _synthesize(client: OpenAI, *, model: str, voice: str, text: str, speaking_rate: float) -> bytes:
    create_func = client.audio.speech.create
    kwargs = {
        "model": model,
        "voice": voice,
        "input": text,
        **_choose_audio_kwargs(create_func, prefer_format="mp3", speed=speaking_rate),
    }
    resp = create_func(**kwargs)
    if hasattr(resp, "read"): return resp.read()
    if hasattr(resp, "content"): return resp.content
    if isinstance(resp, (bytes, bytearray)): return bytes(resp)
    if isinstance(resp, dict):
        if "audio" in resp: return base64.b64decode(resp["audio"])
        if "data" in resp and isinstance(resp["data"], (bytes, bytearray)): return bytes(resp["data"])
    raise RuntimeError(f"Unsupported response type: {type(resp)}")

def _write_bytes(data: bytes, out_path: Path) -> None:
    OUTDIR.mkdir(parents=True, exist_ok=True)
    with open(out_path, "wb") as f:
        f.write(data)

# ---------- Key loading ----------
def _load_key(args) -> None:
    # 優先序：--key > 環境變數 > --env-file/.env
    if args.key:
        os.environ["OPENAI_API_KEY"] = args.key.strip()
        return
    if args.env_file:
        load_dotenv(args.env_file, override=False)
    else:
        load_dotenv(find_dotenv(usecwd=True), override=False)

# ---------- Main ----------
def main() -> None:
    parser = argparse.ArgumentParser(description="Persona TTS generator")
    parser.add_argument("--key", help="OpenAI API Key（可選）")
    parser.add_argument("--env-file", help="指定 .env 檔路徑（可選）")
    args = parser.parse_args()

    _load_key(args)
    if not os.getenv("OPENAI_API_KEY"):
        raise SystemExit("🔑 請先設定 OPENAI_API_KEY，或使用 --key sk-xxxx，或提供 --env-file .env")

    client = OpenAI()

    for code, style in PERSONA_STYLES.items():
        name = style["name"]
        voice = style["voice"]
        rate = style["speaking_rate"]
        pause = style["pause_factor"]
        energy = style["energy"]

        raw_text = SCRIPTS[code]
        text = _apply_prosody(raw_text, pause_factor=pause, energy=energy)

        audio_bytes = _synthesize(
            client,
            model=MODEL,
            voice=voice,
            text=text,
            speaking_rate=rate,
        )

        out_path = OUTDIR / f"{name}_{voice}.mp3"
        _write_bytes(audio_bytes, out_path)
        print(f"✅ Saved: {out_path}  (rate={rate}, pause={pause}, energy={energy})")

if __name__ == "__main__":
    main()
