# app/chat.py
# -*- coding: utf-8 -*-
from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from sqlalchemy.orm import Session
from time import perf_counter
import os

from app.db.session import get_db, engine
from app.db.base import Base
# 注意：不再強制依賴登入；仍然保留 import 以便你之後要改回來
# from app.core.security import get_current_user
from app.models.chat import ChatMessage

router = APIRouter(prefix="/api/chat", tags=["chat"])

# 建表（已存在會略過）
try:
    Base.metadata.create_all(bind=engine)
except Exception as e:
    print("ChatMessage create_all skipped:", e)

# ================= Persona =================
PERSONA_TEMPLATE = """你是 {name}（{persona_zh}）。你的目標是以{tone}陪伴使用者，幫助TA{goal}。
【溝通語言】以繁體中文回覆；若使用者先用其他語言，則跟隨之。
【對話原則】
- 先理解與肯認，再回應；用簡潔、溫柔、無批判的字句。
- 避免醫療診斷與貼標；不給藥物或危險行為指示。
- 善用重述與提問，但**每次最多1個問題**。
- 內容以1~3段為限；每段2~3句。必要時可用條列（最多3點）。
- 若偵測到危機詞彙（立即傷害/自傷/他傷/求救），先肯認感受，再建議尋求當地緊急協助或可信任的人，即刻求助。不要提供具體自傷方法。

【輸出格式（固定）】
請以以下段落輸出（若某段不適用可以省略第二段）：
1) 回應：若干句自然文字。
2) 可以一起做（最多3點，沒有就省略）：以「•」條列可行的小步驟。
3) 問一個問題（一定只有1句）：幫助下一步聚焦的單一問題。

【此型風格加成】
{style_delta}
"""

PERSONA_STYLES: Dict[str, Dict[str, str]] = {
    "empathy":  {"name":"Lumi","persona_zh":"同理型 AI","tone":"共感、安撫與承接情緒的方式","goal":"被好好地聽見與理解",
                 "style_delta":"優先進行情緒標記與肯認；避免過度分析；條列以自我照顧/緩和為主。"},
    "insight":  {"name":"Solin","persona_zh":"洞察型 AI","tone":"澄清、重述與蘇格拉底式提問的方式","goal":"釐清線索、看見關鍵與新觀點",
                 "style_delta":"先用1句重述，再提出1個開放式問題；條列聚焦『關係/假設/需要更多資訊』。"},
    "solution": {"name":"Niko","persona_zh":"解決型 AI","tone":"務實、具體、鼓勵而不強迫的方式","goal":"把感受轉成可行的小步驟",
                 "style_delta":"每次提供1~3個『10分鐘內、低風險』的小步驟；語氣可量化、簡短。"},
    "cognitive":{"name":"Clara","persona_zh":"認知型 AI","tone":"CBT 風格、結構化與溫柔同理的方式","goal":"辨識自動想法與認知偏誤，形成替代想法",
                 "style_delta":"可用『自動想法→證據/反證→替代想法』的文字條列；引導而非糾正。"},
}
ENV_PROMPT_KEYS = {"empathy":"EMOBOT_PROMPT_LUMI","insight":"EMOBOT_PROMPT_SOLIN",
                   "solution":"EMOBOT_PROMPT_NIKO","cognitive":"EMOBOT_PROMPT_CLARA"}

def _persona_prompt(bot_type: str, mode: str, video_meta: Optional[Dict[str, Any]]) -> str:
    env_key = ENV_PROMPT_KEYS.get(bot_type)
    if env_key and os.getenv(env_key):
        base = os.getenv(env_key)  # 雲端覆蓋
    else:
        meta = PERSONA_STYLES.get(bot_type, PERSONA_STYLES["solution"])
        base = PERSONA_TEMPLATE.format(**meta)
    if mode == "video":
        hint = "【影像模式補充】若 video_meta 有非語言線索（表情/停頓/情緒），請先以1句話溫柔回應，再照固定輸出格式回覆。"
        if video_meta: hint += f" 目前 video_meta: {video_meta}。"
        base = base + "\n\n" + hint
    return base

# ================= Schemas =================
class HistoryItem(BaseModel):
    role: str
    content: str

class SendPayload(BaseModel):
    bot_type: str = Field(..., pattern="^(empathy|insight|solution|cognitive)$")
    mode: str = Field("text", pattern="^(text|video)$")
    message: str
    history: List[HistoryItem] = []
    demo: bool = False
    video_meta: Optional[Dict[str, Any]] = None

class SendResult(BaseModel):
    ok: bool
    reply: Optional[str] = None
    bot: Optional[Dict[str, Any]] = None
    error: Optional[str] = None

# ================= DB helper =================
def _insert(db: Session, user_id: int, bot_type: str, mode: str, role: str, content: str, meta: Dict[str, Any] | None = None):
    m = ChatMessage(user_id=user_id, bot_type=bot_type, mode=mode, role=role, content=content, meta=meta or {})
    db.add(m); db.commit(); db.refresh(m)
    return m

# ================= OpenAI =================
def _openai_chat(messages: List[Dict[str, str]], model: str, temperature: float, max_tokens: int) -> str:
    api_key = os.getenv("OPENAI_API_KEY", "")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY not set")

    # 優先新 SDK
    try:
        from openai import OpenAI  # type: ignore
        client = OpenAI(api_key=api_key, base_url=os.getenv("OPENAI_BASE_URL") or None)
        resp = client.chat.completions.create(model=model, messages=messages, temperature=temperature, max_tokens=max_tokens)
        return resp.choices[0].message.content or ""
    except Exception as e_new:
        print("[openai>=1.x] failed:", e_new)

    # 回退舊 SDK
    try:
        import openai  # type: ignore
        openai.api_key = api_key
        if os.getenv("OPENAI_BASE_URL"): openai.api_base = os.getenv("OPENAI_BASE_URL")
        resp = openai.ChatCompletion.create(model=model, messages=messages, temperature=temperature, max_tokens=max_tokens)
        return resp["choices"][0]["message"]["content"].strip()
    except Exception as e_old:
        print("[openai<1] failed:", e_old)
        raise

def _gpt_reply(bot_type: str, mode: str, video_meta: Optional[Dict[str, Any]],
               history: List[HistoryItem], user_message: str) -> Dict[str, Any]:
    system = _persona_prompt(bot_type, mode, video_meta)
    model  = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    temp   = float(os.getenv("OPENAI_TEMPERATURE", "0.7"))
    max_tk = int(os.getenv("OPENAI_MAX_TOKENS", "600"))

    msgs = [{"role": "system", "content": system}]
    for h in history[-10:]:
        role = "assistant" if h.role == "assistant" else "user"
        msgs.append({"role": role, "content": h.content})
    msgs.append({"role": "user", "content": user_message})

    t0 = perf_counter()
    text = _openai_chat(msgs, model=model, temperature=temp, max_tokens=max_tk)
    latency_ms = int((perf_counter() - t0) * 1000)

    return {"text": (text or "").strip(),
            "meta": {"provider":"openai","model":model,"temperature":temp,
                     "max_tokens":max_tk,"latency_ms":latency_ms,"mode":mode,"video_meta":video_meta}}

# ================= Routes =================
@router.post("/send", response_model=SendResult)
def send_chat(payload: SendPayload, request: Request, db: Session = Depends(get_db)):
    """
    不論文字/影像，一律走 OpenAI。
    不再強制驗證；若無登入，使用 X-User-Id 或 0 當匿名使用者。
    """
    bot_type, mode = payload.bot_type, payload.mode
    user_msg = (payload.message or "").strip()
    if not user_msg:
        raise HTTPException(status_code=400, detail="Empty message")

    # 盡可能取得 user_id：先讀自訂標頭，失敗就 0（匿名）
    user_id_hdr = request.headers.get("X-User-Id")
    try:
        user_id = int(user_id_hdr) if user_id_hdr is not None else 0
    except ValueError:
        user_id = 0

    # 入庫使用者訊息
    _insert(db, user_id=user_id, bot_type=bot_type, mode=mode, role="user",
            content=user_msg, meta={"demo": payload.demo, "video_meta": payload.video_meta})

    try:
        out = _gpt_reply(bot_type, mode, payload.video_meta, payload.history, user_msg)
        reply_text, reply_meta, error_text = out["text"], out["meta"], None
    except Exception as e:
        # 最小保底
        defaults = {
            "empathy":  "我在這裡，先一起做個小小的深呼吸。此刻最強烈的感受是什麼？",
            "insight":  "如果把這件事分成兩三個片段，你會從哪一段開始說起？",
            "solution": "我們可以先定一個10分鐘內的小步驟。你想從哪個最容易著手？",
            "cognitive":"把剛剛的自動想法寫下一句，列出一個證據與反證，再試著找個更溫和的替代說法。"
        }
        reply_text = defaults.get(bot_type, defaults["solution"])
        reply_meta = {"provider":"fallback","persona":bot_type,"mode":mode}
        error_text = f"{type(e).__name__}: {str(e)[:180]}"

    # 入庫 AI 回覆
    _insert(db, user_id=user_id, bot_type=bot_type, mode=mode, role="ai", content=reply_text, meta=reply_meta)

    name_map = {"empathy":"Lumi","insight":"Solin","solution":"Niko","cognitive":"Clara"}
    return SendResult(ok=True, reply=reply_text, bot={"type":bot_type, "name":name_map.get(bot_type)}, error=error_text)

# 健康檢查（GET/POST 都可）
@router.get("/health/openai")
@router.post("/health/openai")
def health_openai():
    model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    base  = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
    key   = os.getenv("OPENAI_API_KEY")
    info = {"model": model, "base_url": base, "has_key": bool(key), "sdk": None, "ok": False, "error": None}
    if not key:
        info["error"] = "OPENAI_API_KEY not set"; return info
    try:
        from openai import OpenAI  # type: ignore
        client = OpenAI(api_key=key, base_url=os.getenv("OPENAI_BASE_URL") or None)
        r = client.chat.completions.create(model=model, messages=[{"role":"user","content":"ping"}], max_tokens=4)
        info["sdk"] = ">=1.x"; info["ok"] = bool(r.choices); info["error"]=None; return info
    except Exception as e_new:
        info["error"] = f"[new]{type(e_new).__name__}:{str(e_new)[:120]}"
    try:
        import openai  # type: ignore
        openai.api_key = key
        if os.getenv("OPENAI_BASE_URL"): openai.api_base = os.getenv("OPENAI_BASE_URL")
        r = openai.ChatCompletion.create(model=model, messages=[{"role":"user","content":"ping"}], max_tokens=4)
        info["sdk"] = "<1.0"; info["ok"] = True; info["error"]=None
    except Exception as e_old:
        info["error"] = f"{info['error']} | [old]{type(e_old).__name__}:{str(e_old)[:120]}"
    return info
