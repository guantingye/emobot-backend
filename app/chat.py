# app/chat.py - HeyGen Streaming API integration router
# 完整覆蓋版本（可直接使用），將所有 HeyGen 端點註冊在 /api/chat/heygen/*
# 變更重點：
# 1) Pydantic 模型新增 voice 欄位的防呆 validator：若前端誤傳字串會自動包成 {"voice_id": "..."}
# 2) 嚴格遵循 HeyGen v1 Streaming API 端點與參數
# 3) 提供健康檢查與背景任務工具函式
#
# 依賴：
# - aiohttp
# - fastapi
# - pydantic v2+
#
# 環境變數：
# - HEYGEN_API_KEY (必要)
# - HEYGEN_AVATAR_ID (選用；若未提供則前端需提供 avatar_id)
#
# 預期前端呼叫：
# POST /api/chat/heygen/create_session    { "avatar_id": "...", "voice": {"voice_id":"zh-TW-HsiaoChenNeural"}, "quality":"medium", "version":"v2" }
# POST /api/chat/heygen/start_session     { "session_id": "..." , "sdp": "<optional sdp>" }
# POST /api/chat/heygen/send_text         { "session_id": "...", "text": "你好", "task_type":"repeat", "task_mode":"sync" }
# POST /api/chat/heygen/close_session     { "session_id": "..." }
# GET  /api/chat/health/heygen            -> 檢查設定與路由是否正確

import os
import logging
from typing import Any, Dict, Optional, Literal

import aiohttp
from fastapi import APIRouter, BackgroundTasks, HTTPException
from pydantic import BaseModel, Field, field_validator

logger = logging.getLogger(__name__)

# ---------- 常數與設定 ----------
HEYGEN_API_KEY = os.getenv("HEYGEN_API_KEY", "").strip()
DEFAULT_AVATAR_ID = os.getenv("HEYGEN_AVATAR_ID", "").strip()
HEYGEN_BASE = "https://api.heygen.com/v1"

if not HEYGEN_API_KEY:
    logger.warning("HEYGEN_API_KEY 未設定，HeyGen 端點將無法正常工作。")

# FastAPI Router：統一掛載在 /api/chat
router = APIRouter(prefix="/api/chat", tags=["chat", "heygen"])


# ---------- Pydantic 模型 ----------
class HeyGenTokenRequest(BaseModel):
    """創建 access token 不需要參數（保留型別定義以利日後擴充）"""
    pass


class HeyGenSessionRequest(BaseModel):
    avatar_id: Optional[str] = Field(default=None, description="HeyGen Avatar ID（可由環境變數 HEYGEN_AVATAR_ID 提供預設）")
    voice: Optional[Dict[str, Any]] = Field(default=None, description="例如 {'voice_id': 'zh-TW-HsiaoChenNeural'} 或字串 'zh-TW-HsiaoChenNeural'")
    quality: Literal["low", "medium", "high"] = Field(default="medium")
    version: str = Field(default="v2")

    # 防呆：若傳入是字串就自動包成 {'voice_id': '<字串>'}
    @field_validator("voice", mode="before")
    @classmethod
    def _coerce_voice(cls, v):
        if v is None:
            return v
        if isinstance(v, str):
            return {"voice_id": v}
        if isinstance(v, dict):
            # 允許 {voice_id:...} 或 {id/name:...}
            if "voice_id" in v:
                return v
            vid = v.get("id") or v.get("name")
            return {"voice_id": vid} if vid else v
        raise ValueError("voice 必須是字串 voice_id 或 dict，如 {'voice_id': '...'}")


class HeyGenStartRequest(BaseModel):
    session_id: str
    sdp: Optional[str] = Field(default=None, description="WebRTC SDP（若使用 WebRTC 播放）")


class HeyGenTextRequest(BaseModel):
    session_id: str
    text: str = Field(min_length=1)
    task_type: Literal["repeat"] = Field(default="repeat", description="任務型別，目前以 repeat 為主")
    task_mode: Literal["sync", "async"] = Field(default="sync", description="播放模式")


class HeyGenStopRequest(BaseModel):
    session_id: str


class HeyGenResponse(BaseModel):
    success: bool
    session_id: Optional[str] = None
    access_token: Optional[str] = None
    url: Optional[str] = None
    data: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    code: Optional[int] = None  # 透傳 HeyGen 的 code（例如 100）


# ---------- 內部工具 ----------
def _hg_headers() -> Dict[str, str]:
    return {
        "Content-Type": "application/json",
        "x-api-key": HEYGEN_API_KEY,
    }


async def _hg_post_json(path: str, payload: Dict[str, Any]) -> Dict[str, Any]:
    url = f"{HEYGEN_BASE.rstrip('/')}/{path.lstrip('/')}"
    timeout = aiohttp.ClientTimeout(total=30)
    async with aiohttp.ClientSession(timeout=timeout) as session:
        async with session.post(url, headers=_hg_headers(), json=payload) as resp:
            text = await resp.text()
            try:
                data = await resp.json()
            except Exception:
                logger.error("HeyGen 回應非 JSON：status=%s, text=%s", resp.status, text)
                raise HTTPException(status_code=502, detail=f"HeyGen 非預期回應：{text}")
            if resp.status >= 400:
                logger.error("HeyGen HTTP %s：%s", resp.status, data)
                raise HTTPException(status_code=502, detail=data)
            return data


# ---------- 路由：健康檢查 ----------
@router.get("/health/heygen", response_model=HeyGenResponse)
async def heygen_health():
    ok = bool(HEYGEN_API_KEY)
    routes = [
        "/api/chat/heygen/create_session",
        "/api/chat/heygen/start_session",
        "/api/chat/heygen/send_text",
        "/api/chat/heygen/close_session",
    ]
    return HeyGenResponse(
        success=ok,
        data={
            "ok": ok,
            "base": HEYGEN_BASE,
            "routes": routes,
            "defaults": {
                "avatar_id": DEFAULT_AVATAR_ID or None,
            },
        },
    )


# ---------- 路由：建立 Session ----------
@router.post("/heygen/create_session", response_model=HeyGenResponse)
async def create_session(req: HeyGenSessionRequest):
    if not HEYGEN_API_KEY:
        raise HTTPException(status_code=500, detail="HEYGEN_API_KEY 未設定")

    avatar_id = (req.avatar_id or DEFAULT_AVATAR_ID or "").strip()
    if not avatar_id:
        raise HTTPException(status_code=400, detail="缺少 avatar_id，請於 body.avatar_id 或環境 HEYGEN_AVATAR_ID 設定")

    payload = {
        "avatar_id": avatar_id,
        "quality": req.quality,
        "version": req.version,
    }
    if req.voice:
        payload["voice"] = req.voice  # 這裡已由 validator 保證為 dict

    data = await _hg_post_json("streaming.new", payload)

    code = data.get("code")
    if code != 100:
        # HeyGen 錯誤（例如權限、配額、參數錯誤等）
        err = data.get("message") or data
        logger.error("HeyGen create_session 失敗：%s", err)
        raise HTTPException(status_code=502, detail=err)

    info = data.get("data") or {}
    return HeyGenResponse(
        success=True,
        code=code,
        session_id=info.get("session_id"),
        access_token=info.get("access_token"),
        url=info.get("url"),
        data=info,
    )


# ---------- 路由：啟動 Session（可選，用於 WebRTC/SDP） ----------
@router.post("/heygen/start_session", response_model=HeyGenResponse)
async def start_session(req: HeyGenStartRequest):
    if not HEYGEN_API_KEY:
        raise HTTPException(status_code=500, detail="HEYGEN_API_KEY 未設定")

    payload = {"session_id": req.session_id}
    if req.sdp:
        payload["sdp"] = req.sdp

    data = await _hg_post_json("streaming.start", payload)

    code = data.get("code")
    if code != 100:
        err = data.get("message") or data
        logger.error("HeyGen start_session 失敗：%s", err)
        raise HTTPException(status_code=502, detail=err)

    info = data.get("data") or {}
    return HeyGenResponse(success=True, code=code, data=info)


# ---------- 路由：送出文字任務（repeat/sync） ----------
@router.post("/heygen/send_text", response_model=HeyGenResponse)
async def send_text(req: HeyGenTextRequest):
    if not HEYGEN_API_KEY:
        raise HTTPException(status_code=500, detail="HEYGEN_API_KEY 未設定")

    payload = {
        "session_id": req.session_id,
        "task_type": req.task_type,
        "task_mode": req.task_mode,
        "input_text": req.text,
    }

    data = await _hg_post_json("streaming.task", payload)

    code = data.get("code")
    if code != 100:
        err = data.get("message") or data
        logger.error("HeyGen send_text 失敗：%s", err)
        raise HTTPException(status_code=502, detail=err)

    info = data.get("data") or {}
    return HeyGenResponse(success=True, code=code, data=info)


# ---------- 路由：停止 Session ----------
@router.post("/heygen/close_session", response_model=HeyGenResponse)
async def close_session(req: HeyGenStopRequest):
    if not HEYGEN_API_KEY:
        raise HTTPException(status_code=500, detail="HEYGEN_API_KEY 未設定")

    payload = {"session_id": req.session_id}
    data = await _hg_post_json("streaming.stop", payload)

    code = data.get("code")
    if code != 100:
        err = data.get("message") or data
        logger.error("HeyGen close_session 失敗：%s", err)
        raise HTTPException(status_code=502, detail=err)

    info = data.get("data") or {}
    return HeyGenResponse(success=True, code=code, data=info)


# ---------- 背景任務（供 /api/chat/send 產生 AI 回覆後呼叫） ----------
async def _bg_send_text_to_heygen(session_id: str, text: str):
    """背景播放 AI 文字，避免阻塞主流程。"""
    try:
        await send_text(HeyGenTextRequest(session_id=session_id, text=text))  # 復用上面端點邏輯
    except Exception as e:
        logger.exception("背景送 HeyGen 文字失敗：%s", e)


def enqueue_heygen_tts(background_tasks: BackgroundTasks, session_id: Optional[str], text: Optional[str]):
    """供外部 chat 邏輯呼叫：若有 session_id 與文字，就丟到 HeyGen repeat/sync 播放。"""
    if session_id and text:
        background_tasks.add_task(_bg_send_text_to_heygen, session_id, text)
