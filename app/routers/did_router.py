# app/routers/did_router.py
from __future__ import annotations
import os
import base64
from typing import Optional, Dict, Any

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
import httpx

DID_API_BASE = os.getenv("DID_API_BASE", "https://api.d-id.com")
DID_API_KEY = os.getenv("DID_API_KEY", "")
DID_SOURCE_URL_DEFAULT = os.getenv("DID_SOURCE_URL", "")  # 你的頭像圖片 URL
DID_VOICE_DEFAULT = os.getenv("DID_VOICE_ID", "zh-TW-HsiaoChenNeural")

if not DID_API_KEY:
    # 先不要在 import 階段就 raise，讓 /health 可以告知缺環境變數
    pass

router = APIRouter(prefix="/api/chat/did", tags=["did"])

def _auth_header() -> Dict[str, str]:
    """
    D-ID 為 Basic Auth，內容是 base64("API_KEY:")，冒號不可省略。
    """
    if not DID_API_KEY:
        raise HTTPException(status_code=500, detail="DID_API_KEY is not configured")
    token = base64.b64encode(f"{DID_API_KEY}:".encode("utf-8")).decode("utf-8")
    return {"Authorization": f"Basic {token}"}

class CreateTalkBody(BaseModel):
    text: str = Field(..., description="要轉成語音並生成嘴型同步影片的文字")
    voice_id: Optional[str] = Field(default=DID_VOICE_DEFAULT)
    source_url: Optional[str] = Field(default=None, description="頭像圖片 URL。不傳則用環境變數 DID_SOURCE_URL")
    config: Optional[Dict[str, Any]] = Field(
        default={"fluent": True, "pad_audio": 0.5},
        description="D-ID config 參數"
    )

@router.get("/health")
async def did_health():
    """
    輕量健康檢查：僅檢查環境變數是否設定齊全。
    若你想要真的 ping D-ID，可在此加上一次 GET，但為避免配額浪費先不打外部 API。
    """
    ok = bool(DID_API_KEY)
    return {"ok": ok, "has_api_key": ok, "voice_default": DID_VOICE_DEFAULT, "source_default": bool(DID_SOURCE_URL_DEFAULT)}

@router.post("/create_talk")
async def create_talk(body: CreateTalkBody):
    """
    建立一支 talk，回傳 talk_id。前端可用 talk_id 輪詢查詢狀態並取得 result_url。
    """
    if not DID_API_KEY:
        raise HTTPException(status_code=500, detail="DID_API_KEY is not configured")

    payload = {
        "script": {
            "type": "text",
            "input": body.text,
            # 同步支援兩種鍵，避免 D-ID 版本差異（某些文件用 provider、某些用 voice）
            "provider": {"type": "microsoft", "voice_id": body.voice_id or DID_VOICE_DEFAULT},
            "voice": {"type": "microsoft", "voice_id": body.voice_id or DID_VOICE_DEFAULT},
        },
        "source_url": body.source_url or DID_SOURCE_URL_DEFAULT,
        "config": body.config or {"fluent": True, "pad_audio": 0.5},
    }

    if not payload["source_url"]:
        raise HTTPException(status_code=400, detail="Missing source_url (DID_SOURCE_URL not set and source_url not provided)")

    headers = {
        **_auth_header(),
        "Content-Type": "application/json",
        "Accept": "application/json",
    }

    url = f"{DID_API_BASE.rstrip('/')}/talks"
    async with httpx.AsyncClient(timeout=60.0) as client:
        resp = await client.post(url, headers=headers, json=payload)
    if resp.status_code >= 400:
        try:
            detail = resp.json()
        except Exception:
            detail = {"raw": resp.text}
        raise HTTPException(status_code=resp.status_code, detail=detail)

    data = resp.json()
    # 常見欄位：id, status, result_url(完成後才會有)
    return {
        "ok": True,
        "talk_id": data.get("id"),
        "status": data.get("status"),
        "result_url": data.get("result_url"),
        "raw": data,
    }

@router.get("/get_talk/{talk_id}")
async def get_talk(talk_id: str):
    """
    查詢 talk 狀態。完成時會含 result_url，可直接給 <video> 播放。
    """
    if not DID_API_KEY:
        raise HTTPException(status_code=500, detail="DID_API_KEY is not configured")

    headers = {
        **_auth_header(),
        "Accept": "application/json",
    }
    url = f"{DID_API_BASE.rstrip('/')}/talks/{talk_id}"
    async with httpx.AsyncClient(timeout=60.0) as client:
        resp = await client.get(url, headers=headers)
    if resp.status_code >= 400:
        try:
            detail = resp.json()
        except Exception:
            detail = {"raw": resp.text}
        raise HTTPException(status_code=resp.status_code, detail=detail)

    data = resp.json()
    return {
        "ok": True,
        "talk_id": data.get("id"),
        "status": data.get("status"),
        "result_url": data.get("result_url"),
        "raw": data,
    }
