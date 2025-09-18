# app/routers/did_router.py
from __future__ import annotations

import os
import logging
from typing import Optional, Dict, Any

import httpx
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

router = APIRouter(prefix="/api/chat/did", tags=["did"])
log = logging.getLogger("did")

# --- 環境變數 ---
DID_API_KEY = (os.getenv("DID_API_KEY") or "").strip()
DID_DEFAULT_SOURCE = (os.getenv("DID_SOURCE_URL") or "").strip()  # 可不填；若沒提供，將使用帳戶預設 presenter（若有）
DID_BASE = "https://api.d-id.com"

# --- 請求模型 ---
class CreateTalkBody(BaseModel):
    text: str = Field(..., min_length=1, description="要讓 Avatar 說的文字（請放 LLM 回覆）")
    voice_id: str = Field(default="zh-TW-HsiaoChenNeural", description="Microsoft TTS 聲音名稱")
    source_url: Optional[str] = Field(default=None, description="可公開抓取的頭像 URL；不填則嘗試用伺服器端 DID_SOURCE_URL")
    config: Dict[str, Any] = Field(default_factory=lambda: {"fluent": True, "pad_audio": 0.3})
    webhook: Optional[str] = Field(default=None, description="（選用）完成時回呼網址")

# --- 共用小工具 ---
def _safe_json(resp: httpx.Response) -> Dict[str, Any]:
    try:
        return resp.json()
    except Exception:
        return {"raw": resp.text}

def _auth_headers(extra: Dict[str, str] | None = None) -> Dict[str, str]:
    headers = {
        "accept": "application/json",
        "authorization": f"Basic {DID_API_KEY}",  # D-ID 使用 Basic + API Key
    }
    if extra:
        headers.update(extra)
    return headers

# --- 健康檢查：驗證金鑰可用 ---
@router.get("/health")
async def did_health():
    if not DID_API_KEY:
        return {"ok": False, "reason": "DID_API_KEY missing"}
    try:
        async with httpx.AsyncClient(timeout=15.0) as client:
            r = await client.get(f"{DID_BASE}/talks?limit=1", headers=_auth_headers())
        if r.status_code in (200, 204):
            return {"ok": True}
        return {"ok": False, "status": r.status_code, "detail": _safe_json(r)}
    except Exception as e:
        log.exception("D-ID health check failed")
        return {"ok": False, "error": str(e)}

# --- 建立 talk：把 LLM 文字塞進 script.input 並指定 provider ---
@router.post("/create_talk")
async def create_talk(body: CreateTalkBody):
    if not DID_API_KEY:
        raise HTTPException(status_code=500, detail={"kind": "ConfigError", "description": "DID_API_KEY missing"})

    payload: Dict[str, Any] = {
        "script": {
            "type": "text",
            "input": body.text,  # ✅ 關鍵：用 LLM 回覆文字
            "provider": {        # ✅ 依官方：provider 放在 script 裡
                "type": "microsoft",
                "voice_id": body.voice_id,
                # 若需要風格可加 voice_config，例如：{"style": "Cheerful"}
            },
            # "ssml": False,     # 需要 SSML 再打開
        },
        "config": body.config or {"fluent": True, "pad_audio": 0.3},
    }

    # 頭像來源：優先用前端傳入，再用後端環境變數；都沒有就讓 D-ID 使用帳戶預設 presenter（若方案支援）
    final_source = body.source_url or DID_DEFAULT_SOURCE
    if final_source:
        payload["source_url"] = final_source

    if body.webhook:
        payload["webhook"] = body.webhook

    try:
        async with httpx.AsyncClient(timeout=60.0) as client:
            r = await client.post(
                f"{DID_BASE}/talks",
                headers=_auth_headers({"content-type": "application/json"}),
                json=payload,
            )

        if r.status_code == 402:
            # 額度不足：回傳 ok:false，讓前端能優雅回退
            return {"ok": False, "error_code": "INSUFFICIENT_CREDITS", "detail": _safe_json(r)}

        if 200 <= r.status_code < 300:
            j = r.json()
            return {"ok": True, "talk_id": j.get("id"), "raw": j}

        detail = _safe_json(r)
        log.error("D-ID create_talk failed %s %s", r.status_code, detail)
        raise HTTPException(status_code=r.status_code, detail=detail)

    except httpx.ReadTimeout:
        raise HTTPException(status_code=504, detail={"kind": "UpstreamTimeout", "description": "D-ID timeout"})
    except Exception as e:
        log.exception("create_talk crashed")
        raise HTTPException(status_code=500, detail={"kind": "BackendCrash", "description": str(e)})

# --- 查詢 talk 狀態：status=done 時會有 result_url 可播放 ---
@router.get("/get_talk/{talk_id}")
async def get_talk(talk_id: str):
    if not DID_API_KEY:
        raise HTTPException(status_code=500, detail={"kind": "ConfigError", "description": "DID_API_KEY missing"})
    try:
        async with httpx.AsyncClient(timeout=60.0) as client:
            r = await client.get(f"{DID_BASE}/talks/{talk_id}", headers=_auth_headers())

        if 200 <= r.status_code < 300:
            j = r.json()
            status = j.get("status")
            if status in ("done", "generated", "succeeded"):
                # ✅ 影片只有在 done 時才提供 result_url（可直接 <video src> 播放）
                return {"status": "done", "result_url": j.get("result_url"), "raw": j}
            if status in ("error", "failed"):
                return {"status": "error", "raw": j}
            return {"status": status or "pending", "raw": j}

        detail = _safe_json(r)
        raise HTTPException(status_code=r.status_code, detail=detail)

    except httpx.ReadTimeout:
        raise HTTPException(status_code=504, detail={"kind": "UpstreamTimeout", "description": "D-ID timeout"})
    except Exception as e:
        log.exception("get_talk crashed")
        raise HTTPException(status_code=500, detail={"kind": "BackendCrash", "description": str(e)})
