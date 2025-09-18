# app/routers/did_router.py
from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel, Field
import httpx, os, logging

router = APIRouter(prefix="/api/chat/did", tags=["did"])
log = logging.getLogger("did")

DID_API_KEY = os.getenv("DID_API_KEY", "").strip()
DID_DEFAULT_SOURCE = os.getenv("DID_SOURCE_URL", "").strip()
DID_BASE = "https://api.d-id.com"

class CreateTalkBody(BaseModel):
    text: str = Field(..., min_length=1)
    voice_id: str = Field(default="zh-TW-HsiaoChenNeural")
    source_url: str | None = None
    config: dict = Field(default={"fluent": True, "pad_audio": 0.3})

@router.get("/health")
async def did_health():
    if not DID_API_KEY:
        return {"ok": False, "reason": "DID_API_KEY missing"}
    # 簡單 ping：取得最近一筆 talks（不一定有，但可驗權）
    try:
        async with httpx.AsyncClient(timeout=15.0) as client:
            r = await client.get(
                f"{DID_BASE}/talks?limit=1",
                headers={"accept": "application/json", "authorization": f"Basic {DID_API_KEY}"}
            )
        if r.status_code in (200, 204):
            return {"ok": True}
        return {"ok": False, "status": r.status_code, "detail": _safe_json(r)}
    except Exception as e:
        log.exception("health error")
        return {"ok": False, "error": str(e)}

def _safe_json(resp: httpx.Response):
    try:
        return resp.json()
    except Exception:
        return {"raw": resp.text}

@router.post("/create_talk")
async def create_talk(body: CreateTalkBody):
    if not DID_API_KEY:
        raise HTTPException(status_code=500, detail={"kind": "ConfigError", "description": "DID_API_KEY missing"})

    payload = {
        "script": {
            "type": "text",
            "input": body.text,
            "voice": {"type": "microsoft", "voice_id": body.voice_id},
        },
        "source_url": body.source_url or DID_DEFAULT_SOURCE,
        "config": body.config or {"fluent": True, "pad_audio": 0.3},
    }

    if not payload["source_url"]:
        # 允許前端不傳，後端也沒設；那就讓 D-ID 用 default presenter（若你的帳戶有）
        payload.pop("source_url")

    headers = {
        "accept": "application/json",
        "content-type": "application/json",
        "authorization": f"Basic {DID_API_KEY}",
    }

    try:
        async with httpx.AsyncClient(timeout=60.0) as client:
            r = await client.post(f"{DID_BASE}/talks", headers=headers, json=payload)

        if r.status_code == 402:
            # 額度不足：包裝成 ok:false 方便前端回退
            return {"ok": False, "error_code": "INSUFFICIENT_CREDITS", "detail": _safe_json(r)}
        if 200 <= r.status_code < 300:
            j = r.json()
            return {"ok": True, "talk_id": j.get("id"), "raw": j}

        # 其他錯誤：把 D-ID 的訊息透傳，避免「UnknownError」
        detail = _safe_json(r)
        log.error("D-ID create failed %s %s", r.status_code, detail)
        raise HTTPException(status_code=r.status_code, detail=detail)

    except httpx.ReadTimeout:
        raise HTTPException(status_code=504, detail={"kind": "UpstreamTimeout", "description": "D-ID timeout"})
    except Exception as e:
        log.exception("create_talk crashed")
        raise HTTPException(status_code=500, detail={"kind": "BackendCrash", "description": str(e)})

@router.get("/get_talk/{talk_id}")
async def get_talk(talk_id: str):
    if not DID_API_KEY:
        raise HTTPException(status_code=500, detail={"kind": "ConfigError", "description": "DID_API_KEY missing"})

    headers = {
        "accept": "application/json",
        "authorization": f"Basic {DID_API_KEY}",
    }
    try:
        async with httpx.AsyncClient(timeout=60.0) as client:
            r = await client.get(f"{DID_BASE}/talks/{talk_id}", headers=headers)

        if 200 <= r.status_code < 300:
            j = r.json()
            # D-ID 完成時會提供 result_url / result_url_expire_at
            status = j.get("status")
            if status in ("done", "generated", "succeeded"):
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
