# app/routers/did_agents_stream.py
from __future__ import annotations
import os, logging, traceback
from typing import Optional, Dict, Any
import httpx
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

log = logging.getLogger("did-stream")
router = APIRouter(prefix="/chat/did/agents", tags=["did-agents"])

DID_API_KEY = (os.getenv("DID_API_KEY") or "").strip()
DID_AGENT_ID = (os.getenv("DID_AGENT_ID") or "").strip()
DID_BASE = "https://api.d-id.com"

def _safe_json(r: httpx.Response) -> Dict[str, Any]:
    try:
        return r.json()
    except Exception:
        return {"raw": r.text}

def _auth(extra: Dict[str, str] | None = None) -> Dict[str, str]:
    headers = {
        "accept": "application/json",
        "authorization": f"Basic {DID_API_KEY}",
    }
    if extra:
        headers.update(extra)
    return headers

class CreateStreamBody(BaseModel):
    fluent: bool = True
    compatibility_mode: str = "on"

class SDPBody(BaseModel):
    answer: str
    session_id: str

class ICEBody(BaseModel):
    candidate: Optional[dict] = None
    sdpMid: Optional[str] = None
    sdpMLineIndex: Optional[int] = None
    session_id: str

class SpeakBody(BaseModel):
    stream_id: str
    session_id: str
    text: str = Field(..., min_length=1)

@router.get("/health")
async def health():
    return {
        "ok": bool(DID_API_KEY and DID_AGENT_ID),
        "has_api_key": bool(DID_API_KEY),
        "has_agent_id": bool(DID_AGENT_ID),
    }

@router.get("/health/deep")
async def health_deep():
    """更詳盡：檢查環境、對 D-ID 丟一個最小請求，回傳狀態/訊息（不拋 500）。"""
    out: Dict[str, Any] = {
        "has_api_key": bool(DID_API_KEY),
        "has_agent_id": bool(DID_AGENT_ID),
        "base": DID_BASE,
        "agent_id_len": len(DID_AGENT_ID or ""),
    }
    if not DID_API_KEY or not DID_AGENT_ID:
        out["ok"] = False
        out["reason"] = "missing env"
        return out
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            # 打 agents/{id}/streams（不送 body 看會回什麼），只為了確認可達與權限訊息
            r = await client.post(
                f"{DID_BASE}/agents/{DID_AGENT_ID}/streams",
                headers=_auth({"content-type": "application/json"}),
                json={"compatibility_mode":"on","fluent":True},
            )
        out["status_code"] = r.status_code
        out["data"] = _safe_json(r)
        out["ok"] = 200 <= r.status_code < 500  # 2xx/4xx 都算「可達」
        return out
    except httpx.HTTPError as e:
        return {"ok": False, "kind": "UpstreamHTTP", "type": e.__class__.__name__, "message": str(e), "repr": repr(e)}
    except Exception as e:
        return {"ok": False, "kind": "BackendCrash", "type": e.__class__.__name__, "message": str(e), "repr": repr(e)}

@router.post("/streams")
async def create_stream(body: CreateStreamBody):
    if not DID_API_KEY or not DID_AGENT_ID:
        raise HTTPException(status_code=500, detail={"kind": "ConfigError", "desc": "DID_API_KEY/DID_AGENT_ID missing"})
    payload = {"compatibility_mode": body.compatibility_mode, "fluent": body.fluent}
    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            r = await client.post(
                f"{DID_BASE}/agents/{DID_AGENT_ID}/streams",
                headers=_auth({"content-type": "application/json"}),
                json=payload,
            )
        if 200 <= r.status_code < 300:
            return r.json()  # { id, session_id, offer, ice_servers }
        # 透傳上游錯誤（便於前端顯示與你排錯）
        raise HTTPException(status_code=r.status_code, detail=_safe_json(r))
    except httpx.HTTPError as e:
        tb = traceback.format_exc(limit=3)
        raise HTTPException(status_code=502, detail={
            "kind": "UpstreamHTTP",
            "type": e.__class__.__name__,
            "message": str(e),
            "repr": repr(e),
            "trace": tb,
        })
    except Exception as e:
        tb = traceback.format_exc(limit=3)
        raise HTTPException(status_code=500, detail={
            "kind": "BackendCrash",
            "type": e.__class__.__name__,
            "message": str(e),
            "repr": repr(e),
            "trace": tb,
        })

@router.post("/streams/{stream_id}/sdp")
async def send_sdp(stream_id: str, body: SDPBody):
    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            r = await client.post(
                f"{DID_BASE}/agents/{DID_AGENT_ID}/streams/{stream_id}/sdp",
                headers=_auth({"content-type": "application/json"}),
                json=body.dict(),
            )
        if 200 <= r.status_code < 300:
            return {"ok": True}
        raise HTTPException(status_code=r.status_code, detail=_safe_json(r))
    except httpx.HTTPError as e:
        tb = traceback.format_exc(limit=3)
        raise HTTPException(status_code=502, detail={"kind":"UpstreamHTTP","type":e.__class__.__name__,"message":str(e),"repr":repr(e),"trace":tb})
    except Exception as e:
        tb = traceback.format_exc(limit=3)
        raise HTTPException(status_code=500, detail={"kind":"BackendCrash","type":e.__class__.__name__,"message":str(e),"repr":repr(e),"trace":tb})

@router.post("/streams/{stream_id}/ice")
async def send_ice(stream_id: str, body: ICEBody):
    try:
        async with httpx.AsyncClient(timeout=15.0) as client:
            r = await client.post(
                f"{DID_BASE}/agents/{DID_AGENT_ID}/streams/{stream_id}/ice",
                headers=_auth({"content-type": "application/json"}),
                json=body.dict(),
            )
        if 200 <= r.status_code < 300:
            return {"ok": True}
        raise HTTPException(status_code=r.status_code, detail=_safe_json(r))
    except httpx.HTTPError as e:
        tb = traceback.format_exc(limit=3)
        raise HTTPException(status_code=502, detail={"kind":"UpstreamHTTP","type":e.__class__.__name__,"message":str(e),"repr":repr(e),"trace":tb})
    except Exception as e:
        tb = traceback.format_exc(limit=3)
        raise HTTPException(status_code=500, detail={"kind":"BackendCrash","type":e.__class__.__name__,"message":str(e),"repr":repr(e),"trace":tb})

@router.post("/streams/{stream_id}")
async def speak(stream_id: str, body: SpeakBody):
    payload = {"script": {"type": "text", "input": body.text}, "session_id": body.session_id}
    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            r = await client.post(
                f"{DID_BASE}/agents/{DID_AGENT_ID}/streams/{stream_id}",
                headers=_auth({"content-type": "application/json"}),
                json=payload,
            )
        if 200 <= r.status_code < 300:
            return {"ok": True}
        raise HTTPException(status_code=r.status_code, detail=_safe_json(r))
    except httpx.HTTPError as e:
        tb = traceback.format_exc(limit=3)
        raise HTTPException(status_code=502, detail={"kind":"UpstreamHTTP","type":e.__class__.__name__,"message":str(e),"repr":repr(e),"trace":tb})
    except Exception as e:
        tb = traceback.format_exc(limit=3)
        raise HTTPException(status_code=500, detail={"kind":"BackendCrash","type":e.__class__.__name__,"message":str(e),"repr":repr(e),"trace":tb})

@router.delete("/streams/{stream_id}")
async def close_stream(stream_id: str, session_id: str):
    try:
        async with httpx.AsyncClient(timeout=15.0) as client:
            r = await client.delete(
                f"{DID_BASE}/agents/{DID_AGENT_ID}/streams/{stream_id}",
                headers=_auth({"content-type": "application/json"}),
                json={"session_id": session_id},
            )
        if 200 <= r.status_code < 300:
            return {"ok": True}
        raise HTTPException(status_code=r.status_code, detail=_safe_json(r))
    except httpx.HTTPError as e:
        tb = traceback.format_exc(limit=3)
        raise HTTPException(status_code=502, detail={"kind":"UpstreamHTTP","type":e.__class__.__name__,"message":str(e),"repr":repr(e),"trace":tb})
    except Exception as e:
        tb = traceback.format_exc(limit=3)
        raise HTTPException(status_code=500, detail={"kind":"BackendCrash","type":e.__class__.__name__,"message":str(e),"repr":repr(e),"trace":tb})
