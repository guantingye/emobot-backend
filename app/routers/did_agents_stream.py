# app/routers/did_agents_stream.py
from __future__ import annotations
import os
import logging
from typing import Optional, Dict, Any

import httpx
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

log = logging.getLogger("did-stream")
# 不含 /api，讓 main.py 用 prefix="/api" 掛上最終路徑 /api/chat/did/agents/*
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
    compatibility_mode: str = "on"  # 建議在瀏覽器端時開啟


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
    if not DID_API_KEY:
        return {"ok": False, "reason": "DID_API_KEY missing"}
    if not DID_AGENT_ID:
        return {"ok": False, "reason": "DID_AGENT_ID missing"}
    return {"ok": True}


@router.post("/streams")
async def create_stream(body: CreateStreamBody):
    if not DID_API_KEY or not DID_AGENT_ID:
        raise HTTPException(
            status_code=500,
            detail={"kind": "ConfigError", "desc": "DID_API_KEY/DID_AGENT_ID missing"},
        )
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
        raise HTTPException(status_code=r.status_code, detail=_safe_json(r))
    except Exception as e:
        log.exception("create_stream failed")
        raise HTTPException(status_code=500, detail={"kind": "BackendCrash", "desc": str(e)})


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
    except Exception as e:
        log.exception("send_sdp failed")
        raise HTTPException(status_code=500, detail={"kind": "BackendCrash", "desc": str(e)})


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
    except Exception as e:
        log.exception("send_ice failed")
        raise HTTPException(status_code=500, detail={"kind": "BackendCrash", "desc": str(e)})


@router.post("/streams/{stream_id}")
async def speak(stream_id: str, body: SpeakBody):
    payload = {
        "script": {"type": "text", "input": body.text},
        "session_id": body.session_id,
    }
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
    except Exception as e:
        log.exception("speak failed")
        raise HTTPException(status_code=500, detail={"kind": "BackendCrash", "desc": str(e)})


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
    except Exception as e:
        log.exception("close_stream failed")
        raise HTTPException(status_code=500, detail={"kind": "BackendCrash", "desc": str(e)})
