# app/services/heygen_service.py
import asyncio
import aiohttp
import json
import os
from typing import Dict, Any, Optional
from datetime import datetime, timedelta
import logging

logger = logging.getLogger(__name__)

class HeyGenStreamingService:
    """HeyGen Streaming API 服務類"""
    
    def __init__(self):
        self.api_key = os.getenv("HEYGEN_API_KEY", "")
        self.base_url = "https://api.heygen.com/v2/streaming"
        self.sessions: Dict[str, Dict] = {}
        
        if not self.api_key:
            logger.warning("HeyGen API key not found. Video mode may not work properly.")
    
    async def create_streaming_session(
        self, 
        avatar_id: str = None, 
        voice: str = "zh-TW-HsiaoChenNeural",
        quality: str = "high"
    ) -> Dict[str, Any]:
        """創建HeyGen串流會話"""
        if not self.api_key:
            raise Exception("HeyGen API key not configured")
        
        avatar_id = avatar_id or os.getenv("HEYGEN_AVATAR_ID", "default_avatar")
        
        session_data = {
            "avatar_name": avatar_id,
            "voice": {
                "voice_id": voice,
                "rate": 1.0,
                "emotion": "friendly"
            },
            "quality": quality,
            "language": "zh-TW",
            "version": "v2"
        }
        
        headers = {
            "X-API-KEY": self.api_key,
            "Content-Type": "application/json"
        }
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"{self.base_url}/create_session",
                    headers=headers,
                    json=session_data
                ) as response:
                    if response.status == 200:
                        result = await response.json()
                        if result.get("code") == 100:
                            session_id = result["data"]["session_id"]
                            
                            # 儲存會話資訊
                            self.sessions[session_id] = {
                                "created_at": datetime.utcnow(),
                                "avatar_id": avatar_id,
                                "voice": voice,
                                "status": "active"
                            }
                            
                            return {
                                "success": True,
                                "session_id": session_id,
                                "data": result["data"]
                            }
                        else:
                            return {
                                "success": False,
                                "error": result.get("message", "Session creation failed")
                            }
                    else:
                        error_text = await response.text()
                        return {
                            "success": False,
                            "error": f"HTTP {response.status}: {error_text}"
                        }
                        
        except Exception as e:
            logger.error(f"HeyGen session creation failed: {e}")
            return {
                "success": False,
                "error": str(e)
            }
    
    async def send_text_to_avatar(
        self, 
        session_id: str, 
        text: str,
        emotion: str = "friendly"
    ) -> Dict[str, Any]:
        """發送文字給Avatar進行repeat任務"""
        if not self.api_key:
            return {"success": False, "error": "API key not configured"}
        
        if session_id not in self.sessions:
            return {"success": False, "error": "Session not found"}
        
        repeat_data = {
            "session_id": session_id,
            "text": text,
            "voice": {
                "emotion": emotion,
                "rate": 1.0
            },
            "background": {
                "type": "color",
                "value": "#FFFFFF"
            }
        }
        
        headers = {
            "X-API-KEY": self.api_key,
            "Content-Type": "application/json"
        }
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"{self.base_url}/repeat",
                    headers=headers,
                    json=repeat_data
                ) as response:
                    if response.status == 200:
                        result = await response.json()
                        if result.get("code") == 100:
                            return {
                                "success": True,
                                "message": "Text sent to avatar successfully"
                            }
                        else:
                            return {
                                "success": False,
                                "error": result.get("message", "Failed to send text")
                            }
                    else:
                        error_text = await response.text()
                        return {
                            "success": False,
                            "error": f"HTTP {response.status}: {error_text}"
                        }
                        
        except Exception as e:
            logger.error(f"Failed to send text to avatar: {e}")
            return {
                "success": False,
                "error": str(e)
            }
    
    async def close_session(self, session_id: str) -> Dict[str, Any]:
        """關閉HeyGen會話"""
        if not self.api_key:
            return {"success": False, "error": "API key not configured"}
        
        if session_id not in self.sessions:
            return {"success": False, "error": "Session not found"}
        
        headers = {
            "X-API-KEY": self.api_key,
            "Content-Type": "application/json"
        }
        
        close_data = {
            "session_id": session_id
        }
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"{self.base_url}/close_session",
                    headers=headers,
                    json=close_data
                ) as response:
                    result = await response.json()
                    
                    # 從本地記錄中移除
                    if session_id in self.sessions:
                        del self.sessions[session_id]
                    
                    return {
                        "success": True,
                        "message": "Session closed successfully"
                    }
                    
        except Exception as e:
            logger.error(f"Failed to close session: {e}")
            return {
                "success": False,
                "error": str(e)
            }
    
    async def get_session_info(self, session_id: str) -> Optional[Dict]:
        """獲取會話資訊"""
        return self.sessions.get(session_id)
    
    def cleanup_expired_sessions(self, max_age_hours: int = 2):
        """清理過期的會話記錄"""
        cutoff_time = datetime.utcnow() - timedelta(hours=max_age_hours)
        expired_sessions = [
            sid for sid, info in self.sessions.items()
            if info["created_at"] < cutoff_time
        ]
        
        for sid in expired_sessions:
            del self.sessions[sid]
        
        return len(expired_sessions)

# 全域服務實例
heygen_service = HeyGenStreamingService()