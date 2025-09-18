# app/services/did_service.py - D-ID API 整合服務
import os
import asyncio
import logging
import aiohttp
from typing import Dict, Any, Optional
from pydantic import BaseModel

logger = logging.getLogger(__name__)

class DIDVideoRequest(BaseModel):
    script_text: str
    voice_id: str = "zh-TW-HsiaoChenNeural"
    avatar_url: Optional[str] = None
    presenter_id: Optional[str] = None

class DIDVideoResponse(BaseModel):
    success: bool
    talk_id: Optional[str] = None
    video_url: Optional[str] = None
    status: Optional[str] = None
    error: Optional[str] = None
    data: Optional[Dict] = None

class DIDService:
    def __init__(self):
        self.api_key = os.getenv("DID_API_KEY")
        self.base_url = "https://api.d-id.com"
        self.default_avatar = "https://create-images-results.d-id.com/DefaultPresenters/Noelle_f/image.jpeg"
        self.default_presenter = "amy-jku7W6h58r"  # D-ID 預設中文女性講者
        
    async def create_talk(self, request: DIDVideoRequest) -> DIDVideoResponse:
        """創建 D-ID 說話視頻"""
        if not self.api_key:
            return DIDVideoResponse(
                success=False,
                error="D-ID API key not configured"
            )
        
        # 準備請求資料
        talk_data = {
            "source_url": request.avatar_url or self.default_avatar,
            "script": {
                "type": "text",
                "provider": {
                    "type": "microsoft",
                    "voice_id": request.voice_id,
                    "voice_config": {
                        "style": "friendly",
                        "rate": "1.0",
                        "pitch": "medium"
                    }
                },
                "input": request.script_text
            },
            "config": {
                "fluent": True,
                "pad_audio": 0,
                "stitch": True,
                "align_driver": True,
                "align_expand_factor": 1
            }
        }
        
        # 如果指定了 presenter_id，使用預設講者而非自定義頭像
        if request.presenter_id:
            talk_data["presenter_id"] = request.presenter_id
            del talk_data["source_url"]
        
        headers = {
            "Authorization": f"Basic {self.api_key}",
            "Content-Type": "application/json",
            "Accept": "application/json"
        }
        
        try:
            async with aiohttp.ClientSession() as session:
                # 創建說話任務
                async with session.post(
                    f"{self.base_url}/talks",
                    headers=headers,
                    json=talk_data,
                    timeout=aiohttp.ClientTimeout(total=30)
                ) as response:
                    if response.status == 201:
                        result = await response.json()
                        talk_id = result.get("id")
                        
                        if talk_id:
                            # 等待視頻生成完成
                            video_url = await self._wait_for_completion(talk_id, session, headers)
                            
                            if video_url:
                                return DIDVideoResponse(
                                    success=True,
                                    talk_id=talk_id,
                                    video_url=video_url,
                                    status="completed",
                                    data=result
                                )
                            else:
                                return DIDVideoResponse(
                                    success=False,
                                    talk_id=talk_id,
                                    status="failed",
                                    error="Video generation timeout or failed"
                                )
                        else:
                            return DIDVideoResponse(
                                success=False,
                                error="No talk ID returned from D-ID API"
                            )
                    else:
                        error_text = await response.text()
                        logger.error(f"D-ID API error {response.status}: {error_text}")
                        return DIDVideoResponse(
                            success=False,
                            error=f"HTTP {response.status}: {error_text[:200]}"
                        )
                        
        except asyncio.TimeoutError:
            return DIDVideoResponse(
                success=False,
                error="Request timeout - D-ID service may be slow"
            )
        except Exception as e:
            logger.error(f"D-ID talk creation failed: {e}")
            return DIDVideoResponse(
                success=False,
                error=str(e)
            )
    
    async def _wait_for_completion(self, talk_id: str, session: aiohttp.ClientSession, headers: Dict, max_wait: int = 60) -> Optional[str]:
        """等待視頻生成完成"""
        for _ in range(max_wait):  # 最多等待60秒
            try:
                async with session.get(
                    f"{self.base_url}/talks/{talk_id}",
                    headers=headers,
                    timeout=aiohttp.ClientTimeout(total=10)
                ) as response:
                    if response.status == 200:
                        result = await response.json()
                        status = result.get("status")
                        
                        if status == "done":
                            return result.get("result_url")
                        elif status == "error":
                            logger.error(f"D-ID generation failed: {result.get('error', 'Unknown error')}")
                            return None
                        # 如果還在處理中，繼續等待
                        
            except Exception as e:
                logger.error(f"Error checking D-ID status: {e}")
                
            await asyncio.sleep(1)  # 等待1秒後重試
        
        return None  # 超時
    
    async def get_talk_status(self, talk_id: str) -> DIDVideoResponse:
        """查詢說話視頻狀態"""
        if not self.api_key:
            return DIDVideoResponse(
                success=False,
                error="D-ID API key not configured"
            )
        
        headers = {
            "Authorization": f"Basic {self.api_key}",
            "Accept": "application/json"
        }
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    f"{self.base_url}/talks/{talk_id}",
                    headers=headers,
                    timeout=aiohttp.ClientTimeout(total=15)
                ) as response:
                    if response.status == 200:
                        result = await response.json()
                        return DIDVideoResponse(
                            success=True,
                            talk_id=talk_id,
                            status=result.get("status"),
                            video_url=result.get("result_url"),
                            data=result
                        )
                    else:
                        error_text = await response.text()
                        return DIDVideoResponse(
                            success=False,
                            error=f"HTTP {response.status}: {error_text[:200]}"
                        )
                        
        except Exception as e:
            logger.error(f"D-ID status check failed: {e}")
            return DIDVideoResponse(
                success=False,
                error=str(e)
            )
    
    async def delete_talk(self, talk_id: str) -> bool:
        """刪除說話視頻（清理資源）"""
        if not self.api_key:
            return False
        
        headers = {
            "Authorization": f"Basic {self.api_key}",
        }
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.delete(
                    f"{self.base_url}/talks/{talk_id}",
                    headers=headers,
                    timeout=aiohttp.ClientTimeout(total=15)
                ) as response:
                    return response.status in [200, 204]
                    
        except Exception as e:
            logger.error(f"D-ID talk deletion failed: {e}")
            return False
    
    async def health_check(self) -> Dict[str, Any]:
        """D-ID 服務健康檢查"""
        info = {
            "service": "D-ID",
            "has_api_key": bool(self.api_key),
            "base_url": self.base_url,
            "default_avatar": self.default_avatar,
            "default_presenter": self.default_presenter,
            "ok": False,
            "error": None
        }
        
        if not self.api_key:
            info["error"] = "D-ID API key not configured"
            return info
        
        try:
            headers = {
                "Authorization": f"Basic {self.api_key}",
                "Accept": "application/json"
            }
            
            async with aiohttp.ClientSession() as session:
                # 測試API連通性 - 查詢credits
                async with session.get(
                    f"{self.base_url}/credits",
                    headers=headers,
                    timeout=aiohttp.ClientTimeout(total=10)
                ) as response:
                    if response.status == 200:
                        credits_info = await response.json()
                        info["ok"] = True
                        info["credits_remaining"] = credits_info.get("remaining", "unknown")
                        info["plan"] = credits_info.get("plan_type", "unknown")
                    else:
                        error_text = await response.text()
                        info["error"] = f"HTTP {response.status}: {error_text[:100]}"
                        
        except Exception as e:
            info["error"] = f"{type(e).__name__}: {str(e)[:150]}"
        
        return info