# backend/app/core/timezone.py
from datetime import datetime, timezone, timedelta

# 台灣時區 UTC+8
TW_TZ = timezone(timedelta(hours=8))

def get_tw_time() -> datetime:
    """取得當前台灣時間（帶時區資訊）"""
    return datetime.now(TW_TZ)

def utc_to_tw(utc_time: datetime) -> datetime:
    """將 UTC 時間轉換為台灣時間"""
    if utc_time is None:
        return None
    
    # 如果沒有時區資訊，假設為 UTC
    if utc_time.tzinfo is None:
        utc_time = utc_time.replace(tzinfo=timezone.utc)
    
    # 轉換為台灣時間
    return utc_time.astimezone(TW_TZ)

def format_tw_time(dt: datetime) -> str:
    """格式化台灣時間為 ISO 字串"""
    if dt is None:
        return None
    
    tw_dt = utc_to_tw(dt) if dt.tzinfo != TW_TZ else dt
    return tw_dt.isoformat()