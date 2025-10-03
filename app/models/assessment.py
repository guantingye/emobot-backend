# backend/app/models/assessment.py
from sqlalchemy import Column, Integer, String, Text, DateTime, ForeignKey, Boolean
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.sql import func
from app.db.base import Base

class Assessment(Base):
    __tablename__ = "assessments"
    
    # ✅ 主鍵改為自增 ID，不再以 PID 為唯一鍵
    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    pid = Column(String(10), ForeignKey("users.pid", ondelete="CASCADE"), nullable=False, index=True)
    
    # 測驗資料欄位
    mbti_raw = Column(Text, nullable=True)
    mbti_encoded = Column(JSONB, nullable=True)
    step2_answers = Column(JSONB, nullable=True)
    step3_answers = Column(JSONB, nullable=True)
    step4_answers = Column(JSONB, nullable=True)
    ai_preference = Column(JSONB, nullable=True)
    
    # ✅ 新增：是否為重測
    is_retest = Column(Boolean, default=False, nullable=False)
    
    # ✅ 時間戳記（每次測驗都會有新的時間）
    created_at = Column(DateTime(timezone=True), server_default=func.now(), index=True)
    
    # ✅ 新增：完成時間（當測驗完整完成時更新）
    completed_at = Column(DateTime(timezone=True), nullable=True)