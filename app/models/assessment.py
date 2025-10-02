# backend/app/models/assessment.py
from sqlalchemy import Column, Integer, String, Text, DateTime, ForeignKey
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.sql import func
from app.db.base import Base

class Assessment(Base):
    __tablename__ = "assessments"
    
    id = Column(Integer, primary_key=True, index=True)
    pid = Column(String(10), ForeignKey("users.pid", ondelete="CASCADE"), nullable=False, index=True)
    
    # TEXT 用於 MBTI 字串
    mbti_raw = Column(Text, nullable=True)
    
    # JSONB 用於陣列和物件
    mbti_encoded = Column(JSONB, nullable=True)
    step2_answers = Column(JSONB, nullable=True)
    step3_answers = Column(JSONB, nullable=True)
    step4_answers = Column(JSONB, nullable=True)
    ai_preference = Column(JSONB, nullable=True)
    
    created_at = Column(DateTime(timezone=True), server_default=func.now(), index=True)