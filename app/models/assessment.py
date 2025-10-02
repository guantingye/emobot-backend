# backend/app/models/assessment.py
from sqlalchemy import Column, Integer, String, DateTime, ForeignKey, JSON
from sqlalchemy.sql import func
from app.db.base import Base

class Assessment(Base):
    __tablename__ = "assessments"
    
    id = Column(Integer, primary_key=True, index=True)
    pid = Column(String(10), ForeignKey("users.pid", ondelete="CASCADE"), nullable=False, index=True)
    mbti_raw = Column(JSON)
    mbti_encoded = Column(JSON)
    step2_answers = Column(JSON)
    step3_answers = Column(JSON)
    step4_answers = Column(JSON)
    ai_preference = Column(JSON)
    created_at = Column(DateTime(timezone=True), server_default=func.now())