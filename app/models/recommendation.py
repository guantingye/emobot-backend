# backend/app/models/recommendation.py
from sqlalchemy import Column, Integer, String, DateTime, ForeignKey, JSON
from sqlalchemy.sql import func
from app.db.base import Base

class Recommendation(Base):
    __tablename__ = "recommendations"
    
    id = Column(Integer, primary_key=True, index=True)
    pid = Column(String(10), ForeignKey("users.pid", ondelete="CASCADE"), nullable=False, index=True)
    scores = Column(JSON)
    selected_bot = Column(String(20))
    created_at = Column(DateTime(timezone=True), server_default=func.now())