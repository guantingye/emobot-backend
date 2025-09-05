# app/models/recommendation.py
from sqlalchemy import Column, Integer, String, Float, JSON, DateTime, ForeignKey, func
from sqlalchemy.orm import relationship
from app.db.base import Base

class Recommendation(Base):
    __tablename__ = "recommendations"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True)

    # 以 JSON 儲存四型分數（0~1）與排序結果（0~100）
    scores = Column(JSON, nullable=True)           # 例：{"empathy":0.72,"insight":0.51,"solution":0.63,"cognitive":0.34}
    ranked = Column(JSON, nullable=True)           # 例：[{"type":"empathy","score":87.2}, ...]
    selected_bot = Column(String(16), nullable=True)  # 使用者選擇的機器人類型

    created_at = Column(DateTime, server_default=func.now())

    user = relationship("User", back_populates="recommendations")