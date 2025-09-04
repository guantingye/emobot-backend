# app/models/recommendation.py
from sqlalchemy import Column, Integer, String, JSON, DateTime, ForeignKey, func
from sqlalchemy.orm import relationship
from app.db.base import Base

class Recommendation(Base):
    __tablename__ = "recommendations"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(
        Integer, 
        ForeignKey("users.id", ondelete="CASCADE"), 
        nullable=True,  # 你的資料庫允許 NULL
        index=True
    )
    assessment_id = Column(
        Integer,
        ForeignKey("assessments.id", ondelete="SET NULL"),
        nullable=True
    )

    selected_bot = Column(String, nullable=True)  # 你的資料庫已有此欄位
    scores = Column(JSON, nullable=True)
    features = Column(JSON, nullable=True)
    ranked = Column(JSON, nullable=True)  # 新增的欄位

    created_at = Column(DateTime(timezone=True), server_default=func.now())

    user = relationship("User", back_populates="recommendations", lazy="select")