# app/models/assessment.py
from sqlalchemy import Column, Integer, String, JSON, DateTime, ForeignKey, func
from sqlalchemy.orm import relationship
from app.db.base import Base

class Assessment(Base):
    __tablename__ = "assessments"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(
        Integer, 
        ForeignKey("users.id", ondelete="CASCADE"), 
        nullable=True,  # 你的資料庫允許 NULL
        index=True
    )

    mbti_raw = Column(String, nullable=True)
    mbti_encoded = Column(JSON, nullable=True)
    step2_answers = Column(JSON, nullable=True)
    step3_answers = Column(JSON, nullable=True)
    step4_answers = Column(JSON, nullable=True)
    ai_preference = Column(JSON, nullable=True)
    submitted_at = Column(DateTime(timezone=True), nullable=True)

    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), server_default=func.now())

    user = relationship("User", back_populates="assessments", lazy="select")