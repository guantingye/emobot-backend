# app/models/assessment.py
from sqlalchemy.orm import Mapped, mapped_column, relationship
from sqlalchemy import Integer, DateTime, ForeignKey, String
from sqlalchemy.dialects.postgresql import JSONB
from datetime import datetime
from app.db.base import Base

class Assessment(Base):
    __tablename__ = "assessments"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    user_id: Mapped[int] = mapped_column(ForeignKey("users.id"), index=True)

    mbti_raw: Mapped[str | None] = mapped_column(String(8), nullable=True)
    mbti_encoded: Mapped[list | dict | None] = mapped_column(JSONB, nullable=True)

    step2_answers: Mapped[list | None] = mapped_column(JSONB, nullable=True)
    step3_answers: Mapped[list | None] = mapped_column(JSONB, nullable=True)
    step4_answers: Mapped[list | None] = mapped_column(JSONB, nullable=True)
    ai_preference: Mapped[dict | None] = mapped_column(JSONB, nullable=True)

    submitted_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)

    user = relationship("User", back_populates="assessments")
    recommendations = relationship("Recommendation", back_populates="assessment")
