from sqlalchemy import Integer, DateTime, ForeignKey, String
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column, relationship
from datetime import datetime
from app.db.base import Base

class Assessment(Base):
    __tablename__ = "assessments"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    user_id: Mapped[int] = mapped_column(ForeignKey("users.id"), index=True)

    # Step1
    mbti_raw: Mapped[str | None] = mapped_column(String(8), nullable=True)
    mbti_encoded: Mapped[dict | None] = mapped_column(JSONB, nullable=True)

    # Step2~4 與偏好（保留彈性）
    step2_answers: Mapped[dict | None] = mapped_column(JSONB, nullable=True)
    step3_answers: Mapped[dict | None] = mapped_column(JSONB, nullable=True)
    step4_answers: Mapped[dict | None] = mapped_column(JSONB, nullable=True)
    ai_preference: Mapped[dict | None] = mapped_column(JSONB, nullable=True)

    submitted_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    created_at: Mapped[datetime] = mapped_column(default=datetime.utcnow)

    user = relationship("User", back_populates="assessments")
    recommendations = relationship("Recommendation", back_populates="assessment")
