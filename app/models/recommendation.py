# app/models/recommendation.py
from sqlalchemy.orm import Mapped, mapped_column, relationship
from sqlalchemy import Integer, DateTime, ForeignKey, String
from sqlalchemy.dialects.postgresql import JSONB
from datetime import datetime
from app.db.base import Base

class Recommendation(Base):
    __tablename__ = "recommendations"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    user_id: Mapped[int] = mapped_column(ForeignKey("users.id"), index=True)
    assessment_id: Mapped[int | None] = mapped_column(ForeignKey("assessments.id"), index=True, nullable=True)

    scores: Mapped[dict] = mapped_column(JSONB)
    top_bot: Mapped[str] = mapped_column(String(32))
    features: Mapped[dict | None] = mapped_column(JSONB, nullable=True)

    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)

    assessment = relationship("Assessment", back_populates="recommendations")
