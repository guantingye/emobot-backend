# app/models/user.py
from sqlalchemy.orm import Mapped, mapped_column, relationship
from sqlalchemy import Integer, DateTime, String
from datetime import datetime
from app.db.base import Base

class User(Base):
    __tablename__ = "users"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    pid: Mapped[str] = mapped_column(String(50), unique=True, index=True)
    nickname: Mapped[str | None] = mapped_column(String(100), nullable=True)
    selected_bot: Mapped[str | None] = mapped_column(String(20), nullable=True)

    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)

    assessments = relationship("Assessment", back_populates="user")
    recommendations = relationship("Recommendation", back_populates="user")
    chat_messages = relationship("ChatMessage", back_populates="user")
    mood_records = relationship("MoodRecord", back_populates="user")