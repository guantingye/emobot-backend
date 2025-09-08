# app/models/user.py
from sqlalchemy import Column, Integer, String, DateTime, func
from sqlalchemy.orm import relationship
from app.db.base import Base

class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    pid = Column(String, unique=True, index=True, nullable=False)
    nickname = Column(String, nullable=True)
    selected_bot = Column(String, nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), server_default=func.now())

    # ★ 修復：使用字符串引用避免循環引用問題
    chat_messages = relationship(
        "ChatMessage",
        back_populates="user",
        cascade="all, delete-orphan",
        passive_deletes=True,
        lazy="select"
    )
    assessments = relationship(
        "Assessment",
        back_populates="user",
        cascade="all, delete-orphan", 
        passive_deletes=True,
        lazy="select"
    )
    recommendations = relationship(
        "Recommendation",
        back_populates="user",
        cascade="all, delete-orphan",
        passive_deletes=True,
        lazy="select"
    )
    moods = relationship(
        "MoodRecord", 
        back_populates="user",
        cascade="all, delete-orphan",
        passive_deletes=True,
        lazy="select"
    )
    # ★ 新增聊天會話關聯
    chat_sessions = relationship(
        "ChatSession",
        back_populates="user",
        cascade="all, delete-orphan",
        passive_deletes=True,
        lazy="select"
    )