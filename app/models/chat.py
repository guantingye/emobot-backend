# backend/app/models/chat.py
from sqlalchemy import Column, Integer, String, Text, DateTime, ForeignKey, JSON
from sqlalchemy.sql import func
from app.db.base import Base

class ChatMessage(Base):
    __tablename__ = "chat_messages"
    
    id = Column(Integer, primary_key=True, index=True)
    pid = Column(String(10), ForeignKey("users.pid", ondelete="CASCADE"), nullable=False, index=True)
    role = Column(String(20), nullable=False)
    content = Column(Text, nullable=False)
    bot_type = Column(String(20))
    mode = Column(String(20), default="text")
    created_at = Column(DateTime(timezone=True), server_default=func.now(), index=True)
    meta = Column(JSON)