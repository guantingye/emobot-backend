# app/models/chat.py
from datetime import datetime
from sqlalchemy import Column, Integer, String, DateTime, Text, JSON, ForeignKey
from sqlalchemy.orm import relationship
from app.db.base import Base

class ChatMessage(Base):
    __tablename__ = "chat_messages"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(
        Integer, 
        ForeignKey("users.id", ondelete="CASCADE"), 
        index=True, 
        nullable=False  # 改為必填,確保有 user_id
    )

    # 移除 message_type - 改用 role 欄位
    bot_type = Column(String, nullable=True)
    content = Column(Text, nullable=False)
    
    # 移除 user_mood 和 mood_intensity - 這些資訊不需要在聊天訊息中
    
    # 核心欄位
    role = Column(String, nullable=False)  # "user" 或 "ai"
    mode = Column(String, default="text")  # "text" 或 "video"
    meta = Column(JSON, nullable=True)  # 儲存額外資訊
    
    created_at = Column(DateTime(timezone=True), nullable=False, default=datetime.utcnow)

    # 關聯
    user = relationship("User", back_populates="chat_messages", lazy="select")