# app/models/chat.py  
from datetime import datetime
from sqlalchemy import Column, Integer, String, DateTime, Text, JSON, ForeignKey, func  # ★ 添加 func import
from sqlalchemy.orm import relationship
from app.db.base import Base

class ChatMessage(Base):
    __tablename__ = "chat_messages"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(
        Integer, 
        ForeignKey("users.id", ondelete="CASCADE"), 
        index=True, 
        nullable=True
    )

    message_type = Column(String, nullable=True, default="user")
    bot_type = Column(String, nullable=True)
    content = Column(Text, nullable=False)
    user_mood = Column(String, nullable=True)
    mood_intensity = Column(Integer, nullable=True)
    
    # 新增的欄位
    role = Column(String, nullable=False)
    mode = Column(String, default="text")  
    meta = Column(JSON, nullable=True)
    
    created_at = Column(DateTime(timezone=True), nullable=False, default=datetime.utcnow)  

    user = relationship("User", back_populates="chat_messages", lazy="select")