# app/models/chat.py - 加入 PID 欄位
from datetime import datetime
from sqlalchemy import Column, Integer, String, DateTime, Text, JSON, ForeignKey
from sqlalchemy.orm import relationship
from app.db.base import Base

class ChatMessage(Base):
    __tablename__ = "chat_messages"

    id = Column(Integer, primary_key=True, index=True)
    
    # 用戶識別 - 同時保留 user_id 和 pid
    user_id = Column(
        Integer, 
        ForeignKey("users.id", ondelete="CASCADE"), 
        index=True, 
        nullable=False
    )
    pid = Column(String, index=True, nullable=True)  # ✅ 新增: 直接記錄 PID
    
    # 訊息內容
    bot_type = Column(String, nullable=True)
    content = Column(Text, nullable=False)
    
    # 訊息屬性
    role = Column(String, nullable=False)  # "user" 或 "ai"
    mode = Column(String, default="text")  # "text" 或 "video"
    meta = Column(JSON, nullable=True)
    
    # 時間 - 使用 timezone aware
    created_at = Column(DateTime(timezone=True), nullable=False, default=datetime.utcnow)

    # 關聯
    user = relationship("User", back_populates="chat_messages", lazy="select")