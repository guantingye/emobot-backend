# app/models/chat_session.py
from datetime import datetime
from sqlalchemy import Column, Integer, String, DateTime, Boolean, ForeignKey, func
from sqlalchemy.orm import relationship
from app.db.base import Base

class ChatSession(Base):
    __tablename__ = "chat_sessions"
    
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(
        Integer, 
        ForeignKey("users.id", ondelete="CASCADE"), 
        index=True, 
        nullable=False
    )
    bot_type = Column(String, nullable=True)
    
    # 時間追蹤欄位
    session_start = Column(DateTime(timezone=True), nullable=False, default=datetime.utcnow)
    session_end = Column(DateTime(timezone=True), nullable=True)
    last_activity = Column(DateTime(timezone=True), nullable=False, default=datetime.utcnow)
    
    # 會話狀態
    is_active = Column(Boolean, default=True, nullable=False)
    end_reason = Column(String, nullable=True)  # 'user_ended', 'timeout', 'system'
    
    # 統計資料
    message_count = Column(Integer, default=0, nullable=False)
    
    created_at = Column(DateTime(timezone=True), nullable=False, default=datetime.utcnow)
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())
    
    # 關聯
    user = relationship("User", back_populates="chat_sessions", lazy="select")
    
    @property
    def duration_minutes(self) -> float:
        """計算會話持續時間（分鐘）"""
        if self.session_end:
            delta = self.session_end - self.session_start
        else:
            delta = datetime.utcnow() - self.session_start
        return delta.total_seconds() / 60.0