# app/models/chat.py
# -*- coding: utf-8 -*-
from datetime import datetime
from sqlalchemy import Column, Integer, String, DateTime, Text, JSON

# ✅ 正確路徑：Base 在 app/db/base.py
from app.db.base import Base


class ChatMessage(Base):
    __tablename__ = "chat_messages"

    id         = Column(Integer, primary_key=True, index=True)
    user_id    = Column(Integer, index=True, nullable=False)
    bot_type   = Column(String(32), nullable=False)      # empathy / insight / solution / cognitive
    mode       = Column(String(16), default="text")      # text / video
    role       = Column(String(16), nullable=False)      # user / ai
    content    = Column(Text, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    meta       = Column(JSON, nullable=True)
