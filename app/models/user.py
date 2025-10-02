# backend/app/models/user.py
from sqlalchemy import Column, String, DateTime
from sqlalchemy.sql import func
from app.db.base import Base

class User(Base):
    __tablename__ = "users"
    
    pid = Column(String(10), primary_key=True, index=True)
    nickname = Column(String(50))
    selected_bot = Column(String(20))
    last_login_at = Column(DateTime(timezone=True), server_default=func.now())
    created_at = Column(DateTime(timezone=True), server_default=func.now())