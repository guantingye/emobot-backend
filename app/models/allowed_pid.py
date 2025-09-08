# app/models/allowed_pid.py
from sqlalchemy import Column, Integer, String, DateTime, Boolean, func
from sqlalchemy.orm import relationship
from app.db.base import Base

class AllowedPid(Base):
    __tablename__ = "allowed_pids"
    
    id = Column(Integer, primary_key=True, index=True)
    pid = Column(String, unique=True, index=True, nullable=False)
    is_active = Column(Boolean, default=True, nullable=False)
    description = Column(String, nullable=True)  # 備註說明
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())


