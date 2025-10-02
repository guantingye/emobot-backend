# backend/app/core/security.py
from datetime import datetime, timedelta
from jose import jwt, JWTError
from fastapi import Depends, HTTPException
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from sqlalchemy.orm import Session
from app.core.config import settings
from app.db.session import get_db
from app.models.user import User

ALGORITHM = settings.JWT_ALG
_security = HTTPBearer(auto_error=False)

def create_access_token(pid: str) -> str:
    """建立 JWT token,只需要 PID"""
    expire = datetime.utcnow() + timedelta(minutes=settings.JWT_EXPIRE_MINUTES)
    payload = {"pid": pid, "exp": expire}
    return jwt.encode(payload, settings.JWT_SECRET, algorithm=ALGORITHM)

def get_current_user(
    creds: HTTPAuthorizationCredentials | None = Depends(_security),
    db: Session = Depends(get_db),
) -> User:
    """從 JWT 中取得當前用戶"""
    if creds is None or creds.scheme.lower() != "bearer":
        raise HTTPException(status_code=401, detail="Not authenticated")
    
    try:
        payload = jwt.decode(creds.credentials, settings.JWT_SECRET, algorithms=[ALGORITHM])
        pid = payload.get("pid")
        if not pid:
            raise HTTPException(status_code=401, detail="Invalid token: missing PID")
    except JWTError as e:
        raise HTTPException(status_code=401, detail=f"Invalid token: {str(e)}")
    
    user = db.query(User).filter(User.pid == pid).first()
    if not user:
        raise HTTPException(status_code=401, detail="User not found")
    
    return user