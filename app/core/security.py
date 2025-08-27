from datetime import datetime, timedelta
from fastapi import Depends, HTTPException
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from jose import jwt, JWTError
from sqlalchemy.orm import Session

from app.core.config import settings
from app.db.session import get_db
from app.models.user import User

ALGORITHM = "HS256"
security = HTTPBearer(auto_error=False)

def create_access_token(user: User) -> str:
    """產出與前端相容的 token：同時包含 sub 與 id"""
    expire = datetime.utcnow() + timedelta(minutes=int(settings.JWT_EXPIRE_MINUTES))
    payload = {
        "sub": str(user.id),        # 標準做法
        "id": user.id,              # 你的前端目前也會用到
        "pid": user.pid,
        "nickname": user.nickname,
        "role": "user",
        "exp": expire,
    }
    return jwt.encode(payload, settings.JWT_SECRET, algorithm=ALGORITHM)

def get_current_user(
    creds: HTTPAuthorizationCredentials | None = Depends(security),
    db: Session = Depends(get_db),
) -> User:
    if creds is None or creds.scheme.lower() != "bearer":
        raise HTTPException(status_code=401, detail="Not authenticated")
    token = creds.credentials
    try:
        payload = jwt.decode(token, settings.JWT_SECRET, algorithms=[ALGORITHM])
        uid_raw = payload.get("sub") or payload.get("id") or payload.get("user_id")
        user_id = int(uid_raw) if uid_raw is not None else None
    except JWTError:
        raise HTTPException(status_code=401, detail="Invalid token")
    except Exception:
        raise HTTPException(status_code=401, detail="Invalid token payload")

    if not user_id:
        raise HTTPException(status_code=401, detail="Invalid token payload")

    user = db.get(User, user_id)
    if not user:
        raise HTTPException(status_code=401, detail="User not found")
    return user
