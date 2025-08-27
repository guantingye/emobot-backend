from pydantic import BaseModel
from datetime import datetime

class ChatMessageIn(BaseModel):
    message_type: str  # "user" 或 "bot"
    bot_type: str | None = None
    content: str
    user_mood: str | None = None
    mood_intensity: int | None = None

class ChatMessageOut(BaseModel):
    id: int
    message_type: str
    bot_type: str | None
    content: str
    user_mood: str | None
    mood_intensity: int | None
    created_at: datetime

    class Config:
        from_attributes = True
