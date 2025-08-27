from pydantic import BaseModel
from datetime import datetime

class MoodRecordIn(BaseModel):
    mood: str
    intensity: int | None = None
    note: str | None = None

class MoodRecordOut(BaseModel):
    id: int
    mood: str
    intensity: int | None
    note: str | None
    created_at: datetime

    class Config:
        from_attributes = True
