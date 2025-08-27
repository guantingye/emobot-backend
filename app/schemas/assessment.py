from pydantic import BaseModel
from datetime import datetime
from typing import Any

class AssessmentUpsertIn(BaseModel):
    mbti_raw: str | None = None
    mbti_encoded: list[int] | None = None  # 例如 [0,1,1,1]
    step2Answers: dict | None = None
    step3Answers: dict | None = None
    step4Answers: dict | None = None
    ai_preference: dict | None = None
    submittedAt: datetime | None = None

class AssessmentOut(BaseModel):
    id: int
    mbti_raw: str | None
    mbti_encoded: Any | None
    submitted_at: datetime | None

    class Config:
        from_attributes = True
