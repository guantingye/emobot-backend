from pydantic import BaseModel
from typing import Dict, Any

class RecommendOut(BaseModel):
    scores: Dict[str, float]
    top: str
    features: Any | None = None