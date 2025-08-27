from pydantic import BaseModel

class UserOut(BaseModel):
    id: int
    pid: str
    nickname: str
    selected_bot: str | None

    class Config:
        from_attributes = True
