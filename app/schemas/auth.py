from pydantic import BaseModel

class JoinIn(BaseModel):
    pid: str
    nickname: str

class TokenOut(BaseModel):
    token: str
