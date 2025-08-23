# main.py — Emobot+ FastAPI（Postgres 版，CSV 以 API 匯出，不寫本地檔）
import os
import io
import csv
import json
import re
from datetime import datetime, timedelta
from typing import List, Optional, Any, Dict, Tuple

from fastapi import FastAPI, Depends, HTTPException, Header, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from dotenv import load_dotenv

import jwt  # PyJWT
from sqlalchemy import (
    create_engine, String, Integer, DateTime, Text, ForeignKey, select, func, UniqueConstraint
)
from sqlalchemy.orm import (
    DeclarativeBase, Mapped, mapped_column, relationship, sessionmaker, Session
)

# -------------------------
# 環境變數
# -------------------------
load_dotenv(override=True)

JWT_SECRET = os.getenv("JWT_SECRET", "dev-secret-change-me")

# 支援多個前端網域（逗號分隔）
_frontend = os.getenv("FRONTEND_URL", "*")
ALLOWED_ORIGINS = [o.strip() for o in _frontend.split(",")] if _frontend else ["*"]

def _normalize_db_url(url: str) -> str:
    """把 postgres:// 轉成 postgresql+psycopg2:// 並預設 sslmode=require（雲端 DB 常用）"""
    if url.startswith("postgres://"):
        url = url.replace("postgres://", "postgresql+psycopg2://", 1)
    if url.startswith("postgresql://"):
        url = url.replace("postgresql://", "postgresql+psycopg2://", 1)
    if url.startswith("postgresql+psycopg2://") and "sslmode=" not in url:
        # 非本機則強制 SSL
        if ("localhost" not in url) and ("127.0.0.1" not in url):
            sep = "&" if "?" in url else "?"
            url = f"{url}{sep}sslmode=require"
    return url

DATABASE_URL = _normalize_db_url(os.getenv("DATABASE_URL", "sqlite:///./emobot.db"))

# -------------------------
# 資料庫
# -------------------------
engine = create_engine(
    DATABASE_URL,
    connect_args={"check_same_thread": False} if DATABASE_URL.startswith("sqlite") else {}
)
SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False)

class Base(DeclarativeBase):
    pass

class User(Base):
    __tablename__ = "users"
    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    pid: Mapped[str] = mapped_column(String(16), unique=True, index=True)   # 受試者ID（兩碼數字＋兩碼英文）
    nickname: Mapped[str] = mapped_column(String(100))
    role: Mapped[str] = mapped_column(String(16), default="user")           # user / admin
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    assessment: Mapped["Assessment"] = relationship(back_populates="user", uselist=False)

class Assessment(Base):
    __tablename__ = "assessments"
    __table_args__ = (UniqueConstraint("user_id", name="uq_assessment_user"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    user_id: Mapped[int] = mapped_column(ForeignKey("users.id"), index=True)

    # Step1 MBTI
    mbti_raw: Mapped[Optional[str]] = mapped_column(String(8), nullable=True)   # 例如 "ENFP"
    mbti_encoded: Mapped[Optional[str]] = mapped_column(String(64), nullable=True)  # JSON 字串 "[1,1,1,0]"

    # Step2/3/4：儲存為 JSON 字串
    step2_answers: Mapped[Optional[str]] = mapped_column(Text, nullable=True)   # 1~7
    step3_answers: Mapped[Optional[str]] = mapped_column(Text, nullable=True)   # 1~5
    step4_answers: Mapped[Optional[str]] = mapped_column(Text, nullable=True)   # 1~7

    submitted_at: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)

    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    user: Mapped[User] = relationship(back_populates="assessment")

class MatchChoice(Base):
    __tablename__ = "match_choices"
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    user_id: Mapped[int] = mapped_column(ForeignKey("users.id"), index=True)
    bot_type: Mapped[str] = mapped_column(String(32))  # empathy / insight / solution / cognitive
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)

    user: Mapped[User] = relationship()

class MatchRecLog(Base):
    __tablename__ = "match_recommend_logs"
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    user_id: Mapped[int] = mapped_column(ForeignKey("users.id"), index=True)
    ranked_json: Mapped[str] = mapped_column(Text)  # 儲存前端拿到的推薦結果（完整 JSON）
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)

    user: Mapped[User] = relationship()

Base.metadata.create_all(bind=engine)

# -------------------------
# FastAPI & CORS
# -------------------------
app = FastAPI(title="Emobot+ API (FastAPI)")

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -------------------------
# DB 依賴
# -------------------------
def get_db() -> Session:
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# -------------------------
# JWT
# -------------------------
def create_token(payload: Dict[str, Any], days: int = 30) -> str:
    to_encode = payload.copy()
    to_encode["exp"] = datetime.utcnow() + timedelta(days=days)
    return jwt.encode(to_encode, JWT_SECRET, algorithm="HS256")

def decode_token(token: str) -> Dict[str, Any]:
    return jwt.decode(token, JWT_SECRET, algorithms=["HS256"])

def auth_required(authorization: Optional[str] = Header(None), db: Session = Depends(get_db)) -> User:
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="No token")
    token = authorization.replace("Bearer ", "", 1).strip()
    try:
        payload = decode_token(token)
    except jwt.PyJWTError:
        raise HTTPException(status_code=401, detail="Invalid token")

    user_id = payload.get("id")
    if not user_id:
        raise HTTPException(status_code=401, detail="Invalid token payload")

    user = db.get(User, user_id)
    if not user:
        raise HTTPException(status_code=401, detail="User not found")
    return user

def admin_required(user: User = Depends(auth_required)) -> User:
    if user.role != "admin":
        raise HTTPException(status_code=403, detail="Admin only")
    return user

# -------------------------
# Schemas
# -------------------------
class JoinRequest(BaseModel):
    pid: str = Field(..., description="受試者ID（兩碼數字＋兩碼英文，例如 12AB）")
    nickname: str

class JoinResponseUser(BaseModel):
    id: int
    pid: str
    nickname: str
    role: str

class JoinResponse(BaseModel):
    token: str
    user: JoinResponseUser

class MbtiPayload(BaseModel):
    raw: str
    encoded: List[int] = Field(min_length=4, max_length=4)

class UpsertAssessmentRequest(BaseModel):
    mbti: Optional[MbtiPayload] = None
    step2Answers: Optional[List[int]] = None
    step3Answers: Optional[List[int]] = None
    step4Answers: Optional[List[int]] = None
    submittedAt: Optional[datetime] = None

# -------------------------
# 根路由（顯示 DB 提要方便確認）
# -------------------------
@app.get("/")
def root():
    return {
        "ok": True,
        "name": "Emobot+ API (FastAPI)",
        "db": DATABASE_URL.split("://", 1)[0] + "://***",
    }

# -------------------------
# Auth：暱稱＋受試者ID（兩碼數字＋兩碼英文）
# -------------------------
PID_PATTERN = re.compile(r"^\d{2}[A-Za-z]{2}$")

@app.post("/api/auth/join", response_model=JoinResponse)
def join(payload: JoinRequest, db: Session = Depends(get_db)):
    pid = payload.pid.strip().upper()
    nickname = payload.nickname.strip()

    if not PID_PATTERN.match(pid):
        raise HTTPException(status_code=400, detail="Invalid pid format (e.g., 12AB)")

    user = db.scalar(select(User).where(User.pid == pid))
    if user:
        user.nickname = nickname
    else:
        user = User(pid=pid, nickname=nickname, role="user")
        db.add(user)

    db.commit()
    db.refresh(user)

    token = create_token({"id": user.id, "pid": user.pid, "nickname": user.nickname, "role": user.role})
    return {"token": token, "user": {"id": user.id, "pid": user.pid, "nickname": user.nickname, "role": user.role}}

# -------------------------
# 工具：Assessment 轉輸出
# -------------------------
def _assessment_to_out(a: Optional[Assessment]) -> Optional[Dict[str, Any]]:
    if not a:
        return None
    def loads(s: Optional[str]) -> Optional[List[int]]:
        return json.loads(s) if s else None

    mbti = None
    if a.mbti_raw or a.mbti_encoded:
        mbti = {"raw": a.mbti_raw, "encoded": loads(a.mbti_encoded) or []}

    return {
        "userId": a.user_id,
        "mbti": mbti,
        "step2Answers": loads(a.step2_answers),
        "step3Answers": loads(a.step3_answers),
        "step4Answers": loads(a.step4_answers),
        "submittedAt": a.submitted_at,
        "createdAt": a.created_at,
        "updatedAt": a.updated_at,
    }

# -------------------------
# 作答 upsert（分步；不再寫本地 CSV）
# -------------------------
@app.post("/api/assessments/upsert")
def upsert_assessment(
    payload: UpsertAssessmentRequest,
    user: User = Depends(auth_required),
    db: Session = Depends(get_db)
):
    a = db.scalar(select(Assessment).where(Assessment.user_id == user.id))
    if not a:
        a = Assessment(user_id=user.id)
        db.add(a)

    if payload.mbti:
        a.mbti_raw = payload.mbti.raw.upper().strip()
        a.mbti_encoded = json.dumps(payload.mbti.encoded)

    if payload.step2Answers is not None:
        a.step2_answers = json.dumps(payload.step2Answers)
    if payload.step3Answers is not None:
        a.step3_answers = json.dumps(payload.step3Answers)
    if payload.step4Answers is not None:
        a.step4_answers = json.dumps(payload.step4Answers)
    if payload.submittedAt is not None:
        a.submitted_at = payload.submittedAt

    db.commit()
    db.refresh(a)
    return {"ok": True, "data": _assessment_to_out(a)}

# -------------------------
# 便於自查：查自己的作答
# -------------------------
@app.get("/api/assessments/me")
def my_assessment(user: User = Depends(auth_required), db: Session = Depends(get_db)):
    a = db.scalar(select(Assessment).where(Assessment.user_id == user.id))
    return {"data": _assessment_to_out(a)} if a else {"data": None}

# -------------------------
# 後台列表（需 admin）
# -------------------------
@app.get("/api/admin/assessments")
def list_assessments(
    page: int = Query(1, ge=1),
    limit: int = Query(20, ge=1, le=100),
    _: User = Depends(admin_required),
    db: Session = Depends(get_db)
):
    total = db.scalar(select(func.count(Assessment.id)))
    items = db.scalars(
        select(Assessment)
        .order_by(Assessment.created_at.desc())
        .offset((page - 1) * limit)
        .limit(limit)
    ).all()

    return {
        "page": page,
        "limit": limit,
        "total": total or 0,
        "items": [_assessment_to_out(a) for a in items]
    }

# -------------------------
# 匯出 CSV（後台）— 即時產生，不落地
# -------------------------
@app.get("/api/admin/assessments/export-csv")
def export_csv_all(_: User = Depends(admin_required), db: Session = Depends(get_db)):
    rows = db.scalars(select(Assessment)).all()

    def loads(s): 
        return json.loads(s) if s else None

    sio = io.StringIO()
    writer = csv.writer(sio)
    writer.writerow([
        "submittedAt","userId","pid","nickname",
        "mbti_raw","mbti_encoded",
        "step2Answers","step3Answers","step4Answers",
        "createdAt","updatedAt",
    ])
    for a in rows:
        user = db.get(User, a.user_id)
        if not user: 
            continue
        writer.writerow([
            a.submitted_at.isoformat() if a.submitted_at else "",
            user.id, user.pid, user.nickname,
            a.mbti_raw or "",
            a.mbti_encoded or "",
            json.dumps(loads(a.step2_answers) or [], ensure_ascii=False),
            json.dumps(loads(a.step3_answers) or [], ensure_ascii=False),
            json.dumps(loads(a.step4_answers) or [], ensure_ascii=False),
            a.created_at.isoformat() if a.created_at else "",
            a.updated_at.isoformat() if a.updated_at else "",
        ])

    sio.seek(0)
    headers = {"Content-Disposition": 'attachment; filename="assessments.csv"'}
    return StreamingResponse(iter([sio.getvalue()]), media_type="text/csv", headers=headers)

# =========================================================
# =====           媒合演算法（四型分數）              =====
# =========================================================

# --- 特徵清單與基礎權重（可調整） ---
FEATURES: List[str] = [
    "distress", "self_doubt", "attach_anxiety", "attach_avoidance",
    "extraversion", "intuition", "thinking"
]

BASE_WEIGHTS: Dict[str, float] = {
    "distress": 1.0,
    "self_doubt": 0.9,
    "attach_anxiety": 0.9,
    "attach_avoidance": 0.8,
    "extraversion": 0.5,
    "intuition": 0.6,
    "thinking": 0.6,
}

# --- 四型原型向量（0~1） ---
PROTOTYPES: Dict[str, Dict[str, float]] = {
    "empathy": {   # 同理型
        "distress": 1.0, "self_doubt": 0.9, "attach_anxiety": 0.9, "attach_avoidance": 0.4,
        "extraversion": 0.4, "intuition": 0.6, "thinking": 0.3
    },
    "insight": {   # 洞察型
        "distress": 0.6, "self_doubt": 0.9, "attach_anxiety": 0.6, "attach_avoidance": 0.5,
        "extraversion": 0.5, "intuition": 0.9, "thinking": 0.5
    },
    "solution": {  # 解決型
        "distress": 0.8, "self_doubt": 0.6, "attach_anxiety": 0.5, "attach_avoidance": 0.4,
        "extraversion": 0.5, "intuition": 0.4, "thinking": 0.9
    },
    "cognitive": { # 認知型
        "distress": 0.5, "self_doubt": 0.5, "attach_anxiety": 0.4, "attach_avoidance": 0.6,
        "extraversion": 0.3, "intuition": 0.3, "thinking": 0.9
    },
}

# --- Step2 次量表的題目索引（0-based）。若未知可留空，退回整體平均 ---
ATTACH_ANXIETY_IDX: List[int] = []
ATTACH_AVOID_IDX:   List[int] = []

def _safe_avg(nums: List[Optional[float]]) -> Optional[float]:
    vv = [x for x in nums if x is not None]
    return sum(vv)/len(vv) if vv else None

def _minmax(x: Optional[float], lo: float, hi: float) -> Optional[float]:
    if x is None: return None
    if hi == lo: return 0.5
    x = max(min(x, hi), lo)
    return (x - lo) / (hi - lo)

def _cosine(a: Dict[str, float], b: Dict[str, float], weights: Dict[str, float]) -> float:
    num = 0.0; da = 0.0; db = 0.0
    for k, w in weights.items():
        va = a.get(k); vb = b.get(k)
        if va is None or vb is None: 
            continue
        num += w * va * vb
        da  += w * va * va
        db  += w * vb * vb
    if da == 0 or db == 0: 
        return 0.0
    return num / ((da ** 0.5) * (db ** 0.5))

def compute_features(assessment: dict) -> Dict[str, Optional[float]]:
    mbti = assessment.get("mbti") or {}
    enc  = mbti.get("encoded") or [None, None, None, None]
    E, N, T, P = enc + [None]*(4-len(enc))

    step2 = assessment.get("step2Answers")
    step3 = assessment.get("step3Answers")
    step4 = assessment.get("step4Answers")

    s2_avg = _safe_avg([float(x) for x in step2]) if step2 else None
    s3_avg = _safe_avg([float(x) for x in step3]) if step3 else None
    s4_avg = _safe_avg([float(x) for x in step4]) if step4 else None

    def subscale(avg_idx: List[int], arr: Optional[List[int]]) -> Optional[float]:
        if not arr: return None
        if not avg_idx:
            return _safe_avg([float(x) for x in arr])
        vals = [float(arr[i]) for i in avg_idx if 0 <= i < len(arr) and arr[i] is not None]
        return _safe_avg(vals)

    anx_raw = subscale(ATTACH_ANXIETY_IDX, step2)
    avo_raw = subscale(ATTACH_AVOID_IDX,   step2)

    features: Dict[str, Optional[float]] = {
        "distress":        _minmax(s4_avg, 1, 7),
        "self_doubt":      _minmax(s3_avg, 1, 5),
        "attach_anxiety":  _minmax(anx_raw, 1, 7),
        "attach_avoidance":_minmax(avo_raw, 1, 7),
        "extraversion":    float(E) if E in (0,1) else None,
        "intuition":       float(N) if N in (0,1) else None,
        "thinking":        float(T) if T in (0,1) else None,
    }
    return features

def score_bots(features: Dict[str, Optional[float]]) -> Tuple[Dict[str, float], Dict[str, List[Tuple[str, float]]]]:
    scores: Dict[str, float] = {}
    top_feats: Dict[str, List[Tuple[str, float]]] = {}

    avail = {k: v for k, v in features.items() if v is not None}
    if not avail:
        return {k: 50.0 for k in PROTOTYPES}, {k: [] for k in PROTOTYPES}

    weights = {k: BASE_WEIGHTS.get(k, 1.0) for k in avail.keys()}
    wsum = sum(weights.values()) or 1.0
    weights = {k: w/wsum for k, w in weights.items()}

    for bot, proto in PROTOTYPES.items():
        proto_sub = {k: proto[k] for k in avail.keys()}
        sim = _cosine(avail, proto_sub, weights)  # 0~1
        scores[bot] = round(sim*100, 1)

        contrib = []
        for k in avail.keys():
            contrib.append((k, 1.0 - abs(avail[k] - proto[k])))
        contrib.sort(key=lambda x: x[1], reverse=True)
        top_feats[bot] = contrib[:2]

    return scores, top_feats

def label_map(key: str) -> str:
    mapping = {
        "distress": "困擾程度",
        "self_doubt": "自我懷疑",
        "attach_anxiety": "依附焦慮",
        "attach_avoidance": "依附迴避",
        "extraversion": "外向傾向",
        "intuition": "直覺傾向",
        "thinking": "理性思維",
    }
    return mapping.get(key, key)

# ---- 推薦分數 ----
@app.post("/api/match/recommend")
def match_recommend(user: User = Depends(auth_required), db: Session = Depends(get_db)):
    a = db.scalar(select(Assessment).where(Assessment.user_id == user.id))
    if not a:
        raise HTTPException(status_code=400, detail="No assessment yet")

    data = _assessment_to_out(a) or {}
    feats = compute_features(data)
    scores, top_feats = score_bots(feats)

    names = {
        "empathy": "同理型",
        "insight": "洞察型",
        "solution": "解決型",
        "cognitive": "認知型",
    }
    explain = {}
    for k, items in top_feats.items():
        if not items:
            explain[k] = ""
            continue
        txt = "、".join([label_map(i[0]) for i in items])
        explain[k] = f"根據你的{txt}表現，此型態可能較適合。"

    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)

    return {
        "features": feats,
        "scores": scores,
        "ranked": [{"type": k, "name": names.get(k, k), "score": v, "why": explain.get(k, "")} for k, v in ranked]
    }

# ---- 儲存推薦結果（只寫 DB，不寫檔） ----
@app.post("/api/match/log")
def match_log(payload: Dict[str, Any], user: User = Depends(auth_required), db: Session = Depends(get_db)):
    rec = MatchRecLog(user_id=user.id, ranked_json=json.dumps(payload, ensure_ascii=False))
    db.add(rec)
    db.commit()
    db.refresh(rec)
    return {"ok": True, "loggedAt": rec.created_at.isoformat()}

# ---- 使用者選擇 ----
@app.post("/api/match/choose")
def match_choose(payload: Dict[str, str], user: User = Depends(auth_required), db: Session = Depends(get_db)):
    bot_type = (payload.get("botType") or "").strip().lower()
    if bot_type not in ("empathy", "insight", "solution", "cognitive"):
        raise HTTPException(status_code=400, detail="Invalid botType")

    rec = db.scalar(select(MatchChoice).where(MatchChoice.user_id == user.id))
    if rec:
        rec.bot_type = bot_type
        rec.created_at = datetime.utcnow()
    else:
        rec = MatchChoice(user_id=user.id, bot_type=bot_type)
        db.add(rec)

    db.commit()
    db.refresh(rec)
    return {"ok": True, "choice": {"botType": rec.bot_type, "chosenAt": rec.created_at.isoformat()}}

@app.get("/api/match/me")
def match_me(user: User = Depends(auth_required), db: Session = Depends(get_db)):
    rec = db.scalar(select(MatchChoice).where(MatchChoice.user_id == user.id))
    return {"choice": {"botType": rec.bot_type, "chosenAt": rec.created_at.isoformat()}} if rec else {"choice": None}

# ---- 匯出推薦紀錄 CSV（後台）
@app.get("/api/admin/match-logs/export-csv")
def export_match_log(_: User = Depends(admin_required), db: Session = Depends(get_db)):
    logs = db.scalars(select(MatchRecLog).order_by(MatchRecLog.created_at.desc())).all()
    sio = io.StringIO()
    w = csv.writer(sio)
    w.writerow(["createdAt","userId","pid","nickname","topType","topScore","rankedJson"])
    for row in logs:
        user = db.get(User, row.user_id)
        ranked = []
        top_type = ""; top_score = ""
        try:
            ranked = json.loads(row.ranked_json).get("ranked") or []
            if ranked:
                top_type = ranked[0].get("type", "")
                top_score = ranked[0].get("score", "")
        except Exception:
            pass
        w.writerow([
            row.created_at.isoformat(),
            user.id if user else "", user.pid if user else "", user.nickname if user else "",
            top_type, top_score,
            json.dumps(ranked, ensure_ascii=False),
        ])
    sio.seek(0)
    headers = {"Content-Disposition": 'attachment; filename="match_recommendations.csv"'}
    return StreamingResponse(iter([sio.getvalue()]), media_type="text/csv", headers=headers)

# -------------------------
# 啟動（Render）
# Build: pip install -r requirements.txt
# Start: uvicorn main:app --host 0.0.0.0 --port $PORT
# 環境變數：DATABASE_URL（Postgres）、FRONTEND_URL、JWT_SECRET、PYTHON_VERSION=3.12.5
# -------------------------
