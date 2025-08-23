# recommend.py
from __future__ import annotations
from typing import Dict, List, Optional, Tuple
import math, json

BotKey = str

# ==== 你可以微調的權重與原型 ====

FEATURES: List[str] = [
    "distress", "self_doubt", "attach_anxiety", "attach_avoidance",
    "extraversion", "intuition", "thinking"
]

# 各特徵的基礎權重（會被 bot-specific 調整）
BASE_WEIGHTS: Dict[str, float] = {
    "distress": 1.0,
    "self_doubt": 0.9,
    "attach_anxiety": 0.9,
    "attach_avoidance": 0.8,
    "extraversion": 0.5,
    "intuition": 0.6,
    "thinking": 0.6,
}

# 四型的原型向量（0~1），以及特徵加權（可逐步校正）
PROTOTYPES: Dict[BotKey, Dict[str, float]] = {
    "empathy": {  # 同理型
        "distress": 1.0, "self_doubt": 0.9, "attach_anxiety": 0.9, "attach_avoidance": 0.4,
        "extraversion": 0.4, "intuition": 0.6, "thinking": 0.3
    },
    "insight": {  # 洞察型
        "distress": 0.6, "self_doubt": 0.9, "attach_anxiety": 0.6, "attach_avoidance": 0.5,
        "extraversion": 0.5, "intuition": 0.9, "thinking": 0.5
    },
    "solution": {  # 解決型
        "distress": 0.8, "self_doubt": 0.6, "attach_anxiety": 0.5, "attach_avoidance": 0.4,
        "extraversion": 0.5, "intuition": 0.4, "thinking": 0.9
    },
    "cognitive": {  # 認知型（結構化重組思維）
        "distress": 0.5, "self_doubt": 0.5, "attach_anxiety": 0.4, "attach_avoidance": 0.6,
        "extraversion": 0.3, "intuition": 0.3, "thinking": 0.9
    },
}

# === 你要把 Step2 的題目索引對應到兩個次量表（0-based index）===
# 例：attach_anxiety 由哪些題目平均、attach_avoidance 由哪些題目平均
# 先給占位，請依你的量表題號填入；若留空，會 fallback 用整體平均。
ATTACH_ANXIETY_IDX: List[int] = []   # e.g., [1,3,6,7,12,18,20]
ATTACH_AVOID_IDX:   List[int] = []   # e.g., [0,8,10,11,14,16,21]

# ==== 特徵工程 ====

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
    return num / (math.sqrt(da) * math.sqrt(db))

def compute_features(assessment: dict) -> Dict[str, Optional[float]]:
    """把你的 Assessment 轉為 0~1 特徵；缺就回 None。"""
    mbti = assessment.get("mbti") or {}
    enc  = mbti.get("encoded") or [None, None, None, None]
    E, N, T, P = enc + [None]*(4-len(enc))

    step2 = assessment.get("step2Answers")
    step3 = assessment.get("step3Answers")
    step4 = assessment.get("step4Answers")

    # 平均分 (原量尺：Step2/4 是 1~7；Step3 假設 1~5)
    s2_avg = _safe_avg(step2) if step2 else None
    s3_avg = _safe_avg(step3) if step3 else None
    s4_avg = _safe_avg(step4) if step4 else None

    # 次量表（若無索引，退回整體平均）
    def subscale(avg_idx: List[int], arr: Optional[List[int]], scale_max: int) -> Optional[float]:
        if not arr: return None
        if not avg_idx:
            return _safe_avg([float(x) for x in arr])
        vals = [float(arr[i]) for i in avg_idx if 0 <= i < len(arr) and arr[i] is not None]
        return _safe_avg(vals)

    anx_raw = subscale(ATTACH_ANXIETY_IDX, step2, 7)
    avo_raw = subscale(ATTACH_AVOID_IDX,   step2, 7)

    # 標準化到 0~1
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

def score_bots(features: Dict[str, Optional[float]]) -> Tuple[Dict[BotKey, float], Dict[BotKey, List[Tuple[str, float]]]]:
    """回傳：各 bot 分數(0-100) 與特徵貢獻（前2項）。"""
    scores: Dict[BotKey, float] = {}
    top_feats: Dict[BotKey, List[Tuple[str, float]]] = {}

    # 只使用「有值」的特徵來做加權（動態重分配）
    avail = {k: v for k, v in features.items() if v is not None}
    if not avail:
        return {k: 50.0 for k in PROTOTYPES}, {k: [] for k in PROTOTYPES}

    # 依可用特徵重設權重
    weights = {k: BASE_WEIGHTS.get(k, 1.0) for k in avail.keys()}
    wsum = sum(weights.values())
    weights = {k: w/wsum for k, w in weights.items()}

    for bot, proto in PROTOTYPES.items():
        proto_sub = {k: proto[k] for k in avail.keys()}
        sim = _cosine(avail, proto_sub, weights)  # 0~1
        scores[bot] = round(sim*100, 1)

        # 特徵貢獻（粗略：|user-feature - prototype| 越小貢獻越大）
        contrib = []
        for k in avail.keys():
            contrib.append((k, 1.0 - abs(avail[k] - proto[k])))
        contrib.sort(key=lambda x: x[1], reverse=True)
        top_feats[bot] = contrib[:2]

    return scores, top_feats
