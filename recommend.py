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
# 基於常見的依戀量表，這裡提供一個合理的分組
ATTACH_ANXIETY_IDX: List[int] = [0, 3, 6, 8, 11, 14, 17, 19]   # 焦慮依戀相關題目
ATTACH_AVOID_IDX: List[int] = [2, 4, 5, 7, 9, 10, 13, 16, 21]   # 迴避依戀相關題目

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

def compute_features(assessment_data: dict) -> Dict[str, Optional[float]]:
    """把你的 Assessment 轉為 0~1 特徵；缺就回 None。"""
    print(f"Computing features from: {assessment_data}")
    
    # 從傳入的 assessment_data 中提取數據
    mbti = assessment_data.get("mbti", {})
    if isinstance(mbti, dict):
        enc = mbti.get("encoded", [None, None, None, None])
    else:
        enc = [None, None, None, None]
    
    if len(enc) < 4:
        enc = enc + [None] * (4 - len(enc))
    E, N, T, P = enc[0], enc[1], enc[2], enc[3]

    step2 = assessment_data.get("step2Answers") or assessment_data.get("step2")
    step3 = assessment_data.get("step3Answers") or assessment_data.get("step3") 
    step4 = assessment_data.get("step4Answers") or assessment_data.get("step4")

    print(f"Extracted data - E:{E}, N:{N}, T:{T}, P:{P}")
    print(f"Step2 length: {len(step2) if step2 else 0}")
    print(f"Step3 length: {len(step3) if step3 else 0}")
    print(f"Step4 length: {len(step4) if step4 else 0}")

    # 平均分 (原量尺：Step2/4 是 1~7；Step3 假設 1~5)
    s2_avg = _safe_avg([float(x) for x in step2]) if step2 else None
    s3_avg = _safe_avg([float(x) for x in step3]) if step3 else None
    s4_avg = _safe_avg([float(x) for x in step4]) if step4 else None

    print(f"Averages - Step2: {s2_avg}, Step3: {s3_avg}, Step4: {s4_avg}")

    # 次量表（若無索引，退回整體平均）
    def subscale(avg_idx: List[int], arr: Optional[List], scale_name: str = "") -> Optional[float]:
        if not arr: 
            print(f"  {scale_name}: No data")
            return None
        if not avg_idx:
            result = _safe_avg([float(x) for x in arr])
            print(f"  {scale_name}: Using full average = {result}")
            return result
        vals = []
        for i in avg_idx:
            if 0 <= i < len(arr) and arr[i] is not None:
                vals.append(float(arr[i]))
        result = _safe_avg(vals) if vals else None
        print(f"  {scale_name}: Using subscale indices {avg_idx} = {result}")
        return result

    anx_raw = subscale(ATTACH_ANXIETY_IDX, step2, "attach_anxiety")
    avo_raw = subscale(ATTACH_AVOID_IDX, step2, "attach_avoidance")

    # 標準化到 0~1
    # 注意：這裡的標準化方向需要根據你的量表特性調整
    # 如果高分代表更多困擾/迴避，那麼直接標準化
    # 如果需要反向，可以用 1.0 - _minmax(...)
    
    features: Dict[str, Optional[float]] = {
        "distress":        _minmax(s4_avg, 1, 7) if s4_avg else None,  # 基本心理需求，可能需要反向
        "self_doubt":      _minmax(s3_avg, 1, 5) if s3_avg else None,  # 情緒調節策略
        "attach_anxiety":  _minmax(anx_raw, 1, 7) if anx_raw else None,  # 焦慮依戀
        "attach_avoidance":_minmax(avo_raw, 1, 7) if avo_raw else None,  # 迴避依戀
        "extraversion":    float(E) if E in (0,1) else None,
        "intuition":       float(N) if N in (0,1) else None,
        "thinking":        float(T) if T in (0,1) else None,
    }

    print(f"Computed features: {features}")
    return features

def score_bots(features: Dict[str, Optional[float]]) -> Tuple[Dict[BotKey, float], Dict[BotKey, List[Tuple[str, float]]]]:
    """回傳：各 bot 分數(0-100) 與特徵貢獻（前2項）。"""
    print(f"Scoring bots with features: {features}")
    
    scores: Dict[BotKey, float] = {}
    top_feats: Dict[BotKey, List[Tuple[str, float]]] = {}

    # 只使用「有值」的特徵來做加權（動態重分配）
    avail = {k: v for k, v in features.items() if v is not None}
    if not avail:
        print("No available features, returning default scores")
        return {k: 50.0 for k in PROTOTYPES}, {k: [] for k in PROTOTYPES}

    print(f"Available features: {avail}")

    # 依可用特徵重設權重
    weights = {k: BASE_WEIGHTS.get(k, 1.0) for k in avail.keys()}
    wsum = sum(weights.values())
    if wsum > 0:
        weights = {k: w/wsum for k, w in weights.items()}
    else:
        weights = {k: 1.0/len(avail) for k in avail.keys()}

    print(f"Normalized weights: {weights}")

    for bot, proto in PROTOTYPES.items():
        proto_sub = {k: proto[k] for k in avail.keys()}
        sim = _cosine(avail, proto_sub, weights)  # 0~1
        score = round(sim*100, 1)
        scores[bot] = max(0.0, min(100.0, score))  # 確保在 0-100 範圍內

        print(f"Bot {bot}: similarity = {sim:.3f}, score = {score}")

        # 特徵貢獻（粗略：|user-feature - prototype| 越小貢獻越大）
        contrib = []
        for k in avail.keys():
            user_val = avail[k]
            proto_val = proto[k]
            contribution = 1.0 - abs(user_val - proto_val)
            contrib.append((k, contribution))
        contrib.sort(key=lambda x: x[1], reverse=True)
        top_feats[bot] = contrib[:2]

    print(f"Final scores: {scores}")
    return scores, top_feats