# backend/app/services/recommendation_engine.py
from __future__ import annotations
from typing import Dict, List, Any

CONTRAST_GAMMA = 1.35
VALID_BOTS = ("empathy", "insight", "solution", "cognitive")

# 反向題設定
DERS_REVERSE_1B = [1, 3, 5]
AAS_REVERSE_1B  = [5, 6, 17]
BPNS_REVERSE_1B = [2, 3, 6, 10, 14, 15, 18, 19, 20]
DERS_REV = [i - 1 for i in DERS_REVERSE_1B]
AAS_REV  = [i - 1 for i in AAS_REVERSE_1B]
BPNS_REV = [i - 1 for i in BPNS_REVERSE_1B]

# 工具函數
def _safe_list(values, n: int, fill):
    arr = list(values or [])
    out = []
    for x in arr[:n]:
        try:
            out.append(float(x))
        except Exception:
            out.append(fill)
    if len(out) < n:
        out += [fill] * (n - len(out))
    return out[:n]

def _mean(xs: List[float]) -> float:
    return sum(xs) / len(xs) if xs else 0.0

def _scale01(x: float, lo: float, hi: float) -> float:
    if hi <= lo:
        return 0.0
    v = (x - lo) / (hi - lo)
    return 0.0 if v < 0 else 1.0 if v > 1 else v

def _norm_list(vals: List[float], lo: float, hi: float, rev_idx: List[int]) -> List[float]:
    out = []
    for i, x in enumerate(vals):
        v = _scale01(x, lo, hi)
        if i in rev_idx:
            v = 1.0 - v
        out.append(v)
    return out

def _pow_contrast01(v: float, gamma: float) -> float:
    v = 0.0 if v < 0 else 1.0 if v > 1 else v
    try:
        return v ** float(gamma)
    except Exception:
        return v

# 特徵提取
def features_mbti(mbti_encoded: List[float] | None) -> Dict[str, float]:
    e, n, t, p = _safe_list(mbti_encoded, 4, 0.5)
    return {"E": e, "I": 1 - e, "N": n, "S": 1 - n, "T": t, "F": 1 - t, "P": p, "J": 1 - p}

def features_ders(values: List[float] | None) -> Dict[str, float]:
    raw = _safe_list(values, 18, 3.0)
    n5  = _norm_list(raw, 1.0, 5.0, DERS_REV)
    mu  = _mean(n5)
    var = _mean([(x - mu) ** 2 for x in n5])
    spread = var ** 0.5
    return {"level": mu, "spread": spread}

def features_aas(values: List[float] | None) -> Dict[str, float]:
    raw = _safe_list(values, 24, 3.0)
    n6  = _norm_list(raw, 1.0, 6.0, AAS_REV)
    avoid = _mean(n6[:8])
    mid   = _mean(n6[8:16])
    anx   = _mean(n6[16:24])
    insecure = (avoid + anx) / 2.0
    secure = max(0.0, 1.0 - insecure)
    return {"avoid": avoid, "anx": anx, "secure": secure, "mid": mid}

def features_bpns(values: List[float] | None) -> Dict[str, float]:
    raw = _safe_list(values, 21, 4.0)
    n7  = _norm_list(raw, 1.0, 7.0, BPNS_REV)
    A = _mean(n7[0:7])
    R = _mean(n7[7:14])
    C = _mean(n7[14:21])
    return {"autonomy": A, "relatedness": R, "competence": C}

# ✅ 主演算法（修正版）
def build_recommendation(assessment: Dict[str, Any], user: Dict[str, Any] | None = None) -> Dict[str, Any]:
    """
    推薦演算法 v2.4
    支援 snake_case 和 camelCase 欄位名稱
    """
    mbti = features_mbti(assessment.get("mbti_encoded"))
    
    # ✅ 修正：同時支援兩種命名方式
    aas  = features_aas(
        assessment.get("step2_answers") or assessment.get("step2Answers")
    )
    ders = features_ders(
        assessment.get("step3_answers") or assessment.get("step3Answers")
    )
    bpns = features_bpns(
        assessment.get("step4_answers") or assessment.get("step4Answers")
    )
    
    # 除錯輸出
    print(f"🔍 [Algorithm] MBTI={mbti}, AAS={aas}, DERS={ders}, BPNS={bpns}")

    # 計算四種機器人分數
    empathy = (
        0.35 * aas["anx"] +
        0.20 * (1.0 - bpns["relatedness"]) +
        0.15 * (1.0 - bpns["competence"]) +
        0.15 * mbti["F"] + 0.10 * mbti["I"] +
        0.05 * ders["level"]
    )
    insight = (
        0.30 * mbti["N"] + 0.20 * mbti["T"] +
        0.25 * bpns["autonomy"] + 0.15 * bpns["competence"] +
        0.10 * (1.0 - aas["avoid"])
    )
    solution = (
        0.30 * mbti["J"] + 0.20 * mbti["T"] +
        0.25 * bpns["competence"] + 0.15 * bpns["autonomy"] +
        0.10 * (1.0 - ders["level"])
    )
    cognitive = (
        0.40 * ders["level"] + 0.15 * ders["spread"] +
        0.20 * mbti["I"] + 0.15 * mbti["N"] +
        0.10 * (1.0 - bpns["autonomy"])
    )

    scores01 = {
        "empathy":   max(0.0, min(1.0, empathy)),
        "insight":   max(0.0, min(1.0, insight)),
        "solution":  max(0.0, min(1.0, solution)),
        "cognitive": max(0.0, min(1.0, cognitive)),
    }

    # 對比增強並轉換為 0~100
    ranked = []
    for k, v in scores01.items():
        v_contrast = _pow_contrast01(v, CONTRAST_GAMMA)
        s = round(v_contrast * 100.0, 1)
        s = min(s, 99.0)
        ranked.append({"type": k, "score": s})
    ranked.sort(key=lambda x: x["score"], reverse=True)
    top = ranked[0]
    
    print(f"✅ [Algorithm Result] Top={top['type']}({top['score']}%), Ranked={[(r['type'], r['score']) for r in ranked]}")

    return {
        "ok": True,
        "user": {"pid": (user or {}).get("pid")} if user else None,
        "scores": scores01,
        "ranked": ranked,
        "top": top,
        "algorithm_version": "emobot_v2.4_snake_case_support",
        "params": {
            "contrast_gamma": CONTRAST_GAMMA,
            "reverse": {
                "DERS": DERS_REVERSE_1B, 
                "AAS": AAS_REVERSE_1B, 
                "BPNS": BPNS_REVERSE_1B
            }
        }
    }

# 別名
get_recommendation = build_recommendation
make_recommendation = build_recommendation
recommend_endpoint_payload = build_recommendation
