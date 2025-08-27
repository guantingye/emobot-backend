from typing import Dict, Tuple

# 極簡版範例：依 MBTI 簡單映射到四型機器人
# 可日後替換為你原本的模型/權重（保留介面不變）
BOT_TYPES = ["empathy", "insight", "solution", "cognitive"]

MBTI_WEIGHT = {
    "empathy":  {"F": 1.0, "E": 0.5},
    "insight":  {"N": 1.0, "I": 0.3},
    "solution": {"T": 1.0, "J": 0.5},
    "cognitive":{"P": 1.0, "S": 0.3},
}

def score_bots(mbti_raw: str | None) -> Dict[str, float]:
    mbti = (mbti_raw or "").upper().strip()
    scores = {b: 0.0 for b in BOT_TYPES}
    if len(mbti) >= 4:
        for b in BOT_TYPES:
            for k, w in MBTI_WEIGHT[b].items():
                if k in mbti:
                    scores[b] += w
    return scores

def rank(scores: Dict[str, float]) -> Tuple[str, Dict[str, float]]:
    top = max(scores.keys(), key=lambda k: scores[k]) if scores else "empathy"
    return top, scores
