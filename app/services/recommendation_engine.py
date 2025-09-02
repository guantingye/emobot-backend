# app/services/recommendation_engine.py
# -*- coding: utf-8 -*-
"""
改進的四型機器人推薦演算法（Empathy / Insight / Solution / Cognitive）
- 輸入：MBTI（二元4維：E,N,T,P），AAS(24)、DERS(18)、BPNS(21)
- 改進：更複雜的權重組合、非線性轉換、個人化調整
- 輸出：
  {
    "scores": { type: float in [0,1] },        # 提供給 Dashboard 雷達圖
    "ranked": [ { "type": str, "score": float } ],  # 0~100 相對分數（最高=100）給 MatchResult
    "top": { "type": str, "score": float }     # 最高分(0~100)
  }
"""
from __future__ import annotations
from math import exp, log, sqrt
from typing import Dict, List, Any
import random

# -----------------------
# 工具：安全取值 / 常用轉換
# -----------------------
def _safe_list(x, n=None, fill=0.0):
    if x is None:
        return [fill] * (n or 0)
    if not isinstance(x, list):
        return [fill] * (n or 0)
    return x if (n is None or len(x) == n) else (x + [fill] * max(0, n - len(x)))

def _clip01(v: float) -> float:
    return 0.0 if v < 0 else (1.0 if v > 1 else v)

def _norm(v, lo, hi) -> float:
    # 把 [lo,hi] 線性映射到 [0,1]
    if hi == lo:
        return 0.0
    return _clip01((float(v) - lo) / (hi - lo))

# 新增非線性轉換函數
def _sigmoid_boost(x: float, midpoint: float = 0.5, steepness: float = 6.0) -> float:
    """S型曲線增強，讓分數分佈更有層次"""
    try:
        exp_term = exp(-steepness * (x - midpoint))
        return 1.0 / (1.0 + exp_term)
    except (OverflowError, ZeroDivisionError):
        return 1.0 if x > midpoint else 0.0

def _power_transform(x: float, power: float = 1.2) -> float:
    """冪次轉換，放大差異"""
    return pow(_clip01(x), power)

def _weighted_combination(*args) -> float:
    """加權組合多個因子"""
    total_weight = 0.0
    total_value = 0.0
    
    for i in range(0, len(args), 2):
        if i + 1 < len(args):
            weight = float(args[i])
            value = float(args[i + 1])
            total_weight += weight
            total_value += weight * value
    
    return (total_value / total_weight) if total_weight > 0 else 0.0

# -----------------------
# 量表計分（子量表平均）
# -----------------------
def score_aas(values: List[float]) -> Dict[str, float]:
    """
    成人依附量表 (AAS) 24 題，1~7 刻度。
    改進版：加入反向題處理和更精確的分組
    """
    v = _safe_list(values, 24, 4.0)
    
    # 反向題處理 (基於更精確的AAS量表定義)
    reverse_items = {4, 15}  # 0-based: 第5題、第16題
    def n7(x, idx):
        raw_val = float(x)
        if idx in reverse_items:
            raw_val = 8 - raw_val
        return _norm(raw_val, 1, 7)

    # 更精確的因子分組
    sub = {
        "Secure":   [6, 12, 15, 17, 20, 22],    # 安全型
        "Anxious":  [2, 4, 7, 11, 18, 19],     # 焦慮型  
        "Avoidant": [0, 3, 5, 9, 13, 16],      # 逃避型
        "Fearful":  [1, 8, 10, 14, 21, 23],    # 害怕型
    }
    
    def mean_idx(idxs):
        if not idxs: return 0.0
        valid_scores = [n7(v[i], i) for i in idxs if i < len(v)]
        return sum(valid_scores) / len(valid_scores) if valid_scores else 0.0

    return {k: mean_idx(idxs) for k, idxs in sub.items()}


def score_ders(values: List[float]) -> Dict[str, float]:
    """
    DERS-18，1~5 刻度。改進版：更精確的反向題和構面定義
    """
    v = _safe_list(values, 18, 3.0)
    
    # 精確的反向題（Awareness構面）
    reverse_idx = {0, 3, 5}  # 第1, 4, 6題
    def n5(x, idx):
        raw_val = float(x)
        if idx in reverse_idx:
            raw_val = 6 - raw_val  
        return _norm(raw_val, 1, 5)

    # DERS-18 的六個構面
    sub = {
        "Awareness":     [0, 3, 5],
        "Clarity":       [1, 2, 4], 
        "Goals":         [6, 10, 13],
        "Impulse":       [7, 14, 16],
        "Nonacceptance": [8, 11, 12],
        "Strategies":    [9, 15, 17],
    }
    
    def mean_idx(idxs):
        if not idxs: return 0.0
        valid_scores = [n5(v[i], i) for i in idxs if i < len(v)]
        return sum(valid_scores) / len(valid_scores) if valid_scores else 0.0

    return {k: mean_idx(idxs) for k, idxs in sub.items()}


def score_bpns(values: List[float]) -> Dict[str, float]:
    """
    基本心理需求量表 BPNS 21 題，1~7 刻度。改進版：更精確的構面定義
    """
    v = _safe_list(values, 21, 4.0)
    
    # 精確的反向題定義
    reverse_idx = {3, 10, 19, 2, 14, 18, 6, 15, 17}
    def n7(x, idx):
        raw_val = float(x)
        if idx in reverse_idx:
            raw_val = 8 - raw_val
        return _norm(raw_val, 1, 7)

    # BPNS三大構面
    sub = {
        "Autonomy":   [0, 3, 7, 10, 13, 16, 19],
        "Competence": [2, 4, 9, 12, 14, 18],
        "Relatedness":[1, 5, 6, 8, 11, 15, 17, 20],
    }
    
    def mean_idx(idxs):
        if not idxs: return 0.0
        valid_scores = [n7(v[i], i) for i in idxs if i < len(v)]
        return sum(valid_scores) / len(valid_scores) if valid_scores else 0.0

    return {k: mean_idx(idxs) for k, idxs in sub.items()}

# -----------------------
# 改進的主演算法：多層次權重計算
# -----------------------
def compute_four_scores_enhanced(assessment: Dict[str, Any]) -> Dict[str, float]:
    """
    改進版分數計算：使用更複雜的權重和非線性轉換
    """
    # 取資料
    mbti = _safe_list(assessment.get("mbti_encoded"), 4, 0)
    E, N, T, P = [1 if int(x) == 1 else 0 for x in mbti]
    I, S, F, J = (1 - E), (1 - N), (1 - T), (1 - P)

    aas  = score_aas(assessment.get("step2Answers", []))
    ders = score_ders(assessment.get("step3Answers", []))
    bpns = score_bpns(assessment.get("step4Answers", []))

    # ---- 層次一：MBTI 基礎權重 ----
    mbti_base = {
        "empathy":   _weighted_combination(0.4, F, 0.2, E, 0.1, P, 0.05, N),
        "insight":   _weighted_combination(0.35, N, 0.2, I, 0.15, P, 0.05, F),
        "solution":  _weighted_combination(0.3, J, 0.2, T, 0.15, S, 0.1, E),
        "cognitive": _weighted_combination(0.3, T, 0.2, S, 0.15, I, 0.1, J),
    }

    # ---- 層次二：依附風格調節 ----
    # 不同依附風格對應不同AI需求強度
    attachment_weights = {
        "empathy": _weighted_combination(
            0.35, aas.get("Anxious", 0),     # 焦慮型最需要同理
            0.25, aas.get("Fearful", 0),     # 害怕型次之
            0.15, 1 - aas.get("Avoidant", 0),  # 逃避型較不需要
            0.1, aas.get("Secure", 0)        # 安全型有基本需求
        ),
        "insight": _weighted_combination(
            0.3, aas.get("Fearful", 0),      # 害怕型需要洞察
            0.2, 1 - aas.get("Secure", 0),   # 非安全型
            0.15, aas.get("Anxious", 0),     # 焦慮型也需要理解
            0.1, aas.get("Avoidant", 0)      # 逃避型有些需求
        ),
        "solution": _weighted_combination(
            0.4, aas.get("Secure", 0),       # 安全型最能接受解決導向
            0.2, 1 - aas.get("Anxious", 0),  # 非焦慮型
            0.15, 1 - aas.get("Fearful", 0), # 非害怕型
            0.1, aas.get("Avoidant", 0)      # 逃避型也可能需要
        ),
        "cognitive": _weighted_combination(
            0.35, aas.get("Avoidant", 0),    # 逃避型最偏好認知
            0.2, aas.get("Secure", 0),       # 安全型也可以
            0.15, 1 - aas.get("Anxious", 0), # 非焦慮型
            0.1, aas.get("Fearful", 0)       # 害怕型較少需求
        ),
    }

    # ---- 層次三：情緒調節困難調節 ----
    emotion_regulation = {
        "empathy": _weighted_combination(
            0.25, ders.get("Nonacceptance", 0),  # 情緒不接納
            0.2, ders.get("Strategies", 0),      # 策略缺乏  
            0.15, ders.get("Impulse", 0),        # 衝動控制困難
            0.1, ders.get("Clarity", 0)          # 情緒不清楚
        ),
        "insight": _weighted_combination(
            0.3, ders.get("Awareness", 0),       # 情緒覺察困難
            0.25, ders.get("Clarity", 0),        # 情緒清楚度困難
            0.15, ders.get("Goals", 0),          # 目標導向困難
            0.05, ders.get("Strategies", 0)
        ),
        "solution": _weighted_combination(
            0.35, ders.get("Goals", 0),          # 目標導向困難
            0.25, ders.get("Strategies", 0),     # 策略缺乏
            0.15, ders.get("Impulse", 0),        # 衝動控制
            0.1, ders.get("Nonacceptance", 0)    # 情緒接納困難
        ),
        "cognitive": _weighted_combination(
            0.2, 1 - ders.get("Awareness", 0),   # 覺察能力好（反向）
            0.2, 1 - ders.get("Impulse", 0),     # 衝動控制好
            0.15, 1 - ders.get("Nonacceptance", 0), # 情緒接納好
            0.1, ders.get("Clarity", 0)          # 需要釐清
        ),
    }

    # ---- 層次四：基本心理需求調節 ----
    psychological_needs = {
        "empathy": _weighted_combination(
            0.4, 1 - bpns.get("Relatedness", 0.5), # 關係需求未滿足
            0.15, 1 - bpns.get("Autonomy", 0.5),   # 自主需求未滿足
            0.1, 1 - bpns.get("Competence", 0.5)   # 勝任需求未滿足
        ),
        "insight": _weighted_combination(
            0.25, 1 - bpns.get("Autonomy", 0.5),   # 自主需求未滿足
            0.2, 1 - bpns.get("Relatedness", 0.5), # 關係需求未滿足
            0.1, bpns.get("Competence", 0.5)       # 有基本勝任感
        ),
        "solution": _weighted_combination(
            0.35, 1 - bpns.get("Competence", 0.5), # 勝任需求未滿足
            0.2, 1 - bpns.get("Autonomy", 0.5),    # 自主需求未滿足
            0.1, bpns.get("Relatedness", 0.5)      # 有基本關係感
        ),
        "cognitive": _weighted_combination(
            0.3, bpns.get("Competence", 0.5),      # 勝任感滿足
            0.2, bpns.get("Autonomy", 0.5),        # 自主感滿足
            0.1, bpns.get("Relatedness", 0.5)      # 關係感適中
        ),
    }

    # ---- 層次五：整合計算與非線性轉換 ----
    base_line = 0.1  # 基線分數
    raw_scores = {}
    
    for bot_type in ["empathy", "insight", "solution", "cognitive"]:
        # 多層次權重整合
        integrated_score = _weighted_combination(
            0.25, mbti_base[bot_type],
            0.30, attachment_weights[bot_type], 
            0.25, emotion_regulation[bot_type],
            0.20, psychological_needs[bot_type]
        )
        
        # 加上基線並應用非線性轉換
        enhanced_score = base_line + integrated_score * 0.9
        
        # 根據類型特性進行差異化調整
        if bot_type == "empathy":
            # 同理型：情緒敏感度加成
            enhanced_score = _sigmoid_boost(enhanced_score, 0.4, 5.0)
        elif bot_type == "insight":
            # 洞察型：複雜度適應加成
            enhanced_score = _power_transform(enhanced_score, 1.3)
        elif bot_type == "solution":
            # 解決型：目標導向加成
            enhanced_score = _sigmoid_boost(enhanced_score, 0.6, 4.0)
        elif bot_type == "cognitive":
            # 認知型：理性思考加成
            enhanced_score = _power_transform(enhanced_score, 1.1)
        
        raw_scores[bot_type] = _clip01(enhanced_score)
    
    # ---- 層次六：相對差異強化 ----
    # 確保分數有足夠差異性，避免過於接近
    scores_list = list(raw_scores.values())
    mean_score = sum(scores_list) / len(scores_list)
    std_dev = sqrt(sum((s - mean_score) ** 2 for s in scores_list) / len(scores_list))
    
    # 如果標準差太小，放大差異
    if std_dev < 0.15:
        for bot_type in raw_scores:
            deviation = raw_scores[bot_type] - mean_score
            enhanced_deviation = deviation * 1.8  # 放大差異
            raw_scores[bot_type] = _clip01(mean_score + enhanced_deviation)
    
    # ---- 層次七：隨機微調（避免完全重複） ----
    # 加入少量隨機性，讓同樣測試結果也有些微變化
    random.seed(hash(str(assessment.get("mbti_encoded", []))) % 10000)  # 使用MBTI做種子
    for bot_type in raw_scores:
        noise = (random.random() - 0.5) * 0.05  # ±2.5% 隨機調整
        raw_scores[bot_type] = _clip01(raw_scores[bot_type] + noise)
    
    return raw_scores


def _normalize_to_01_per_user(raw: Dict[str, float]) -> Dict[str, float]:
    """把本人的四型 raw 分數壓到 0~1（max=1, min≥0），給雷達圖用。"""
    vals = list(raw.values())
    mx, mn = max(vals), min(vals)
    # 避免全部相等
    if mx - mn < 1e-6:
        return {k: 0.5 for k in raw}
    return {k: (v - mn) / (mx - mn) for k, v in raw.items()}

def _to_percentage_enhanced(norm01: Dict[str, float]) -> List[Dict[str, Any]]:
    """
    改進版百分比轉換：確保分數分佈更合理
    最高分 = 100，其他按比例分配，但保證有足夠區別
    """
    if not norm01:
        return []
    
    # 找到最高分
    max_val = max(norm01.values())
    if max_val <= 0:
        max_val = 1.0
    
    # 計算相對百分比，但加入非線性調整
    results = []
    for bot_type, score in norm01.items():
        # 基本百分比
        percentage = (score / max_val) * 100.0
        
        # 非線性調整：讓分數差距更明顯
        if percentage >= 85:
            adjusted = 90 + (percentage - 85) * 0.67  # 90-100區間
        elif percentage >= 70:
            adjusted = 75 + (percentage - 70) * 1.0   # 75-90區間  
        elif percentage >= 50:
            adjusted = 55 + (percentage - 50) * 1.0   # 55-75區間
        elif percentage >= 30:
            adjusted = 35 + (percentage - 30) * 1.0   # 35-55區間
        else:
            adjusted = 15 + percentage * 0.67         # 15-35區間
        
        results.append({
            "type": bot_type,
            "score": round(min(100.0, max(15.0, adjusted)), 1)
        })
    
    # 按分數排序
    results.sort(key=lambda x: x["score"], reverse=True)
    
    # 確保最高分是100
    if results:
        results[0]["score"] = 100.0
        
    return results

def build_recommendation(assessment: Dict[str, Any], user: Dict[str, Any] = None) -> Dict[str, Any]:
    """
    改進版推薦主函式：產生更有區別性的分數
    """
    try:
        # 使用改進版演算法
        raw = compute_four_scores_enhanced(assessment)
        
        # 生成雷達圖用的 0~1 分數  
        norm01 = _normalize_to_01_per_user(raw)
        
        # 生成排序用的百分比分數
        ranked = _to_percentage_enhanced(norm01)
        
        # 找出推薦的最高分
        top = ranked[0] if ranked else {"type": "empathy", "score": 100.0}
        
        return {
            "ok": True,
            "user": {"pid": (user or {}).get("pid")} if user else None,
            "scores": norm01,     # 0~1，給雷達圖
            "ranked": ranked,     # 0~100，給選擇頁面
            "top": top,           # 最高推薦
            "algorithm_version": "enhanced_v2.2",
            "features": {
                "raw_scores": raw,
                "enhancement": "multi_layer_weighted_nonlinear",
                "differentiation": "enhanced"
            }
        }
        
    except Exception as e:
        print(f"Enhanced recommendation failed: {e}")
        # 提供預設回應
        fallback_scores = {"empathy": 0.65, "insight": 0.58, "solution": 0.72, "cognitive": 0.45}
        fallback_ranked = [
            {"type": "solution", "score": 100.0},
            {"type": "empathy", "score": 90.3},
            {"type": "insight", "score": 80.6},
            {"type": "cognitive", "score": 62.5}
        ]
        return {
            "ok": True,
            "user": {"pid": (user or {}).get("pid")} if user else None,
            "scores": fallback_scores,
            "ranked": fallback_ranked,
            "top": {"type": "solution", "score": 100.0},
            "algorithm_version": "fallback_v1.0",
            "error": str(e)
        }

# 兼容性別名
get_recommendation = build_recommendation
make_recommendation = build_recommendation
recommend_endpoint_payload = build_recommendation