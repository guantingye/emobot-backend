# recommend.py - 簡化版本
def compute_recommendation(input_data: dict) -> dict:
    """
    簡化版推薦演算法 - 先讓 API 運行起來
    """
    
    # 基於 MBTI 的簡單推薦邏輯
    mbti = input_data.get("mbti", [1, 1, 1, 0])
    
    # 簡單的規則基礎推薦
    if mbti[0] == 1:  # E (外向)
        if mbti[1] == 1:  # N (直覺)
            recommended = "Solution-FocusedAI"
            recommended_id = 3
        else:  # S (感覺)
            recommended = "EmpathicAI" 
            recommended_id = 1
    else:  # I (內向)
        if mbti[2] == 1:  # T (思考)
            recommended = "CognitiveAI"
            recommended_id = 4
        else:  # F (情感)
            recommended = "InsightfulAI"
            recommended_id = 2
    
    # 模擬相似度分數
    scores = {
        "EmpathicAI": 0.65,
        "InsightfulAI": 0.72,
        "Solution-FocusedAI": 0.85,
        "CognitiveAI": 0.68
    }
    
    # 讓推薦的 AI 有最高分數
    scores[recommended] = 0.85
    
    return {
        "best": recommended,
        "confidence": scores[recommended],
        "all_scores": [
            {"label": name, "sim": round(score, 3)} 
            for name, score in scores.items()
        ],
        "ders_scores": {
            "Awareness": 3.5,
            "Clarity": 3.2,
            "Goals": 3.8,
            "Impulse": 3.1,
            "Nonacceptance": 3.4,
            "Strategies": 3.6
        },
        "aas_scores": {
            "Secure": 4.2,
            "Anxious": 2.8,
            "Avoidant": 3.1,
            "Fearful": 2.5
        },
        "bpns_scores": {
            "Autonomy": 4.1,
            "Competence": 4.3,
            "Relatedness": 3.9
        },
        "tsne": {
            "points": [
                {"label": "EmpathicAI", "x": -1.2, "y": 0.8},
                {"label": "InsightfulAI", "x": 0.5, "y": -1.1},
                {"label": "Solution-FocusedAI", "x": 1.3, "y": 0.3},
                {"label": "CognitiveAI", "x": -0.6, "y": -0.9},
                {"label": "User", "x": 0.1, "y": 0.2}
            ],
            "link": {"from": "User", "to": recommended}
        }
    }