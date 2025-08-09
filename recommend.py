import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.manifold import TSNE


def compute_recommendation(input_data: dict) -> dict:
    # --- 處理 MBTI ---
    mbti = np.array(input_data["mbti"]).reshape(1, -1)

    # --- 處理 DERS-18 ---
    ders_raw = np.array(input_data["ders"], dtype=float)
    ders_reverse = [1, 3, 5]
    ders_raw[ders_reverse] = 6 - ders_raw[ders_reverse]
    DERS_subscales = {
        "Awareness": [0, 3, 5],
        "Clarity": [1, 2, 4],
        "Goals": [7, 11, 14],
        "Impulse": [8, 15, 17],
        "Nonacceptance": [6, 12, 13],
        "Strategies": [9, 10, 16]        
    }
    ders_scores = {k: float(np.mean(ders_raw[idxs])) for k, idxs in DERS_subscales.items()}

    # --- 處理 AAS-24 ---
    aas_raw = np.array(input_data["aas"], dtype=float)
    aas_reverse = [5, 6, 17]
    aas_raw[aas_reverse] = 7 - aas_raw[aas_reverse]
    AAS_subscales = {
        "Secure": [7, 13, 16, 18, 21, 23],
        "Anxious": [3, 5, 8, 12, 19, 20],
        "Avoidant": [1, 4, 6, 10, 14, 17],
        "Fearful": [0, 2, 9, 11, 15, 22]
    }
    aas_scores = {k: float(np.mean(aas_raw[idxs])) for k, idxs in AAS_subscales.items()}

    # --- 處理 BPNS-21 ---
    bpns_raw = np.array(input_data["bpns"], dtype=float)
    bpns_reverse = [2, 3, 6, 10, 14, 15, 18, 19, 20]
    bpns_raw[bpns_reverse] = 8 - bpns_raw[bpns_reverse]
    BPNS_subscales = {
        "Autonomy": [0, 3, 7, 10, 13, 16, 19],
        "Competence": [2, 4, 9, 12, 14, 18],
        "Relatedness": [1, 5, 6, 8, 11, 15, 17, 20]
    }
    bpns_scores = {k: float(np.mean(bpns_raw[idxs])) for k, idxs in BPNS_subscales.items()}

    # --- 組合使用者向量 ---
    feature_vector = np.hstack([
        mbti.ravel(),
        list(ders_scores.values()),
        list(aas_scores.values()),
        list(bpns_scores.values())
    ]).reshape(1, -1)

    # --- 定義四型 AI Profiles ---
    np.random.seed(42)
    n_features = feature_vector.shape[1]
    ai_profiles = {
        "EmpathicAI": np.linspace(3, 5, n_features) + np.random.normal(0, 0.2, n_features),
        "InsightfulAI": np.linspace(2, 4.5, n_features) + np.random.normal(0, 0.2, n_features),
        "CognitiveAI": np.linspace(2, 5, n_features) + np.random.normal(0, 0.2, n_features),
        "Solution-FocusedAI": np.linspace(3, 4, n_features) + np.random.normal(0, 0.2, n_features)
    }
    ai_df = pd.DataFrame(ai_profiles).T

    # --- 標準化與相似度計算 ---
    scaler = StandardScaler().fit(ai_df)
    ai_scaled = scaler.transform(ai_df)
    user_scaled = scaler.transform(feature_vector)
    sims = {
        label: float(cosine_similarity(user_scaled, ai_scaled[i].reshape(1, -1))[0, 0])
        for i, label in enumerate(ai_df.index)
    }

    # --- 排序推薦結果 ---
    ranked = sorted(sims.items(), key=lambda x: x[1], reverse=True)
    best, confidence = ranked[0][0], round(ranked[0][1], 2)

    # --- t-SNE 座標 ---
    combined = np.vstack([ai_scaled, user_scaled])
    n_samples = combined.shape[0]
    # perplexity 必須小於樣本數，預設使用 min(6, n_samples-1)
    perp = min(6, max(1, n_samples - 1))
    tsne = TSNE(n_components=2, perplexity=perp, random_state=42)
    coords = tsne.fit_transform(combined)
    points = [
        {"label": lbl, "x": float(coords[i, 0]), "y": float(coords[i, 1])}
        for i, lbl in enumerate(list(ai_df.index) + ["User"])
    ]
    link = {"from": "User", "to": best}

    # --- 回傳結果 ---
    return {
        "best": best,
        "confidence": confidence,
        "all_scores": [{"label": lbl, "sim": round(sim, 3)} for lbl, sim in ranked],
        "ders_scores": ders_scores,
        "aas_scores": aas_scores,
        "bpns_scores": bpns_scores,
        "tsne": {"points": points, "link": link}
    }
