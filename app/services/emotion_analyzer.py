# backend/app/services/emotion_analyzer.py - 門檻調整至30句
from typing import Dict, List, Any, Tuple
from datetime import datetime, timedelta
from collections import Counter, defaultdict
import re
import math

EMOTION_KEYWORDS = {
    "焦慮不安": {
        "keywords": ["擔心", "害怕", "緊張", "焦慮", "不安", "恐懼", "驚慌", "煩躁", "忐忑", "惶恐"],
        "weight": 1.2,
        "color": "#FF6B6B"
    },
    "憂鬱低落": {
        "keywords": ["難過", "憂鬱", "低落", "沮喪", "失落", "悲傷", "痛苦", "絕望", "孤單", "空虛"],
        "weight": 1.2,
        "color": "#4ECDC4"
    },
    "憤怒煩躁": {
        "keywords": ["生氣", "憤怒", "火大", "不爽", "煩", "討厭", "氣", "恨", "抓狂"],
        "weight": 1.1,
        "color": "#FF8C42"
    },
    "壓力疲憊": {
        "keywords": ["壓力", "累", "疲憊", "忙", "趕", "爆炸", "受不了", "撐不住", "崩潰"],
        "weight": 1.15,
        "color": "#95E1D3"
    },
    "快樂滿足": {
        "keywords": ["開心", "快樂", "高興", "喜歡", "棒", "好", "讚", "爽", "興奮", "滿足"],
        "weight": 0.9,
        "color": "#FEE440"
    },
    "平靜放鬆": {
        "keywords": ["平靜", "放鬆", "舒服", "安心", "穩定", "還好", "沒事", "淡定", "從容"],
        "weight": 0.8,
        "color": "#38B6FF"
    },
    "困惑迷茫": {
        "keywords": ["困惑", "迷茫", "不知道", "猶豫", "糾結", "矛盾", "疑惑", "迷失"],
        "weight": 1.0,
        "color": "#B983FF"
    }
}

TOPIC_KEYWORDS = {
    "人際關係": {
        "keywords": ["朋友", "家人", "同事", "關係", "相處", "吵架", "溝通", "孤單", "社交", "人際"],
        "weight": 1.0,
        "color": "#FF6B9D"
    },
    "工作職場": {
        "keywords": ["工作", "上班", "老闆", "同事", "加班", "專案", "業績", "報告", "職場", "工作"],
        "weight": 1.1,
        "color": "#5B8DEE"
    },
    "學業成長": {
        "keywords": ["考試", "成績", "學校", "老師", "作業", "報告", "讀書", "課業", "學習", "進修"],
        "weight": 1.0,
        "color": "#00D9FF"
    },
    "情感愛情": {
        "keywords": ["感情", "戀愛", "分手", "喜歡", "曖昧", "告白", "失戀", "交往", "伴侶", "喜歡"],
        "weight": 1.2,
        "color": "#FF85A2"
    },
    "自我認同": {
        "keywords": ["自己", "自我", "價值", "意義", "迷惘", "為什麼", "身分", "人生", "未來", "目標"],
        "weight": 1.15,
        "color": "#A8E6CF"
    },
    "身心健康": {
        "keywords": ["身體", "健康", "生病", "痛", "不舒服", "睡不著", "失眠", "頭痛", "疲勞", "焦慮"],
        "weight": 1.2,
        "color": "#FFB6B9"
    }
}

INTENSITY_MODIFIERS = {
    "極強": {
        "words": ["非常非常", "超級超級", "極度", "完全", "絕對"],
        "multiplier": 1.5
    },
    "強": {
        "words": ["非常", "超級", "真的很", "太", "特別", "相當"],
        "multiplier": 1.3
    },
    "中強": {
        "words": ["很", "蠻", "挺", "十分", "頗"],
        "multiplier": 1.1
    },
    "中": {
        "words": ["有點", "還算", "稍微", "些許", "算是"],
        "multiplier": 0.9
    },
    "弱": {
        "words": ["一點點", "不太", "微微", "似乎", "好像"],
        "multiplier": 0.7
    }
}

def time_decay_weight(days_ago: int) -> float:
    """計算時間衰減權重"""
    return math.exp(-0.05 * days_ago)

def calculate_emotion_intensity(text: str, base_score: float) -> float:
    """計算情緒強度"""
    intensity = base_score
    
    for level_data in INTENSITY_MODIFIERS.values():
        for word in level_data["words"]:
            if word in text:
                intensity *= level_data["multiplier"]
                break
    
    exclamation_count = text.count("！") + text.count("!")
    question_count = text.count("？？") + text.count("??")
    
    intensity *= (1 + exclamation_count * 0.1)
    intensity *= (1 + question_count * 0.08)
    
    if re.search(r'[A-Z]{3,}', text):
        intensity *= 1.15
    
    return min(intensity, 1.0)

def detect_emotions_advanced(text: str, days_ago: int = 0) -> Dict[str, float]:
    """進階情緒偵測"""
    emotions = {}
    
    for emotion, data in EMOTION_KEYWORDS.items():
        keywords = data["keywords"]
        weight = data["weight"]
        
        count = sum(text.count(word) for word in keywords)
        
        if count > 0:
            base_score = min(count * 0.15, 0.8)
            intensity = calculate_emotion_intensity(text, base_score)
            weighted_score = intensity * weight
            time_weighted = weighted_score * time_decay_weight(days_ago)
            emotions[emotion] = time_weighted
    
    return emotions

def detect_topics_advanced(text: str) -> Dict[str, float]:
    """進階議題偵測"""
    topics = {}
    
    for topic, data in TOPIC_KEYWORDS.items():
        keywords = data["keywords"]
        weight = data["weight"]
        
        count = sum(text.count(word) for word in keywords)
        
        if count > 0:
            score = min(count * 0.2, 0.9) * weight
            topics[topic] = score
    
    return topics

def analyze_emotion_trends(timeline_data: List[Dict]) -> Dict[str, Any]:
    """分析情緒趨勢變化"""
    if len(timeline_data) < 5:
        return {"trend": "insufficient_data"}
    
    recent_emotions = [d["dominant_emotion"] for d in timeline_data[-10:]]
    emotion_counter = Counter(recent_emotions)
    
    negative_emotions = ["焦慮不安", "憂鬱低落", "憤怒煩躁", "壓力疲憊"]
    negative_count = sum(1 for e in recent_emotions if e in negative_emotions)
    
    if negative_count > len(recent_emotions) * 0.7:
        trend = "concerning"
        trend_description = "近期負面情緒較多，建議尋求支持"
    elif negative_count > len(recent_emotions) * 0.4:
        trend = "fluctuating"
        trend_description = "情緒起伏較大，注意自我調適"
    else:
        trend = "stable"
        trend_description = "情緒狀態相對穩定"
    
    return {
        "trend": trend,
        "description": trend_description,
        "dominant_emotions": dict(emotion_counter.most_common(3))
    }

def generate_professional_summary(
    emotion_freq: Dict[str, int],
    emotion_intensity: Dict[str, float],
    topics: Dict[str, float],
    message_count: int,
    trend_info: Dict[str, Any]
) -> str:
    """生成專業的心理分析摘要"""
    
    if message_count < 30:
        return f"目前對話次數為 {message_count} 次，建議累積至少 30 次對話後再進行完整分析，以獲得更準確的心理狀態評估。"
    
    summary_parts = []
    
    if emotion_freq:
        top_emotion = max(emotion_freq.items(), key=lambda x: x[1])[0]
        freq_count = emotion_freq[top_emotion]
        
        summary_parts.append(f"📊 情緒特徵分析：")
        summary_parts.append(f"在 {message_count} 次對話中，你最常表達「{top_emotion}」的情緒（出現 {freq_count} 次）")
        
        if top_emotion == "焦慮不安":
            summary_parts.append("💡 建議：嘗試正念呼吸法或漸進式肌肉放鬆，有助於降低焦慮感")
        elif top_emotion == "憂鬱低落":
            summary_parts.append("💡 建議：持續的低落情緒需要關注，建議與專業心理諮商師討論")
        elif top_emotion == "壓力疲憊":
            summary_parts.append("💡 建議：適度休息和規律運動可以有效釋放壓力，也可嘗試時間管理技巧")
        elif top_emotion == "快樂滿足":
            summary_parts.append("💡 很棒！你的正向情緒表達頻率較高，繼續保持這樣的狀態")
    
    if topics:
        top_topic = max(topics.items(), key=lambda x: x[1])[0]
        summary_parts.append(f"\n🎯 核心議題：你最關注的主題是「{top_topic}」")
        
        if top_topic == "工作職場":
            summary_parts.append("建議建立工作與生活的界線，避免過度投入")
        elif top_topic == "人際關係":
            summary_parts.append("人際議題是常見的壓力源，學習適當的溝通技巧很重要")
        elif top_topic == "自我認同":
            summary_parts.append("自我探索是成長的重要過程，給自己時間慢慢釐清")
    
    if trend_info and trend_info.get("trend") != "insufficient_data":
        summary_parts.append(f"\n📈 情緒趨勢：{trend_info.get('description', '')}")
    
    summary_parts.append("\n✨ 持續與 AI 夥伴對話，有助於更深入地理解自己的情緒模式")
    
    return "\n".join(summary_parts)

def analyze_chat_messages(messages: List[Dict[str, Any]]) -> Dict[str, Any]:
    """專業版聊天訊息分析 - 門檻調整為30句"""
    
    user_messages = [m for m in messages if m.get("role") == "user"]
    
    if len(user_messages) < 30:
        return {
            "ok": False,
            "has_sufficient_data": False,
            "message_count": len(user_messages),
            "required_count": 30,
            "message": f"目前對話次數為 {len(user_messages)} 次，至少需要 30 次對話才能進行完整分析"
        }
    
    emotion_frequency = Counter()
    emotion_intensity_sum = defaultdict(float)
    emotion_count = defaultdict(int)
    topic_scores = Counter()
    timeline_data = []
    
    now = datetime.now()
    
    for msg in user_messages:
        text = msg.get("content", "")
        created_at_str = msg.get("created_at")
        
        days_ago = 0
        if created_at_str:
            try:
                msg_date = datetime.fromisoformat(created_at_str.replace('Z', '+00:00'))
                days_ago = (now - msg_date).days
            except:
                days_ago = 0
        
        emotions = detect_emotions_advanced(text, days_ago)
        
        if emotions:
            dominant_emotion = max(emotions.items(), key=lambda x: x[1])[0]
            dominant_score = emotions[dominant_emotion]
            
            emotion_frequency[dominant_emotion] += 1
            
            for emotion, score in emotions.items():
                emotion_intensity_sum[emotion] += score
                emotion_count[emotion] += 1
            
            timeline_data.append({
                "date": created_at_str,
                "dominant_emotion": dominant_emotion,
                "score": dominant_score,
                "all_emotions": emotions
            })
        
        topics = detect_topics_advanced(text)
        for topic, score in topics.items():
            topic_scores[topic] += score
    
    emotion_intensity_avg = {
        emotion: emotion_intensity_sum[emotion] / emotion_count[emotion]
        for emotion in emotion_count
    }
    
    trend_info = analyze_emotion_trends(timeline_data)
    
    summary = generate_professional_summary(
        dict(emotion_frequency),
        emotion_intensity_avg,
        dict(topic_scores),
        len(user_messages),
        trend_info
    )
    
    topic_radar_data = {}
    for topic, score in topic_scores.most_common(6):
        topic_radar_data[topic] = round(score * 20, 1)

    return {
        "ok": True,
        "has_sufficient_data": True,
        "message_count": len(user_messages),
        "emotion_frequency": dict(emotion_frequency.most_common(7)),
        "emotion_intensity": {k: round(v * 100, 1) for k, v in emotion_intensity_avg.items()},
        "topic_radar": topic_radar_data,
        "timeline_data": timeline_data[-30:],
        "trend_analysis": trend_info,
        "summary": summary,
        "analysis_meta": {
            "analyzed_at": datetime.now().isoformat(),
            "period_start": user_messages[0].get("created_at") if user_messages else None,
            "period_end": user_messages[-1].get("created_at") if user_messages else None,
            "total_days": days_ago if user_messages else 0
        },
        "emotion_colors": {k: v["color"] for k, v in EMOTION_KEYWORDS.items()},
        "topic_colors": {k: v["color"] for k, v in TOPIC_KEYWORDS.items()}
    }