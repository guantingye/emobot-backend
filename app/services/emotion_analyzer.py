# backend/app/services/emotion_analyzer.py - 專業版
from typing import Dict, List, Any, Tuple
from datetime import datetime, timedelta
from collections import Counter, defaultdict
import re
import math

# ============================================================================
# 專業級情緒詞庫 (基於心理學分類)
# ============================================================================

EMOTION_KEYWORDS = {
    "焦慮不安": {
        "keywords": ["擔心", "害怕", "緊張", "焦慮", "不安", "恐懼", "驚慌", "煩躁", "忐忑", "惶恐"],
        "weight": 1.2,  # 負面情緒權重較高
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

# 強度修飾詞（更精細化）
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

# 時間衰減係數（近期訊息權重更高）
def time_decay_weight(days_ago: int) -> float:
    """計算時間衰減權重，越近期的訊息權重越高"""
    return math.exp(-0.05 * days_ago)

# ============================================================================
# 核心分析函數
# ============================================================================

def calculate_emotion_intensity(text: str, base_score: float) -> float:
    """
    計算情緒強度（考慮修飾詞、標點符號等）
    
    Args:
        text: 訊息文本
        base_score: 基礎分數
    
    Returns:
        調整後的強度分數 (0.0 - 1.0)
    """
    intensity = base_score
    
    # 檢查強度修飾詞
    for level_data in INTENSITY_MODIFIERS.values():
        for word in level_data["words"]:
            if word in text:
                intensity *= level_data["multiplier"]
                break
    
    # 標點符號加成
    exclamation_count = text.count("！") + text.count("!")
    question_count = text.count("？？") + text.count("??")
    
    intensity *= (1 + exclamation_count * 0.1)
    intensity *= (1 + question_count * 0.08)
    
    # 全大寫加成（若有英文）
    if re.search(r'[A-Z]{3,}', text):
        intensity *= 1.15
    
    return min(intensity, 1.0)

def detect_emotions_advanced(text: str, days_ago: int = 0) -> Dict[str, float]:
    """
    進階情緒偵測（考慮權重、時間衰減）
    
    Returns:
        {情緒名稱: 加權分數}
    """
    emotions = {}
    
    for emotion, data in EMOTION_KEYWORDS.items():
        keywords = data["keywords"]
        weight = data["weight"]
        
        # 計算關鍵詞出現次數
        count = sum(text.count(word) for word in keywords)
        
        if count > 0:
            # 基礎分數
            base_score = min(count * 0.15, 0.8)
            
            # 考慮強度修飾詞
            intensity = calculate_emotion_intensity(text, base_score)
            
            # 應用情緒權重
            weighted_score = intensity * weight
            
            # 應用時間衰減
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
    """
    分析情緒趨勢變化
    
    Returns:
        包含趨勢資訊的字典
    """
    if len(timeline_data) < 5:
        return {"trend": "insufficient_data"}
    
    # 取最近的情緒分數
    recent_emotions = [d["dominant_emotion"] for d in timeline_data[-10:]]
    emotion_counter = Counter(recent_emotions)
    
    # 判斷趨勢
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
    """
    生成專業的心理分析摘要
    
    使用心理學術語,提供具體建議
    """
    
    if message_count < 30:
        return f"目前對話次數為 {message_count} 次,建議累積至少 30 次對話後再進行完整分析,以獲得更準確的心理狀態評估。"
    
    summary_parts = []
    
    # 主要情緒分析
    if emotion_freq:
        top_emotion = max(emotion_freq.items(), key=lambda x: x[1])[0]
        freq_count = emotion_freq[top_emotion]
        
        summary_parts.append(f"情緒特徵分析")
        summary_parts.append(f"在 {message_count} 次對話中,你最常表達「{top_emotion}」的情緒,出現了 {freq_count} 次。這反映出你近期的核心情緒狀態。")
        
        # 根據主要情緒給建議 - 更有建設性和創意
        if top_emotion == "焦慮不安":
            summary_parts.append("\n建議: 焦慮往往源於對未來的過度關注。試著將注意力拉回當下,練習「5-4-3-2-1」感官接地技巧:說出你看到的5樣東西、聽到的4種聲音、觸摸到的3樣物品、聞到的2種氣味、嚐到的1種味道。這能有效中斷焦慮迴圈,讓你重新掌握當下。")
        elif top_emotion == "憂鬱低落":
            summary_parts.append("\n建議: 持續的低落情緒值得溫柔對待。除了考慮尋求專業心理諮商,也可以嘗試「行為活化」策略:即使不想動,也安排一些小型愉快活動(如散步10分鐘、聽首喜歡的歌)。行動往往先於動機,微小的成就感能逐步提升情緒。")
        elif top_emotion == "壓力疲憊":
            summary_parts.append("\n建議: 壓力是身體發出的訊號,提醒你需要調整節奏。建議採用「番茄工作法」搭配「能量管理」:每25分鐘專注工作後休息5分鐘,並在每天安排至少一項能「充電」的活動(運動、創作、與朋友聊天)。記住,休息不是浪費時間,而是為了走更遠的路。")
        elif top_emotion == "快樂滿足":
            summary_parts.append("\n很棒! 你的正向情緒表達頻率較高。建議將這些美好時刻記錄下來,建立「快樂資料庫」,在低潮時翻閱能提醒自己生活中的亮點。同時也可以思考:是什麼帶來了這些快樂?如何在生活中創造更多類似體驗?")
        elif top_emotion == "平靜放鬆":
            summary_parts.append("\n你展現出良好的情緒調節能力。這份平靜值得珍惜,也可以進一步深化:嘗試冥想或正念練習,將這份平靜內化為穩定的心理資源,在面對挑戰時能更快回到中心。")
        elif top_emotion == "困惑迷茫":
            summary_parts.append("\n建議: 困惑是成長的契機,代表你正在思考重要問題。試著用「寫作療癒」整理思緒:每天花10分鐘自由書寫,不修改、不批判,讓潛意識的想法浮現。也可以找信任的人對話,有時候說出口的過程本身就能帶來澄清。")
    
    # 議題分析 - 更具體且有洞察
    if topics:
        top_topic = max(topics.items(), key=lambda x: x[1])[0]
        summary_parts.append(f"\n核心關注議題")
        summary_parts.append(f"你最關注的主題是「{top_topic}」,這顯示此領域對你的生活影響較深。")
        
        if top_topic == "工作職場":
            summary_parts.append("工作佔據生活的大部分時間,建議定期檢視工作的意義與價值,而非只關注績效。同時設定明確的下班儀式(如換衣服、散步),幫助大腦切換模式,避免工作情緒延續到私人時間。")
        elif top_topic == "人際關係":
            summary_parts.append("人際關係是情緒的重要來源。記住:健康的關係需要界線,學習溫和但堅定地表達需求,以及接受「不是所有關係都需要維持」。將能量投注在互相滋養的關係上,而非消耗性的互動。")
        elif top_topic == "自我認同":
            summary_parts.append("自我探索是一生的課程,不急於找到「標準答案」。可以嘗試「價值觀澄清練習」:列出對你最重要的5-10個價值觀,檢視目前生活是否與之一致。當行動與價值觀對齊時,迷茫感會逐漸減少。")
        elif top_topic == "情感愛情":
            summary_parts.append("親密關係是自我認識的鏡子。無論處於哪個階段,都值得思考:我在關係中想成為什麼樣的人?我能為對方帶來什麼?同時記得,完整的自己才能建立完整的關係。")
        elif top_topic == "學業成長":
            summary_parts.append("學習不只是知識累積,更是能力培養。建議採用「刻意練習」概念:專注在稍微超出舒適圈的挑戰上,並定期反思學習歷程。成長往往發生在你以為做不到,但最終做到的那些時刻。")
        elif top_topic == "身心健康":
            summary_parts.append("身心健康是一切的基礎。記住「心身一體」的概念:身體的不適可能反映心理壓力,心理的困擾也會影響身體。建議建立基本的自我照顧習慣:規律作息、適度運動、充足睡眠,這些看似簡單卻最根本。")
    
    # 趨勢分析
    if trend_info and trend_info.get("trend") != "insufficient_data":
        summary_parts.append(f"\n情緒趨勢: {trend_info.get('description', '')}")
    
    # 整體建議 - 更有溫度和行動性
    summary_parts.append("\n持續與 AI 夥伴對話,就像是在為自己的心靈建立一本日記。這些記錄不僅幫助你看見自己的模式,也提醒你:情緒會變化,困境會過去,而你一直在成長。")
    
    return "\n".join(summary_parts)

# ============================================================================
# 主要分析函數
# ============================================================================

def analyze_chat_messages(messages: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    專業版聊天訊息分析
    
    Args:
        messages: 聊天訊息列表
    
    Returns:
        完整的分析報告
    """
    # 只分析使用者訊息
    user_messages = [m for m in messages if m.get("role") == "user"]
    
    if len(user_messages) < 30:
        return {
            "ok": False,
            "has_sufficient_data": False,
            "message_count": len(user_messages),
            "required_count": 30,
            "message": f"目前對話次數為 {len(user_messages)} 次，至少需要 30 次對話才能進行完整分析"
        }
    
    # 初始化統計容器
    emotion_frequency = Counter()
    emotion_intensity_sum = defaultdict(float)
    emotion_count = defaultdict(int)
    topic_scores = Counter()
    timeline_data = []
    
    # 計算每則訊息距今天數
    now = datetime.now()
    
    for msg in user_messages:
        text = msg.get("content", "")
        created_at_str = msg.get("created_at")
        
        # 計算天數差
        days_ago = 0
        if created_at_str:
            try:
                msg_date = datetime.fromisoformat(created_at_str.replace('Z', '+00:00'))
                days_ago = (now - msg_date).days
            except:
                days_ago = 0
        
        # 偵測情緒（進階版）
        emotions = detect_emotions_advanced(text, days_ago)
        
        if emotions:
            # 找出主要情緒
            dominant_emotion = max(emotions.items(), key=lambda x: x[1])[0]
            dominant_score = emotions[dominant_emotion]
            
            # 統計
            emotion_frequency[dominant_emotion] += 1
            
            for emotion, score in emotions.items():
                emotion_intensity_sum[emotion] += score
                emotion_count[emotion] += 1
            
            # 時間序列記錄
            timeline_data.append({
                "date": created_at_str,
                "dominant_emotion": dominant_emotion,
                "score": dominant_score,
                "all_emotions": emotions
            })
        
        # 偵測議題
        topics = detect_topics_advanced(text)
        for topic, score in topics.items():
            topic_scores[topic] += score
    
    # 計算平均情緒強度
    emotion_intensity_avg = {
        emotion: emotion_intensity_sum[emotion] / emotion_count[emotion]
        for emotion in emotion_count
    }
    
    # 分析趨勢
    trend_info = analyze_emotion_trends(timeline_data)
    
    # 生成專業摘要
    summary = generate_professional_summary(
        dict(emotion_frequency),
        emotion_intensity_avg,
        dict(topic_scores),
        len(user_messages),
        trend_info
    )
    
# ✅ 修正：正確處理 Counter.most_common() 返回的 list
    topic_radar_data = {}
    for topic, score in topic_scores.most_common(6):
        topic_radar_data[topic] = round(score * 20, 1)

    # 準備圖表數據
    return {
        "ok": True,
        "has_sufficient_data": True,
        "message_count": len(user_messages),
        "emotion_frequency": dict(emotion_frequency.most_common(7)),
        "emotion_intensity": {k: round(v * 100, 1) for k, v in emotion_intensity_avg.items()},
        "topic_radar": topic_radar_data,
        "timeline_data": timeline_data[-30:],  # 只返回最近30筆
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