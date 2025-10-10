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
    生成專業的心理分析摘要 - 深度版
    
    結合心理學理論與人性化關懷
    """
    
    if message_count < 30:
        return f"目前對話次數為 {message_count} 次,建議累積至少 30 次對話後再進行完整分析,以獲得更準確的心理狀態評估。"
    
    summary_parts = []
    
    # 開場白 - 溫暖的問候
    summary_parts.append("📖 **你的心情故事**")
    summary_parts.append(f"在過去的 **{message_count} 次對話**中,你與我分享了內心的起伏與思考。每一次的表達都是勇敢的自我揭露,讓我們一起來看看這段時間你的心理軌跡。")
    
    # 主要情緒分析 - 深度解讀
    if emotion_freq:
        top_emotion = max(emotion_freq.items(), key=lambda x: x[1])[0]
        freq_count = emotion_freq[top_emotion]
        total_count = sum(emotion_freq.values())
        emotion_percentage = round((freq_count / total_count) * 100, 1)
        
        summary_parts.append(f"\n📊 **情緒核心樣貌**")
        summary_parts.append(f"你最常表達的是「**{top_emotion}**」,在對話中出現了 **{freq_count} 次**,佔所有情緒表達的 **{emotion_percentage}%**。這不只是數字,它反映了你最近生活中的**主要情緒基調**。")
        
        # 情緒強度的深度分析
        if emotion_intensity and top_emotion in emotion_intensity:
            intensity_value = emotion_intensity[top_emotion]
            
            if intensity_value > 75:
                summary_parts.append(f"\n當你感受到這種情緒時,**強度指數達到 {round(intensity_value, 1)}/100**。這是相當高的強度,意味著這些情緒深深地影響著你。你可能會感覺被情緒淹沒,難以抽離。這樣的感受是**真實且重要的**,不需要否認或壓抑。")
            elif intensity_value > 50:
                summary_parts.append(f"\n當你感受到這種情緒時,**強度指數達到 {round(intensity_value, 1)}/100**。這是明顯且清晰的情緒體驗。你能清楚感受到它的存在,但還不至於失控。這個強度顯示你**既能感受情緒,也還保有一定的觀察距離**。")
            elif intensity_value > 30:
                summary_parts.append(f"\n當你感受到這種情緒時,**強度指數達到 {round(intensity_value, 1)}/100**。這是溫和而持續的情緒狀態。它像背景音樂一樣存在,不會太過強烈,但確實影響著你的**日常感受與決策**。")
            else:
                summary_parts.append(f"\n當你感受到這種情緒時,**強度指數達到 {round(intensity_value, 1)}/100**。這是較為柔和的情緒體驗。你展現出**良好的情緒調節能力**,能在感受情緒的同時保持內在平衡。")
        
        # 根據主要情緒給予深度且溫暖的建議
        if top_emotion == "焦慮不安":
            summary_parts.append("\n💡 **寫給焦慮的你**")
            summary_parts.append("焦慮就像心裡住著一位**過度盡責的守衛**,總想提前預防所有可能的危險。它的出發點是保護你,但有時會過度警戒。")
            summary_parts.append("\n從心理學角度來看,焦慮往往源於:**對未來的不確定感、對失控的恐懼、以及完美主義的壓力**。讓我們一起理解它,而非對抗它。")
            summary_parts.append("\n**立即可用的技巧:**")
            summary_parts.append("• **5-4-3-2-1 接地法**: 當焦慮來襲,說出你看見的5樣東西、聽到的4種聲音、觸摸的3樣物品、聞到的2種氣味、嚐到的1種味道。這能將你的**意識從未來拉回當下**。")
            summary_parts.append("• **焦慮對話練習**: 把焦慮想像成一個人,問它:「你在擔心什麼?」「你想保護我免於什麼?」「這個擔心有多少是真實的?」透過**對話而非對抗**,你會發現焦慮背後的真實需求。")
            summary_parts.append("• **控制圈與影響圈**: 將擔心的事情分類:哪些是你**能控制的**(如準備程度、應對方式)?哪些是**完全無法控制的**(如他人想法、結果)?把能量放在控制圈內,接受影響圈的不確定。")
            summary_parts.append("\n**長期建議**: 考慮學習**認知行為治療(CBT)**技巧,它能幫助你辨識與改變焦慮的思考模式。如果焦慮持續影響睡眠、工作或人際關係,尋求專業心理師的協助是**智慧而非軟弱**的選擇。")
            
        elif top_emotion == "憂鬱低落":
            summary_parts.append("\n💡 **寫給低落的你**")
            summary_parts.append("憂鬱像是**心靈的重感冒**,它會讓一切都變得灰暗、沉重,連平常喜歡的事都失去了色彩。如果你正經歷這樣的感受,我想告訴你:**這不是你的錯,你沒有「不夠努力」或「不夠堅強」**。")
            summary_parts.append("\n憂鬱情緒可能來自:生活壓力的累積、失落與失望的疊加、生理因素(如荷爾蒙變化、睡眠不足)、或是**長期忽略自己需求**的結果。")
            summary_parts.append("\n**溫柔的應對方式:**")
            summary_parts.append("• **微行動計畫**: 不要期待自己立刻「振作」。從**最小的行動**開始:今天只要起床洗臉、出門買杯飲料、或給朋友傳個訊息。**行動會創造動力**,而非等動力出現才行動。")
            summary_parts.append("• **情緒命名練習**: 每天花2分鐘,用**具體的詞彙描述你的感受**(不只是「難過」,而是「空虛的」、「疲憊的」、「失望的」)。命名情緒能降低它 30% 的強度,這是神經科學證實的現象。")
            summary_parts.append("• **社交連結,即使很小**: 憂鬱會讓你想縮回殼裡,但**隔離會加深憂鬱**。不需要社交活動,只要「**在人群中存在**」就好(去咖啡廳讀書、跟家人吃頓飯、傳訊息給朋友)。")
            summary_parts.append("• **自我慈悲,而非自我批判**: 當負面想法出現(「我好廢」、「我什麼都做不好」),試著問自己:**如果好友這樣說自己,我會怎麼安慰他?** 然後,用同樣的溫柔對待自己。")
            summary_parts.append("\n**重要提醒**: 如果低落情緒**持續超過兩週、影響日常功能、或出現自傷念頭**,請務必尋求專業協助。憂鬱症是可以治療的,你**值得獲得專業的支持**。")
            
        elif top_emotion == "憤怒煩躁":
            summary_parts.append("\n💡 **寫給憤怒的你**")
            summary_parts.append("憤怒是一種**力量強大的情緒信號**,它在告訴你:「有什麼不對勁了」、「我的界線被侵犯了」、「我的需求沒有被看見」。憤怒本身不是問題,**如何理解與表達它才是關鍵**。")
            summary_parts.append("\n從心理學來看,憤怒的底層常常是:**無力感、受傷、恐懼、或長期累積的委屈**。它像是冰山露出水面的一角,底下藏著更深層的情緒。")
            summary_parts.append("\n**理解你的憤怒:**")
            summary_parts.append("• **憤怒日誌**: 記錄每次生氣的情境:誰?什麼事?你的**身體反應**(心跳加速、肌肉緊繃)?你的**想法**(他憑什麼、太不公平)?你的**行動**(大吼、摔門、冷戰)?")
            summary_parts.append("  一週後回顧,你會看見**觸發模式**:是特定的人事物?還是當你累了、餓了、壓力大時更容易生氣?")
            summary_parts.append("• **冰山練習**: 下次生氣時,暫停並問自己:「**憤怒底下,我真正的感受是什麼?**」可能是受傷(覺得不被尊重)、恐懼(擔心失去控制)、悲傷(期待落空)。找到真正的情緒,才能有效溝通。")
            summary_parts.append("\n**健康的表達方式:**")
            summary_parts.append("• **暫停技巧**: 當怒氣衝上來,**給自己10分鐘**:深呼吸、離開現場、喝杯水。情緒高峰通常在 **90 秒內**會自然下降,不要在高峰時做決定或說話。")
            summary_parts.append("• **「我訊息」溝通**: 不說「你總是...」(指責),改說「當...發生時,我感覺...,因為我需要...」。例如:「當你沒有回我訊息時,我感覺被忽視,因為我需要知道你是否收到重要訊息。」")
            summary_parts.append("• **運動釋放**: 憤怒是**高能量情緒**,需要出口。跑步、拳擊、大掃除都能幫助釋放,讓身體代謝掉壓力荷爾蒙。")
            
        elif top_emotion == "壓力疲憊":
            summary_parts.append("\n💡 **寫給疲憊的你**")
            summary_parts.append("疲憊是身體與心靈發出的**紅色警報**:「我需要休息了」、「負荷超載了」。現代社會常把忙碌當美德,但**持續透支會導致耗竭(Burnout)**,影響健康、工作和人際關係。")
            summary_parts.append("\n心理學研究顯示,壓力疲憊不只是「做太多」,更常是:**做了不重要的事太多、無法掌控的事太多、以及休息的質量不足**。")
            summary_parts.append("\n**壓力管理策略:**")
            summary_parts.append("• **能量審計**: 列出你**一週的活動**,標註每項活動是「充電」(給你能量)還是「耗電」(消耗能量)。你會驚訝地發現,有些你以為的「休息」(如滑手機)其實也在耗電。目標是**增加充電活動,減少非必要的耗電活動**。")
            summary_parts.append("• **優先順序矩陣**: 將待辦事項分為四類:「緊急且重要」(先做)、「重要不緊急」(排程)、「緊急不重要」(授權或快速處理)、「不緊急不重要」(刪除)。大部分壓力來自**第三類假性緊急事件**佔據太多時間。")
            summary_parts.append("• **微休息習慣**: 不要等到崩潰才休息。每工作 **50 分鐘休息 10 分鐘**,真正離開座位:伸展、喝水、看窗外、做幾次深呼吸。研究顯示,頻繁的微休息比長時間工作後的長休息**更能維持生產力與心理健康**。")
            summary_parts.append("• **睡眠優先**: 疲憊時最需要的是**優質睡眠**。建立睡前儀式:睡前1小時關閉螢幕、調暗燈光、做些放鬆活動(閱讀、冥想、輕柔伸展)。睡眠是**最強大的修復機制**。")
            summary_parts.append("\n**長期思考**: 如果壓力已是常態,可能需要重新檢視生活結構:這份工作/這種生活方式真的適合我嗎?我的**價值觀與生活是否一致**?有時候,改變環境比改變自己更有效。")
            
        elif top_emotion == "快樂滿足":
            summary_parts.append("\n✨ **珍貴的光亮時刻**")
            summary_parts.append("你的對話中充滿正向情緒,這是多麼美好的狀態!但我想提醒你,**快樂不需要理由,也不需要感到愧疚**。有些人在快樂時會覺得「我是不是太幸運了」、「這不會持久的」,這種想法反而會削弱快樂。")
            summary_parts.append("\n心理學中有個概念叫「**品味(Savoring)**」,指的是**有意識地延長與加深正向體驗**。當好事發生時,大部分人會快速帶過,但你可以:")
            summary_parts.append("• **慢下來**: 當你感到快樂時,**暫停 30 秒**,告訴自己「我現在很快樂」,感受這份喜悅在身體的哪個部位(心口暖暖的?肩膀放鬆了?嘴角上揚?)。")
            summary_parts.append("• **分享喜悅**: 向他人述說好事,能**將快樂放大 2 倍**。找個會真心為你開心的人分享,而非會潑冷水的人。")
            summary_parts.append("• **快樂資料庫**: 每晚記下 **3 件好事**,無論多小(陽光很好、午餐很美味、朋友的一句話)。建立這個「**心理存款**」,在低潮時可以提領。")
            summary_parts.append("• **探索快樂源頭**: 思考**什麼帶來了這些快樂**?是成就感?是連結感?是自主性?是新鮮感?了解自己的快樂公式,你就能**主動設計更多快樂**。")
            summary_parts.append("\n研究顯示,**感恩、善行、與人連結、投入熱愛的事物**,是最能持續提升幸福感的四大因素。你已經在正向的軌道上,繼續保持,也別忘了將這份光亮分享給他人。")
            
        elif top_emotion == "平靜放鬆":
            summary_parts.append("\n✨ **內在的平靜力量**")
            summary_parts.append("能夠維持平靜,在這個快節奏、高壓力的世界裡是**難能可貴的能力**。這不是冷漠或麻木,而是一種**與世界保持適當距離的智慧**。")
            summary_parts.append("\n心理學將這種狀態稱為「**心理韌性(Resilience)**」的表現:你能在風浪中保持內在穩定,不輕易被外界擾動。這通常來自:**自我覺察、接納現實、以及對生活的掌控感**。")
            summary_parts.append("\n**深化你的平靜:**")
            summary_parts.append("• **正念練習**: 每天 **5-10 分鐘**,專注於呼吸或身體感覺,不評判、不抓取、不抗拒。這能將你的平靜**從狀態變成特質**,成為內在穩定的資源。")
            summary_parts.append("• **價值觀對齊**: 平靜常來自於「我知道什麼對我重要,我正走在對的路上」。定期檢視:**我的生活與我的價值觀一致嗎**?不一致時會產生內在衝突,一致時會感到平和。")
            summary_parts.append("• **接納練習**: 平靜不是控制一切,而是**接納無法控制的部分**。對於無法改變的事(過去、他人、某些結果),練習說「我接納它的存在」,把能量放在你能改變的地方。")
            summary_parts.append("\n你的平靜是**給自己和他人的禮物**。在混亂的時刻,你能成為穩定的力量。繼續培養這份特質,它會在你最需要時成為避風港。")
            
        elif top_emotion == "困惑迷茫":
            summary_parts.append("\n💡 **在迷霧中前行**")
            summary_parts.append("困惑和迷茫是**轉變期的標誌**,代表舊的框架不再適用,新的尚未成形。這個階段很不舒服,因為人類天生渴望確定性,但請記住:**所有成長都始於不確定**。")
            summary_parts.append("\n心理學家 William Bridges 的「轉變三階段」理論指出:結束→混亂中立區→新開始。你現在可能在**「混亂中立區」**,這裡充滿困惑,但也充滿可能性。")
            summary_parts.append("\n**在迷茫中找到方向:**")
            summary_parts.append("• **表達性書寫**: 每天 **10-15 分鐘自由書寫**,不修改、不批判,讓潛意識說話。寫下「我困惑的是...」、「我害怕的是...」、「我渴望的是...」。通常寫著寫著,**答案會自己浮現**。")
            summary_parts.append("• **實驗心態**: 不要急著找「正確答案」。把這段時間當作**實驗期**:嘗試不同的可能(副業、興趣、人際圈),觀察哪些讓你有能量、有意義。**行動會帶來答案**,而非思考。")
            summary_parts.append("• **對話澄清**: 找信任的人(朋友、導師、諮商師)深度對話。**說出口的過程本身就是整理**,他人的提問能幫你看見盲點。不要擔心「麻煩別人」,真正的朋友會樂意陪伴你。")
            summary_parts.append("• **向內探問**: 問自己:**如果我知道答案,那會是什麼?** 如果失敗不可能,我會選擇什麼?** 有時候,你其實知道答案,只是**還沒準備好承認或行動**。")
            summary_parts.append("\n**給自己時間**: 迷茫期可能持續數週到數月,這是正常的。不要強迫自己「快點想清楚」。**信任過程,答案會在對的時候出現**。而且,有時候最好的選擇不是「想清楚」,而是「在行動中逐漸清晰」。")
    
    # 情緒多樣性分析
    if len(emotion_freq) >= 3:
        top_3_emotions = [k for k, v in sorted(emotion_freq.items(), key=lambda x: x[1], reverse=True)[:3]]
        summary_parts.append(f"\n🎨 **你的情緒色盤**")
        summary_parts.append(f"你的情緒樣貌呈現豐富的多元性,主要的三種色調是:**{top_3_emotions[0]}**({emotion_freq[top_3_emotions[0]]}次)、**{top_3_emotions[1]}**({emotion_freq[top_3_emotions[1]]}次)、**{top_3_emotions[2]}**({emotion_freq[top_3_emotions[2]]}次)。")
        summary_parts.append("\n這種多樣性顯示你是一個**情感豐富的人**,能夠體驗生活的各種層次。情緒健康不是「永遠快樂」或「沒有負面情緒」,而是:")
        summary_parts.append("• **覺察**: 你能意識到自己的情緒嗎?")
        summary_parts.append("• **接納**: 你能允許所有情緒存在,而不批判嗎?")
        summary_parts.append("• **表達**: 你能用健康的方式表達情緒嗎?")
        summary_parts.append("• **調節**: 當情緒過於強烈時,你能找到方法調節嗎?")
        summary_parts.append("\n你正在透過對話練習這些能力,這本身就是**情緒智慧(Emotional Intelligence)**的體現。繼續這段自我探索的旅程。")
    
    # 核心議題的深度分析
    if topics:
        top_topic = max(topics.items(), key=lambda x: x[1])[0]
        topic_score = topics[top_topic]
        
        summary_parts.append(f"\n🎯 **生命的核心命題**")
        summary_parts.append(f"你最關注的生命領域是「**{top_topic}**」(關注度指數: {round(topic_score, 1)})。這個主題在你的對話中反覆出現,顯示它對你的生活有著**深刻的影響**。")
        
        if top_topic == "工作職場":
            summary_parts.append("\n工作不只是謀生工具,它往往與**自我價值、成就感、人生意義**緊密相連。當工作成為主要關注點,值得深入思考:")
            summary_parts.append("• **意義感**: 這份工作與你的**長期目標、價值觀**一致嗎?還是只是「還可以」或「還能忍受」?")
            summary_parts.append("• **成長空間**: 你在其中學到新技能、拓展能力嗎?還是已經進入舒適區或倦怠期?")
            summary_parts.append("• **工作與自我的關係**: 你是否過度將**自我價值綁定在工作表現**上?記住,你的價值不等於你的產出。")
            summary_parts.append("• **界線設定**: 能否清楚區分**工作時間與私人時間**?建立「下班儀式」(換衣服、運動、做晚餐)幫助大腦切換。")
            summary_parts.append("\n如果工作帶來持續的壓力或不滿,可以思考:**這是環境的問題(可以轉換),還是期待的問題(可以調整)?** 有時改變環境,有時改變心態,兩者都是有效的策略。")
            
        elif top_topic == "人際關係":
            summary_parts.append("\n人際關係是**情緒的主要來源**,也是心理健康的關鍵因素。哈佛大學 75 年的幸福研究發現:**良好的關係是幸福與健康的最大預測因子**。")
            summary_parts.append("但關係也是壓力源。當人際成為主要困擾,值得探索:")
            summary_parts.append("• **界線議題**: 你是否清楚自己的界線?能溫和但堅定地說「不」嗎?還是總是**過度配合、壓抑需求**?")
            summary_parts.append("• **依附模式**: 你在關係中的模式是什麼?**焦慮型**(過度擔心被拋棄)、**迴避型**(保持距離以自保)、還是**安全型**(舒適地親近也能獨立)?了解自己的依附風格,能幫助你理解關係中的反應。")
            summary_parts.append("• **關係盤點**: 將你的人際關係分類:哪些讓你**充能**(相處後感到開心、被理解)?哪些讓你**耗能**(相處後感到疲憊、委屈)?有意識地投資充能關係,減少耗能互動。")
            summary_parts.append("• **溝通技巧**: 很多衝突來自**溝通不良**,而非根本差異。學習「非暴力溝通」:觀察(不評判)→感受→需求→請求,能大幅改善關係品質。")
            summary_parts.append("\n記住:**你無法改變他人,但能改變自己在關係中的位置**。有時候,最健康的選擇是離開不健康的關係;有時候,是學習新的相處方式。")
            
        elif top_topic == "自我認同":
            summary_parts.append("\n「我是誰?」「我要往哪裡去?」這些**存在性問題**是人生重要的探索。當自我認同成為核心關注,你正處於**自我整合的關鍵期**。")
            summary_parts.append("\n心理學家 Erik Erikson 提出,人生各階段都有不同的認同任務。**自我認同不是找到「標準答案」,而是持續整合經驗、價值觀與選擇的過程**。")
            summary_parts.append("• **價值觀澄清**: 做「**價值觀排序練習**」:列出 10-15 個重要價值(如自由、家庭、成就、創造、服務、誠實、冒險等),選出最重要的 **5 個**,然後檢視:我的生活體現這些價值嗎?")
            summary_parts.append("• **多元身分**: 你不是單一身分,而是**多重角色的組合**(工作者、朋友、子女、伴侶、創作者...)。當某個角色受挫,記得你還有其他身分。不要讓單一領域定義你的全部價值。")
            summary_parts.append("• **敘事治療**: 寫下你的**人生故事**:重要轉折點、影響你的人與事、你的價值觀從何而來。透過書寫,你能看見自己的主線與發展軌跡。")
            summary_parts.append("• **實驗與探索**: 認同不是「想出來」的,而是**做出來**的。嘗試新的興趣、角色、環境,在行動中觀察「什麼讓我感到**真實**、**有意義**、**有活力**」。")
            summary_parts.append("\n**給自己耐心**: 自我認同的探索可能持續數年,尤其在人生轉換期(畢業、換工作、失戀、搬家)會特別強烈。**這不是危機,而是成長的邀請**。允許自己在不確定中前行,答案會逐漸清晰。")
            
        elif top_topic == "情感愛情":
            summary_parts.append("\n親密關係是**最深刻的自我認識途徑**,也是最大的成長契機。在關係中,我們會看見自己的需求、恐懼、依附模式、以及愛與被愛的能力。")
            summary_parts.append("\n無論你處於哪個階段(單身、曖昧、熱戀、穩定期、衝突期、分離),都值得思考:")
            summary_parts.append("• **你的依附風格**: 關係心理學研究發現,**童年與主要照顧者的互動**形成我們的依附模式。**安全型**能親近也能獨立;**焦慮型**渴望親密卻怕被拋棄;**迴避型**重視獨立卻難以親密。了解自己與對方的依附風格,能解釋很多關係動力。")
            summary_parts.append("• **完整的自己**: 健康的關係不是「你完整了我」,而是「**兩個完整的人選擇同行**」。在關係中,你是否保持自我(興趣、朋友、目標)?還是過度融合、失去界線?")
            summary_parts.append("• **溝通與修復**: 所有關係都會有衝突,**重點不是不吵架,而是如何修復**。能否在衝突後:承認錯誤、表達感受、理解對方、找到解決方式?修復能力決定關係品質。")
            summary_parts.append("• **愛的語言**: Gary Chapman 提出五種愛的語言:肯定言語、精心時刻、贈送禮物、服務行動、身體接觸。**你如何表達愛?你如何接收愛?** 不匹配時會產生「我明明很愛你,你卻感受不到」的困境。")
            summary_parts.append("\n如果關係帶來持續的痛苦(不被尊重、控制、情緒勒索、暴力),請記得:**你值得被好好對待**。離開不健康的關係不是失敗,而是愛自己的勇氣。")
            
        elif top_topic == "學業成長":
            summary_parts.append("\n學習是終身的旅程,但在學校系統中,它常被窄化為「考試」與「成績」。當學業成為主要關注,值得重新思考:**學習的本質是什麼?**")
            summary_parts.append("\n心理學家 Carol Dweck 提出「**成長型思維 vs. 固定型思維**」:固定型認為能力是天生的(「我就是不擅長數學」),成長型相信能力可以培養(「我還不擅長,但可以進步」)。這個差異會**深刻影響學習動機與表現**。")
            summary_parts.append("• **刻意練習**: 不是「花很多時間」就會進步,而是要**專注在稍微超出能力範圍的挑戰**,並立即獲得反饋、調整策略。這是頂尖表現者的共同特質。")
            summary_parts.append("• **學習動機**: 你為什麼學習?是**內在動機**(好奇、興趣、成就感)還是**外在動機**(分數、他人期待、避免懲罰)?研究顯示,內在動機能帶來更深入的學習與更持久的興趣。如何找到學習的意義,而非只是完成任務?")
            summary_parts.append("• **錯誤與失敗**: 神經科學研究顯示,**大腦在犯錯時學得最多**。將錯誤視為「成長證據」而非「能力不足」,你會更願意挑戰困難、冒險嘗試。")
            summary_parts.append("• **後設認知**: 不只是學習,更要「**學習如何學習**」。定期反思:什麼方法對我有效?我的學習瓶頸在哪?如何調整策略?這種自我覺察能提升學習效率 2-3 倍。")
            summary_parts.append("\n記住:**成績不代表你的價值,也不決定你的未來**。它只是現階段學習成果的一個指標。培養真正的能力(思考、創造、解決問題、與人協作)比分數更重要。")
            
        elif top_topic == "身心健康":
            summary_parts.append("\n當健康成為主要關注,身體可能在向你發出訊號:**我需要被照顧**、**壓力太大了**、**生活失衡了**。")
            summary_parts.append("\n現代醫學越來越認識到「**心身醫學(Psychosomatic Medicine)**」的重要性:**身心是一體的**。心理壓力會表現為身體症狀(頭痛、胃痛、失眠、免疫力下降),而身體不適也會影響心理狀態(疲勞導致憂鬱、疼痛導致焦慮)。")
            summary_parts.append("• **基本的自我照顧**: 這不是「應該做」,而是**不可協商的必需品**:")
            summary_parts.append("  - **睡眠**: 成人需要 7-9 小時。睡眠不足會導致情緒失調、認知下降、免疫力降低。把睡眠當作「**充電**」,而非「浪費時間」。")
            summary_parts.append("  - **運動**: 不需要激烈運動,**每天 20-30 分鐘中等強度活動**(快走、游泳、跳舞)就能顯著改善情緒與健康。運動是天然的抗憂鬱藥。")
            summary_parts.append("  - **飲食**: 你吃進去的食物**直接影響大腦功能與情緒**。減少精緻糖、增加蔬果與 Omega-3,能改善心理健康。")
            summary_parts.append("  - **社交連結**: 孤立會增加死亡率 30%,相當於每天抽 15 根菸。定期與人連結(面對面,而非只是線上)是健康的關鍵。")
            summary_parts.append("• **壓力與疾病**: 長期壓力會導致**慢性發炎**,這是許多疾病的根源(心血管疾病、自體免疫疾病、癌症)。學習壓力管理不是奢侈,而是**健康的必要投資**。")
            summary_parts.append("• **身體訊號**: 學習**傾聽身體**:哪裡緊繃?哪裡疼痛?哪裡有壓力?身體比頭腦更誠實,它會告訴你真實的狀態。")
            summary_parts.append("\n如果身體症狀持續,請尋求醫療協助。同時也考慮:**這些症狀背後,有什麼心理或生活壓力因素?** 有時候,治療症狀的同時,也需要處理根源。")
    
    # 情緒趨勢的深度分析
    if trend_info and trend_info.get("trend") != "insufficient_data":
        trend = trend_info.get("trend")
        description = trend_info.get("description", "")
        
        summary_parts.append(f"\n📈 **情緒軌跡觀察**")
        summary_parts.append(f"{description}")
        
        if trend == "concerning":
            summary_parts.append("\n當負面情緒持續佔主導,這是**身心需要照顧的明確信號**。這不代表你「不夠堅強」或「想太多」,而是你的系統在說:**現在的狀態無法持續**。")
            summary_parts.append("\n**建議行動**:")
            summary_parts.append("• **尋求專業支持**: 心理諮商不是「崩潰了才去」,而是**預防與自我投資**。就像身體不舒服看醫生,心理不適也該尋求專業。")
            summary_parts.append("• **建立支持系統**: 找 1-3 位可以真實表達情緒的人(朋友、家人、支持團體)。**說出來本身就是療癒**。")
            summary_parts.append("• **降低要求**: 暫時降低對自己的期待,**只做必要的事**。給自己復原的時間與空間。")
            summary_parts.append("• **危機資源**: 如果有自傷或傷人念頭,請立即聯繫**生命線 1995**、**張老師 1980**、**安心專線 1925**,或前往急診。你的生命很重要。")
            
        elif trend == "fluctuating":
            summary_parts.append("\n情緒的起伏是正常的,但如果**波動過大**影響生活(今天很high明天很low、情緒突然轉變、難以預測自己的狀態),值得關注。")
            summary_parts.append("\n**可能原因**:")
            summary_parts.append("• **外在壓力源**: 生活中是否有不穩定因素(工作變動、關係衝突、經濟壓力)?")
            summary_parts.append("• **生理因素**: 睡眠不足、荷爾蒙變化、血糖波動都會影響情緒穩定。")
            summary_parts.append("• **情緒調節能力**: 是否缺乏有效的情緒管理策略?")
            summary_parts.append("\n**穩定策略**:")
            summary_parts.append("• **規律作息**: 固定的睡眠、飲食、運動時間能**穩定生理與心理節奏**。")
            summary_parts.append("• **情緒日誌**: 記錄情緒變化與觸發因素,找出**模式與週期**(是否與月經週期、工作週期、特定人事物有關)。")
            summary_parts.append("• **學習技巧**: 正念、呼吸練習、漸進式肌肉放鬆,都能幫助你在情緒高峰時**找到內在穩定錨**。")
            
        else:  # stable
            summary_parts.append("\n情緒的相對穩定顯示你具備良好的**心理平衡能力**。這不是說你不會有情緒,而是你能在起伏中維持一定的穩定性。")
            summary_parts.append("\n**維持穩定的秘訣**:")
            summary_parts.append("• 繼續你正在做的事(規律作息、支持系統、壓力管理)")
            summary_parts.append("• 定期「**心理健檢**」:每週花 10 分鐘反思:我的狀態如何?有什麼需要調整?")
            summary_parts.append("• 不要等到崩潰才處理,**預防勝於治療**")
    
    # 整合性結語 - 溫暖且賦能
    summary_parts.append(f"\n💝 **寫在最後**")
    summary_parts.append(f"這 **{message_count} 次對話**不只是文字紀錄,更是你**自我理解的旅程**。每一次你願意說出內心感受,都是勇氣的展現。")
    summary_parts.append("\n心理學家 Carl Rogers 說:「**當我被理解,我就能成長**。」在這裡,你不需要偽裝、不需要完美,只需要真實。")
    summary_parts.append("\n**請記住幾件事**:")
    summary_parts.append("• **情緒沒有對錯**: 所有情緒都是有效的訊號,包括那些「不應該有」的情緒(憤怒、嫉妒、悲傷)。")
    summary_parts.append("• **改變需要時間**: 心理成長不是線性的,會有進步、停滯、甚至倒退。**這都是過程的一部分**。")
    summary_parts.append("• **你不孤單**: 你經歷的困擾,很多人也在經歷。尋求幫助不是軟弱,而是智慧。")
    summary_parts.append("• **你值得被好好對待**: 包括被自己好好對待。對自己溫柔一點,就像你對待摯友那樣。")
    summary_parts.append("\n持續這段對話,持續自我探索。**你正在成為更理解自己、更完整的自己**。這條路不總是容易,但你走得很好。我會一直在這裡,陪你一起前行。")
    
    return "\n\n".join(summary_parts)

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