# backend/app/services/memory_service.py
import re
from typing import List, Dict, Set
from collections import Counter, defaultdict
from datetime import datetime
from sqlalchemy.orm import Session
from app.models.chat import ChatMessage


def get_user_memory_context(db: Session, pid: str, bot_type: str) -> str:
    """
    提取用戶的重要記憶資訊，生成記憶上下文
    
    Args:
        db: 資料庫連線
        pid: 用戶 PID
        bot_type: 機器人類型
        
    Returns:
        記憶上下文字串
    """
    
    # 獲取歷史對話（最多100則）
    messages = (
        db.query(ChatMessage)
        .filter(ChatMessage.pid == pid, ChatMessage.bot_type == bot_type)
        .order_by(ChatMessage.created_at.desc())
        .limit(100)
        .all()
    )
    
    if len(messages) < 5:
        return ""
    
    # 提取關鍵資訊
    memory_items = []
    
    # 1. 提取核心話題
    topics = extract_key_topics(messages)
    if topics:
        memory_items.append(f"經常討論的話題：{', '.join(topics)}")
    
    # 2. 提取提及的人物
    people = extract_mentioned_people(messages)
    if people:
        memory_items.append(f"提到的重要人物：{', '.join(people)}")
    
    # 3. 提取設定的目標
    goals = extract_goals(messages)
    if goals:
        memory_items.append(f"提過的目標/計畫：{', '.join(goals)}")
    
    # 4. 提取情緒模式
    emotion_pattern = extract_emotion_patterns(messages)
    if emotion_pattern:
        memory_items.append(f"情緒特徵：{emotion_pattern}")
    
    # 5. 提取重複出現的困擾
    recurring_issues = extract_recurring_issues(messages)
    if recurring_issues:
        memory_items.append(f"反覆出現的困擾：{', '.join(recurring_issues)}")
    
    if not memory_items:
        return ""
    
    # 組合記憶上下文
    context = "\n\n[用戶背景記憶 - 請自然地運用這些資訊，不要直接複述]\n"
    context += "\n".join(f"- {item}" for item in memory_items)
    context += "\n"
    
    return context


def extract_key_topics(messages: List[ChatMessage]) -> List[str]:
    """提取對話中的關鍵主題"""
    
    topics_keywords = {
        "工作壓力": ["工作", "老闆", "同事", "加班", "職場", "上班", "公司"],
        "人際關係": ["朋友", "家人", "關係", "吵架", "社交", "相處"],
        "感情問題": ["感情", "戀愛", "分手", "伴侶", "喜歡", "曖昧", "交往"],
        "學業": ["考試", "成績", "學校", "課業", "讀書", "作業"],
        "自我認同": ["自己", "迷惘", "價值", "人生", "意義", "未來"],
        "家庭": ["爸", "媽", "父母", "家裡", "家人"],
        "焦慮": ["焦慮", "擔心", "緊張", "不安", "恐慌"],
        "睡眠問題": ["失眠", "睡不著", "睡眠", "做夢", "惡夢"]
    }
    
    found_topics = []
    all_text = " ".join([m.content for m in messages if m.role == "user"])
    
    topic_counts = {}
    for topic, keywords in topics_keywords.items():
        count = sum(all_text.count(kw) for kw in keywords)
        if count >= 2:  # 至少出現2次
            topic_counts[topic] = count
    
    # 按出現頻率排序
    sorted_topics = sorted(topic_counts.items(), key=lambda x: x[1], reverse=True)
    found_topics = [topic for topic, _ in sorted_topics[:3]]
    
    return found_topics


def extract_mentioned_people(messages: List[ChatMessage]) -> List[str]:
    """提取對話中提到的重要人物"""
    
    people = set()
    
    # 人物關係詞
    relationship_patterns = [
        r"(?:我的)?(?:男|女)?朋友",
        r"(?:我)?(?:男|女)?同事",
        r"(?:我的)?老闆",
        r"(?:我的)?主管",
        r"(?:我的)?(?:爸|媽|父母|家人)",
        r"(?:我的)?伴侶",
        r"(?:我的)?(?:男|女)友"
    ]
    
    for msg in messages:
        if msg.role == "user":
            for pattern in relationship_patterns:
                if re.search(pattern, msg.content):
                    match = re.search(pattern, msg.content)
                    if match:
                        people.add(match.group(0).replace("我的", "").replace("我", ""))
    
    return list(people)[:5]


def extract_goals(messages: List[ChatMessage]) -> List[str]:
    """提取用戶提到的目標或計畫"""
    
    goal_patterns = [
        r"想要([^,。\n]{2,15})",
        r"希望([^,。\n]{2,15})",
        r"打算([^,。\n]{2,15})",
        r"計[劃畫]([^,。\n]{2,15})",
        r"目標是([^,。\n]{2,15})"
    ]
    
    goals = []
    recent_messages = [m for m in messages[:30] if m.role == "user"]  # 只看最近30則
    
    for msg in recent_messages:
        for pattern in goal_patterns:
            matches = re.findall(pattern, msg.content)
            for match in matches:
                cleaned = match.strip()
                if len(cleaned) >= 3 and cleaned not in goals:
                    goals.append(cleaned)
    
    return goals[:3]


def extract_emotion_patterns(messages: List[ChatMessage]) -> str:
    """分析情緒模式"""
    
    emotions = []
    
    emotion_keywords = {
        "焦慮不安": ["焦慮", "擔心", "緊張", "不安", "害怕", "恐懼"],
        "低落憂鬱": ["難過", "憂鬱", "沮喪", "失落", "悲傷", "痛苦"],
        "煩躁生氣": ["煩", "火大", "生氣", "不爽", "憤怒", "氣"],
        "疲憊無力": ["累", "疲憊", "無力", "沒力氣", "撐不住"]
    }
    
    for msg in messages:
        if msg.role == "user":
            for emotion, keywords in emotion_keywords.items():
                if any(kw in msg.content for kw in keywords):
                    emotions.append(emotion)
    
    if not emotions:
        return ""
    
    most_common = Counter(emotions).most_common(2)
    
    if len(most_common) == 1:
        return f"經常感到{most_common[0][0]}"
    else:
        return f"經常感到{most_common[0][0]}和{most_common[1][0]}"


def extract_recurring_issues(messages: List[ChatMessage]) -> List[str]:
    """提取反覆出現的困擾"""
    
    issue_keywords = {
        "拖延": ["拖延", "一直拖", "不想做", "做不下去"],
        "完美主義": ["完美", "不夠好", "做不好", "要求很高"],
        "自我懷疑": ["懷疑自己", "不相信自己", "沒自信", "覺得自己不行"],
        "社交困難": ["不敢說", "不知道怎麼", "講話", "社交", "尷尬"],
        "情緒失控": ["控制不住", "忍不住", "爆發", "失控"],
        "失眠": ["睡不著", "失眠", "睡不好", "一直醒"]
    }
    
    issue_counts = defaultdict(int)
    
    for msg in messages:
        if msg.role == "user":
            for issue, keywords in issue_keywords.items():
                if any(kw in msg.content for kw in keywords):
                    issue_counts[issue] += 1
    
    # 只保留出現3次以上的
    recurring = [issue for issue, count in issue_counts.items() if count >= 3]
    
    return recurring[:3]