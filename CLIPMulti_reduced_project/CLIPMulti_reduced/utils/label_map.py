# utils/label_map.py
from typing import Dict

# 三类 -> 五类 soft label 映射
# 5 类对应顺序：
# 0: happy | 1: sad | 2: angry | 3: calm | 4: anxious
SOFT_MAP = {
    "positive": {0: 0.7, 3: 0.3},          # 正向情绪 = 开心 + 平静
    "neutral":  {3: 0.6, 4: 0.4},          # 中性情绪 = 平静 + 紧张
    "negative": {1: 0.6, 2: 0.4},          # 负向情绪 = 悲伤 + 愤怒
}

# 硬映射备用
HARD_MAP = {
    "positive": 0,
    "neutral": 3,
    "negative": 1
}

def map_label_soft(label: str) -> Dict[int, float]:
    """将三分类标签转换为五类 soft 分布"""
    if not isinstance(label, str):
        label = str(label)
    k = label.strip().lower()
    return SOFT_MAP.get(k, {3: 1.0})  # 默认 calm

def map_label_hard(label: str) -> int:
    """将三分类标签转换为五类索引"""
    if not isinstance(label, str):
        label = str(label)
    k = label.strip().lower()
    return HARD_MAP.get(k, 3)
