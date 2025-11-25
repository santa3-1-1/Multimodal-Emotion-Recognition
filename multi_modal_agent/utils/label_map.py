# multi_modal_agent/utils/label_map.py
"""
标签映射文件

DEFAULT_CANDIDATES: 全局候选情绪标签顺序（必须与其他模块一致）
LABEL_MAP: 将文本情绪分类器的标签（如 POSITIVE / NEGATIVE / NEUTRAL）映射为
对 DEFAULT_CANDIDATES 的软分布（和为1）

映射策略（示例）：
- POSITIVE -> 主要映射到 "happy"
- NEGATIVE -> 分配到 "sad","angry","anxious" 三项（按比例）
- NEUTRAL  -> "calm"
"""
from typing import List, Dict

# 全局候选情绪（顺序固定）
DEFAULT_CANDIDATES: List[str] = ["happy", "sad", "angry", "calm", "anxious"]

# LABEL_MAP: key -> distribution over DEFAULT_CANDIDATES (length must match DEFAULT_CANDIDATES)
# 分布为软标签（和为1）
LABEL_MAP: Dict[str, List[float]] = {
    # 主流 transformers sentiment-analysis 通常返回 'POSITIVE' / 'NEGATIVE'
    "POSITIVE": [1.0, 0.0, 0.0, 0.0, 0.0],  # 完全映射为 happy
    "NEGATIVE": [0.0, 0.6, 0.3, 0.0, 0.1],  # 主要是 sad，其次 angry，少量 anxious
    "NEUTRAL": [0.0, 0.0, 0.0, 1.0, 0.0],   # calm
    # lower-case keys for robustness
    "positive": [1.0, 0.0, 0.0, 0.0, 0.0],
    "negative": [0.0, 0.6, 0.3, 0.0, 0.1],
    "neutral": [0.0, 0.0, 0.0, 1.0, 0.0],
    # Some models may return other labels; provide a fallback mapping (uniform)
    "OTHER": [1/5, 1/5, 1/5, 1/5, 1/5],
}

# Exported for convenience
__all__ = ["DEFAULT_CANDIDATES", "LABEL_MAP"]
