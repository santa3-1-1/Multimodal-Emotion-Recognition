# multi_modal_agent/utils/language_detect.py
"""
language_detect.py

占位模板：如果需要做语言探测（例如处理非英文文本）可以在这里实现。
当前为占位返回 'en'（英文）以保证 pipeline 能工作；你可以替换为真实实现。

示例使用方法（可选）:
from .language_detect import detect_language
lang = detect_language("I feel good")
"""
from typing import Optional

def detect_language(text: str) -> Optional[str]:
    """
    占位实现：简单判断若包含 ASCII 字符则返回 'en'，否则返回 None。
    建议在未来用 langdetect 或 fasttext 进行替换。
    """
    if not text:
        return None
    try:
        # 如果大部分字符为 ASCII，返回 en
        ascii_chars = sum(1 for ch in text if ord(ch) < 128)
        if ascii_chars / max(1, len(text)) > 0.6:
            return "en"
    except Exception:
        pass
    return None
