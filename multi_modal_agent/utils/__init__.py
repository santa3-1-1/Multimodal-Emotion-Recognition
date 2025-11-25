# multi_modal_agent/utils/__init__.py
# package init for utils
from .label_map import DEFAULT_CANDIDATES, LABEL_MAP
from .language_detect import detect_language

__all__ = ["DEFAULT_CANDIDATES", "LABEL_MAP", "detect_language"]
