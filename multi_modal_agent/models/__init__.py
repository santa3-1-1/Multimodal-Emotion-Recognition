# multi_modal_agent/models/__init__.py
# package init for models
from .clip_model import CLIPWrapper
from .text_model import TextWrapper

__all__ = ["CLIPWrapper", "TextWrapper"]
