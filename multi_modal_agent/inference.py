# multi_modal_agent/inference.py
"""
predict(image_path, text):
    - 调用 CLIPWrapper.get_probs
    - 调用 TextWrapper.get_probs
    - 使用 fusion.rule_fusion 融合
输出：
    {"scores": {label: float,...}, "top": label}
"""
from typing import Dict, Any
import torch

from .models import CLIPWrapper, TextWrapper
from .utils.label_map import DEFAULT_CANDIDATES
from .fusion import rule_fusion


# create singletons (lazy init on module import)
_clip_wrapper = None
_text_wrapper = None


def _get_clip_wrapper():
    global _clip_wrapper
    if _clip_wrapper is None:
        _clip_wrapper = CLIPWrapper()
    return _clip_wrapper


def _get_text_wrapper():
    global _text_wrapper
    if _text_wrapper is None:
        _text_wrapper = TextWrapper()
    return _text_wrapper


def predict(image_path: str, text: str) -> Dict[str, Any]:
    """
    主预测函数，返回字典：
    {
       "scores": {"happy":0.7, "sad":0.1, ...},
       "top": "happy"
    }
    """
    # Load models
    clip = _get_clip_wrapper()
    text_model = _get_text_wrapper()

    # Candidates
    candidates = DEFAULT_CANDIDATES

    # Get probs
    try:
        clip_probs = clip.get_probs(image_path, candidates)  # torch.Tensor
    except Exception as e:
        # If image failed, fallback to uniform
        import torch as _torch
        clip_probs = _torch.ones(len(candidates), dtype=_torch.float32) / len(candidates)

    try:
        text_probs = text_model.get_probs(text)  # torch.Tensor
    except Exception as e:
        import torch as _torch
        text_probs = _torch.ones(len(candidates), dtype=_torch.float32) / len(candidates)

    # fuse
    final = rule_fusion(clip_probs, text_probs)

    # prepare output
    scores = {candidates[i]: float(final[i].item()) for i in range(len(candidates))}
    # top label
    top_label = max(scores.items(), key=lambda x: x[1])[0]

    result = {
        "scores": scores,
        "top": top_label
    }
    return result
