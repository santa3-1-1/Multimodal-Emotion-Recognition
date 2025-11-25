# multi_modal_agent/fusion.py
"""
融合逻辑：rule_fusion
默认权重：
    final = 0.6 * clip_probs + 0.4 * text_probs

输入：
    clip_probs: torch.Tensor shape (N,)
    text_probs: torch.Tensor shape (N,)

返回：
    torch.Tensor shape (N,) （归一化概率）
"""
import torch

DEFAULT_CLIP_WEIGHT = 0.6
DEFAULT_TEXT_WEIGHT = 0.4


def rule_fusion(clip_probs: torch.Tensor, text_probs: torch.Tensor,
                clip_weight: float = DEFAULT_CLIP_WEIGHT,
                text_weight: float = DEFAULT_TEXT_WEIGHT) -> torch.Tensor:
    if clip_probs is None and text_probs is None:
        raise ValueError("Both clip_probs and text_probs are None")

    if clip_probs is None:
        final = text_probs.clone()
    elif text_probs is None:
        final = clip_probs.clone()
    else:
        # ensure float tensors
        clip_probs = clip_probs.float()
        text_probs = text_probs.float()
        # simple weighted sum
        final = clip_weight * clip_probs + text_weight * text_probs

    # ensure non-negative
    final = torch.clamp(final, min=0.0)
    s = final.sum()
    if s > 0:
        final = final / s
    else:
        # fallback to uniform
        final = torch.ones_like(final) / final.numel()
    return final
