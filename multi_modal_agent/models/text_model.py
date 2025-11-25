# multi_modal_agent/models/text_model.py
"""
TextWrapper: 基于 transformers pipeline("sentiment-analysis") 对输入文本做情感推断，
并将 pipeline 的 label+score 映射为 candidate_texts 上的软概率分布（soft labels）。
"""
from typing import List
import torch
from transformers import pipeline

from ..utils.label_map import DEFAULT_CANDIDATES, LABEL_MAP  # relative import within package


class TextWrapper:
    def __init__(self, device: str = None):
        """
        初始化文本情感分析 pipeline
        device: "cpu" or "cuda" or None(auto-detect)
        """
        # pipeline device mapping: -1 -> cpu, 0 -> cuda:0
        self.device = device
        self._device_flag = -1
        if device is not None:
            if device.startswith("cuda"):
                # choose first GPU
                self._device_flag = 0
            else:
                self._device_flag = -1
        else:
            # auto select
            try:
                import torch as _torch
                self._device_flag = 0 if _torch.cuda.is_available() else -1
            except Exception:
                self._device_flag = -1

        # create pipeline
        # use default model for sentiment-analysis
        self.nlp = pipeline("sentiment-analysis", device=self._device_flag)

        # candidates order must match DEFAULT_CANDIDATES
        self.candidates = DEFAULT_CANDIDATES

    def _map_label_to_distribution(self, label: str, score: float) -> torch.Tensor:
        """
        使用 LABEL_MAP 将 pipeline label 和 score 转成 candidates 上的概率向量
        LABEL_MAP 中每个 label 对应一个与 candidates 同长度的分布（和为1）
        我们把 pipeline 的置信度 score 与映射分布相乘，然后归一化，得到最终概率分布。
        """
        label_key = label.upper()
        # Some pipelines return 'POSITIVE'/'NEGATIVE' or 'LABEL_0' style; check LABEL_MAP keys
        if label_key not in LABEL_MAP:
            # try lower-case alternatives
            label_key = label_key.lower()
        mapping = LABEL_MAP.get(label_key)
        if mapping is None:
            # fallback: uniform distribution
            import torch as _torch
            return _torch.ones(len(self.candidates)) / len(self.candidates)

        # mapping is expected to be list/tuple sum to 1 of same length
        import torch as _torch
        mapping_tensor = _torch.tensor(mapping, dtype=_torch.float32)

        # incorporate score (0..1). A simple scheme: multiply and normalize.
        # This preserves mapping proportions while reflecting classifier confidence.
        scaled = mapping_tensor * float(score)
        if scaled.sum() <= 0:
            # fallback to uniform
            return _torch.ones(len(self.candidates), dtype=_torch.float32) / len(self.candidates)
        probs = scaled / scaled.sum()
        return probs

    def get_probs(self, text: str) -> torch.Tensor:
        """
        对文本进行情感分析并返回与 DEFAULT_CANDIDATES 对应的概率向量
        """
        if text is None or text == "":
            # empty text -> uniform distribution
            import torch as _torch
            return _torch.ones(len(self.candidates), dtype=_torch.float32) / len(self.candidates)

        out = self.nlp(text, truncation=True)
        # pipeline usually returns list of dicts e.g. [{"label":"POSITIVE","score":0.98}]
        if isinstance(out, list) and len(out) > 0:
            pred = out[0]
            label = pred.get("label", "NEUTRAL")
            score = float(pred.get("score", 0.5))
        else:
            label = "NEUTRAL"
            score = 0.5

        probs = self._map_label_to_distribution(label, score)
        return probs
