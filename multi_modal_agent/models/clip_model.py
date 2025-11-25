# multi_modal_agent/models/clip_model.py
"""
CLIPWrapper: 使用 openai/clip-vit-base-patch16 进行图像->文本候选对比推理
提供 get_probs(image_path, candidate_texts) -> torch.Tensor(probabilities)
"""
from typing import List
import os

import torch
import torch.nn.functional as F
from transformers import CLIPModel, CLIPProcessor
from PIL import Image


class CLIPWrapper:
    def __init__(self, model_name: str = "openai/clip-vit-base-patch16", device: str = None):
        """
        初始化 CLIP 模型与 processor
        """
        self.model_name = model_name
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        # Load model and processor
        self.model = CLIPModel.from_pretrained(self.model_name).to(self.device)
        self.processor = CLIPProcessor.from_pretrained(self.model_name)

    def _load_image(self, image_path: str) -> Image.Image:
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image not found: {image_path}")
        image = Image.open(image_path).convert("RGB")
        return image

    def get_probs(self, image_path: str, candidate_texts: List[str]) -> torch.Tensor:
        """
        对单张图片与若干候选文本求概率分布

        Args:
            image_path: 本地图片路径
            candidate_texts: 文本候选列表（e.g. ["happy","sad","angry","calm","anxious"]）

        Returns:
            probs: torch.Tensor shape (len(candidate_texts),) - 概率分布 (sum -> 1)
        """
        image = self._load_image(image_path)

        # Prepare inputs for CLIP
        inputs = self.processor(text=candidate_texts, images=image, return_tensors="pt", padding=True)
        # move tensors to device
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        # forward
        with torch.no_grad():
            outputs = self.model(**inputs)
            # logits_per_image: (batch_image=1, num_texts)
            logits_per_image = outputs.logits_per_image
            # take first (and only) image
            logits = logits_per_image[0]
            probs = F.softmax(logits, dim=0)

        return probs.cpu()
