from transformers import CLIPModel, CLIPProcessor
import torch


class CLIPEncoder:
    def __init__(self, model_name, device="cuda"):
        print(f"🧠 Loading CLIP model from {model_name} to {device}")

        # 确定设备
        if device == "cuda" and torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")

        # 在目标设备上加载模型
        self.model = CLIPModel.from_pretrained(
            model_name,
            local_files_only=True
        ).to(self.device)

        # 设置为评估模式
        self.model.eval()

        self.processor = CLIPProcessor.from_pretrained(model_name, local_files_only=True)
        print(f"✅ CLIP模型加载完成，设备: {self.device}")

    def encode_image(self, images):
        # 自动检测输入类型
        if isinstance(images, torch.Tensor):
            # 已经是 tensor，直接送入模型，不再二次预处理
            with torch.no_grad():
                image_features = self.model.get_image_features(images)
            return image_features
        else:
            # 否则假设是 PIL 图像或路径，进行标准预处理
            inputs = self.processor(images=images, return_tensors="pt").to(self.device)
            with torch.no_grad():
                image_features = self.model.get_image_features(**inputs)
            return image_features

    def encode_text(self, texts):
        """编码文本"""
        if isinstance(texts, str):
            texts = [texts]

        inputs = self.processor(
            text=texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=77
        ).to(self.device)

        with torch.no_grad():
            txt_features = self.model.get_text_features(**inputs)
        return txt_features