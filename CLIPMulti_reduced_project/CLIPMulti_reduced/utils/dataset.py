import pandas as pd
from PIL import Image
from torch.utils.data import Dataset
import torch
from utils.label_map import map_label_soft, map_label_hard

class ImageTextDataset(Dataset):
    def __init__(self, csv_path, image_root=None, transform=None, soft_label=True):
        """
        csv需包含列：image_path, text(或caption), emotion(标签为positive/neutral/negative)
        """
        self.df = pd.read_csv(csv_path)
        self.image_root = image_root
        self.transform = transform
        self.soft_label = soft_label

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = row['image_path'] if self.image_root is None else f"{self.image_root}/{row['image_path']}"
        img = Image.open(img_path).convert('RGB')
        if self.transform is not None:
            img = self.transform(img)

        text = row.get('text', row.get('caption', ''))
        label_str = str(row.get('emotion', '')).strip().lower()

        if self.soft_label:
            dist = map_label_soft(label_str)
            probs = [0.0] * 5
            for k, v in dist.items():
                probs[k] = v
            label_tensor = torch.tensor(probs, dtype=torch.float32)
        else:
            label_tensor = torch.tensor(map_label_hard(label_str), dtype=torch.long)

        return img, text, label_tensor
