
import torch
import torch.nn as nn

class FusionHead(nn.Module):
    def __init__(self, img_dim=512, txt_dim=512, hidden=512, num_classes=5):
        super().__init__()
        self.fc1 = nn.Linear(img_dim + txt_dim, hidden)
        self.act = nn.ReLU()
        self.fc2 = nn.Linear(hidden, num_classes)

    def forward(self, img_feat, txt_feat):
        # expect tensors: (batch, dim)
        if img_feat.dim() == 1:
            img_feat = img_feat.unsqueeze(0)
        if txt_feat.dim() == 1:
            txt_feat = txt_feat.unsqueeze(0)
        x = torch.cat([img_feat, txt_feat], dim=1)
        x = self.act(self.fc1(x))
        return self.fc2(x)
