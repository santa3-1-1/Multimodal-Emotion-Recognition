import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from torchvision import transforms
import pandas as pd
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

from models.fusion_head import FusionHead
from models.clip_encoder import CLIPEncoder
from utils.dataset import ImageTextDataset
import argparse


# ------------------------------------------------------
# 🔍 验证阶段函数（KL散度 + Cosine相似度）
# ------------------------------------------------------
def evaluate_model(fusion, clip, val_dl, device):
    fusion.eval()
    clip.model.eval()

    total_kl = 0.0
    total_cos = 0.0
    n_batches = 0

    with torch.no_grad():
        for imgs, texts, soft_labels in val_dl:
            imgs = imgs.to(device)
            soft_labels = soft_labels.to(device)

            # 提取特征（无梯度）
            img_emb = clip.encode_image(imgs)
            txt_emb = clip.encode_text(texts)
            img_emb = F.normalize(img_emb, dim=-1)
            txt_emb = F.normalize(txt_emb, dim=-1)

            logits = fusion(img_emb, txt_emb)
            log_probs = F.log_softmax(logits, dim=1)

            # KL 散度评估
            kl = F.kl_div(log_probs, soft_labels, reduction='batchmean')

            # 平均余弦相似度评估（反映图文语义对齐程度）
            cos_sim = F.cosine_similarity(img_emb, txt_emb, dim=-1).mean()

            total_kl += kl.item()
            total_cos += cos_sim.item()
            n_batches += 1

    avg_kl = total_kl / n_batches
    avg_cos = total_cos / n_batches
    return avg_kl, avg_cos


# ------------------------------------------------------
# 🧠 主训练函数
# ------------------------------------------------------
def train_from_args(args):
    device = torch.device(args.device)
    print(f"🧠 当前设备：{device}")

    clip = CLIPEncoder(model_name="/home/data/xiaoyu/models/clip-vit-base-patch16", device=args.device)

    # 读取 CSV 数据
    csv_path = args.data_csv
    if not os.path.isabs(csv_path):
        csv_path = os.path.join(os.path.dirname(__file__), csv_path)
    df = pd.read_csv(csv_path)
    print(f"📄 样本总数：{len(df)}")

    temp_csv = "temp_subset.csv"
    df.to_csv(temp_csv, index=False)

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])
    full_dataset = ImageTextDataset(temp_csv, transform=transform, soft_label=True)

    # ✅ 随机划分训练/验证集（避免顺序泄漏）
    indices = list(range(len(full_dataset)))
    train_idx, val_idx = train_test_split(indices, test_size=0.15, random_state=42, shuffle=True)
    train_ds = Subset(full_dataset, train_idx)
    val_ds = Subset(full_dataset, val_idx)
    print(f"📊 训练集: {len(train_ds)} | 验证集: {len(val_ds)}")

    train_dl = DataLoader(train_ds, batch_size=64, shuffle=True)
    val_dl = DataLoader(val_ds, batch_size=64)

    # ✅ 初始化融合层与优化器
    num_classes = 5
    fusion = FusionHead(img_dim=512, txt_dim=512, hidden=512, num_classes=num_classes).to(device)
    optimizer = torch.optim.AdamW(fusion.parameters(), lr=5e-5, weight_decay=1e-4)
    kl_loss_fn = nn.KLDivLoss(reduction='batchmean')

    epochs = 10
    best_val_score = float('inf')  # KL 越小越好

    # 📈 曲线记录
    train_losses, val_kls, val_coss = [], [], []

    for epoch in range(epochs):
        fusion.train()
        clip.model.eval()  # 冻结 CLIP
        total_loss = 0.0
        n_batches = 0

        for imgs, texts, soft_labels in train_dl:
            imgs = imgs.to(device)
            soft_labels = soft_labels.to(device)

            with torch.no_grad():
                img_emb = clip.encode_image(imgs)
                txt_emb = clip.encode_text(texts)
                img_emb = F.normalize(img_emb, dim=-1)
                txt_emb = F.normalize(txt_emb, dim=-1)

            logits = fusion(img_emb, txt_emb)
            log_probs = F.log_softmax(logits, dim=1)

            # KL loss + 对齐损失
            loss_kl = kl_loss_fn(log_probs, soft_labels)
            cos_sim = F.cosine_similarity(img_emb, txt_emb, dim=-1).mean()
            loss_cos = 1 - cos_sim
            loss = loss_kl + 0.1 * loss_cos

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            n_batches += 1

        avg_train_loss = total_loss / n_batches
        val_kl, val_cos = evaluate_model(fusion, clip, val_dl, device)

        train_losses.append(avg_train_loss)
        val_kls.append(val_kl)
        val_coss.append(val_cos)

        print(f"📘 Epoch {epoch+1}/{epochs} | 训练Loss={avg_train_loss:.4f} | 验证KL={val_kl:.4f} | 验证Cos={val_cos:.3f}")

        # ✅ 保存最佳（最小 KL）
        if val_kl < best_val_score:
            best_val_score = val_kl
            os.makedirs(args.output_dir, exist_ok=True)
            save_path = os.path.join(args.output_dir, 'fusion_head.pt')
            torch.save(fusion.state_dict(), save_path)
            print(f"💾 保存最佳模型：验证KL={val_kl:.4f}, 平均对齐Cos={val_cos:.3f}")

    # ✅ 绘制曲线
    os.makedirs(args.output_dir, exist_ok=True)
    plt.figure(figsize=(10, 5))

    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label="Train Loss")
    plt.plot(val_kls, label="Val KL")
    plt.legend()
    plt.title("训练损失 / 验证 KL")

    plt.subplot(1, 2, 2)
    plt.plot(val_coss, label="Val Cosine Similarity", color="orange")
    plt.legend()
    plt.title("验证集平均余弦相似度")

    plt.tight_layout()
    plot_path = os.path.join(args.output_dir, "training_curve.png")
    plt.savefig(plot_path)
    print(f"📊 训练曲线已保存为 {plot_path}")
    print(f"✅ 训练结束，最优验证KL={best_val_score:.4f}")


# ------------------------------------------------------
# 🚀 CLI 入口
# ------------------------------------------------------
if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    p.add_argument('--data_csv', required=True)
    p.add_argument('--output_dir', default='checkpoints')
    args = p.parse_args()
    train_from_args(args)
