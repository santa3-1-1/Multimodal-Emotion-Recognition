import os
import re
import torch
import torch.nn.functional as F
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from textwrap import fill
from transformers import (
    CLIPProcessor, CLIPModel,
    AutoTokenizer, AutoModelForSequenceClassification,
    MarianMTModel, MarianTokenizer
)
from models.fusion_head import FusionHead
from models.clip_encoder import CLIPEncoder
from utils.label_map import map_label_soft

# ✅ 防止服务器无显示时报错
matplotlib.use('Agg')
matplotlib.rcParams['font.sans-serif'] = ['SimHei']
matplotlib.rcParams['axes.unicode_minus'] = False


# -----------------------------------------------------
# 🔍 检测语言
# -----------------------------------------------------
def detect_language(text: str) -> str:
    chinese_chars = len(re.findall(r'[\u4e00-\u9fff]', text))
    total_chars = len(text)
    ratio = chinese_chars / total_chars if total_chars > 0 else 0
    return "chinese" if ratio > 0.3 else "english"


# -----------------------------------------------------
# 🌐 中文翻译（优先用 opus-mt-zh-en 完整翻译）
# -----------------------------------------------------
def translate_chinese_to_english(text):
    model_path = "/home/data/xiaoyu/models/opus-mt-zh-en"

    if os.path.exists(model_path):
        try:
            print("🌐 检测到本地翻译模型，使用 opus-mt-zh-en 完整翻译...")
            tokenizer = MarianTokenizer.from_pretrained(model_path, local_files_only=True)
            model = MarianMTModel.from_pretrained(model_path, local_files_only=True)
            inputs = tokenizer(text, return_tensors="pt", truncation=True)
            translated = model.generate(**inputs, max_length=256)
            result = tokenizer.decode(translated[0], skip_special_tokens=True)
            return result
        except Exception as e:
            print(f"⚠️ 翻译模型加载失败，使用关键词替换模式。原因: {e}")

    # 🚑 回退模式：关键词替换
    translations = {
        '快乐': 'happy', '高兴': 'happy', '开心': 'happy',
        '悲伤': 'sad', '难过': 'sad', '伤心': 'sad',
        '生气': 'angry', '愤怒': 'angry', '恼火': 'angry',
        '平静': 'calm', '安宁': 'calm', '平和': 'calm',
        '焦虑': 'anxious', '紧张': 'anxious', '担心': 'worried',
        '漂亮': 'beautiful', '美丽': 'beautiful',
        '谢谢': 'thank you', '感谢': 'thanks',
        '心情': 'mood', '情绪': 'emotion',
        '治愈': 'healing', '安慰': 'comfort'
    }
    translated = text
    for cn, en in translations.items():
        translated = translated.replace(cn, en)
    return translated


# -----------------------------------------------------
# 🧠 主预测函数
# -----------------------------------------------------
def predict(image_path, text, device='cpu', use_trainable_fusion=True,
            fusion_checkpoint='/home/data/xiaoyu/CLIPMulti_reduced_project/checkpoints/fusion_head.pt'):
    device = torch.device(device)
    print(f"🖥️ 当前运行设备：{device}")

    # ✅ 使用本地 CLIP 模型
    clip_model_path = '/home/data/xiaoyu/models/clip-vit-base-patch16'
    clip_model = CLIPModel.from_pretrained(clip_model_path, local_files_only=True).to(device)
    clip_processor = CLIPProcessor.from_pretrained(clip_model_path, local_files_only=True)

    # ✅ 检测语言并翻译
    lang = detect_language(text)
    print(f"🌐 检测到输入语言：{lang}")
    original_text = text
    clip_text = translate_chinese_to_english(text) if lang == "chinese" else text
    if lang == "chinese":
        print(f"🔤 中文翻译为英文: {text} -> {clip_text}")

    # ✅ 本地英文情感分析模型（手动推理）
    print("✅ 使用本地英文情感分析模型（手动推理）")
    sentiment_model_path = "/home/data/xiaoyu/models/textattack-distilbert-base-uncased-imdb"
    tokenizer = AutoTokenizer.from_pretrained(sentiment_model_path, local_files_only=True)
    sent_model = AutoModelForSequenceClassification.from_pretrained(sentiment_model_path, local_files_only=True).to(device)
    sent_model.eval()

    inputs_for_sent = tokenizer(clip_text, return_tensors='pt', truncation=True, padding=True)
    inputs_for_sent = {k: v.to(device) for k, v in inputs_for_sent.items()}

    with torch.no_grad():
        outputs_sent = sent_model(**inputs_for_sent)
        probs_sent = F.softmax(outputs_sent.logits, dim=-1).cpu().squeeze(0)

    if hasattr(sent_model.config, 'id2label') and isinstance(sent_model.config.id2label, dict):
        id2label = sent_model.config.id2label
    else:
        id2label = {0: 'NEGATIVE', 1: 'POSITIVE'}

    label_id = int(torch.argmax(probs_sent).item())
    text_label = id2label.get(label_id, str(label_id))
    score = float(probs_sent[label_id].item())
    text_result_raw = {'label': text_label, 'score': score}

    # ✅ 文本 soft 分布
    soft_dist = map_label_soft(text_label)
    text_pseudo = torch.zeros(5)
    for k, v in soft_dist.items():
        text_pseudo[k] = v * score
    text_pseudo = text_pseudo / text_pseudo.sum()

    # ✅ 图像特征预测
    image = Image.open(image_path).convert('RGB')
    candidate_texts = [
        'a happy bright sunny scene',
        'a sad gloomy scene',
        'an angry violent scene',
        'a calm peaceful scene',
        'an anxious tense scene'
    ]
    inputs = clip_processor(text=candidate_texts, images=image, return_tensors='pt', padding=True).to(device)

    with torch.no_grad():
        outputs = clip_model(**inputs)
        logits_per_image = outputs.logits_per_image
        probs = F.softmax(logits_per_image, dim=1).cpu().squeeze(0)

    # ✅ 融合层
    if use_trainable_fusion and os.path.exists(fusion_checkpoint):
        print(f"🧠 使用训练好的融合层：{fusion_checkpoint}")
        clip_encoder = CLIPEncoder(model_name=clip_model_path, device=device)
        fusion = FusionHead(img_dim=512, txt_dim=512, hidden=512, num_classes=5).to(device)
        fusion.load_state_dict(torch.load(fusion_checkpoint, map_location=device))
        fusion.eval()

        # ✅ 把 PIL 图像先转为 CLIP 预处理后的 tensor
        # ✅ 直接传入原始 PIL 图像，让 clip_encoder 自己处理
        img_feat = clip_encoder.encode_image(image).to(device)
        txt_feat = clip_encoder.encode_text([clip_text]).to(device)

        txt_feat = clip_encoder.encode_text([clip_text]).to(device)

        with torch.no_grad():
            logits = fusion(img_feat, txt_feat)
            combined = F.softmax(logits, dim=1).cpu().squeeze(0)
        method = 'trained_fusion_head'
    else:
        print("⚙️ 未检测到融合层或未启用训练融合，使用规则融合。")
        combined = 0.6 * probs + 0.4 * text_pseudo
        combined = combined / combined.sum()
        method = 'rule_based_softmap'

    # ✅ 输出结果
    comb_top = int(torch.argmax(combined).item())
    fused_result = {
        'method': method,
        'scores': {t: float(s) for t, s in zip(candidate_texts, combined.tolist())},
        'top': candidate_texts[comb_top],
        'original_text': original_text,
        'translated_text': clip_text if lang == "chinese" else None
    }

    visualize_results(image, original_text, fused_result)
    return fused_result

# -----------------------------------------------------
# 📊 可视化（仅英文，不显示中文）
# -----------------------------------------------------
def visualize_results(image, input_text, fused_result):
    scores = fused_result["scores"]
    labels = list(scores.keys())
    values = list(scores.values())
    top_label = fused_result["top"]

    plt.figure(figsize=(12, 6))

    # ---- 左侧显示图片 + 英文文本 ----
    plt.subplot(1, 2, 1)
    plt.imshow(image)
    plt.axis("off")

    # 如果有翻译，就显示翻译；否则显示原文本
    caption = fused_result.get("translated_text") or fused_result["original_text"]
    # 用自动换行处理长文本
    caption_wrapped = fill(caption, width=40)
    plt.text(0.5, 1.05, caption_wrapped,
             transform=plt.gca().transAxes,
             fontsize=12, ha='center', va='bottom',
             wrap=True, style='italic', color='#333')

    # ---- 右侧显示预测柱状图 ----
    plt.subplot(1, 2, 2)
    bars = plt.barh(labels, values)
    for bar, label in zip(bars, labels):
        if label == top_label:
            bar.set_color('orange')
    plt.xlabel("Probability", fontsize=10)
    plt.title(f'Predicted Emotion: {top_label}', fontsize=12)
    plt.tight_layout()
    plt.savefig("result_visual.png", bbox_inches='tight')
    print("✅ Visualization saved as result_visual.png (English only)")


# -----------------------------------------------------
# 🏁 CLI 入口
# -----------------------------------------------------
def predict_from_args(args):
    predict(
        image_path=args.image_path,
        text=args.text,
        device=args.device,
        use_trainable_fusion=True,
        fusion_checkpoint='/home/data/xiaoyu/CLIPMulti_reduced_project/CLIPMulti_reduced/checkpoints/fusion_head.pt'
    )
