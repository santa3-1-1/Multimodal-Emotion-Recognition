# tools/label_gui.py
import os
import pandas as pd
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from PIL import Image, ImageTk

# ======================
# ✅ 自动定位 CSV 路径（相对路径）
# ======================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CSV_PATH = os.path.join(BASE_DIR, "mvsa_dataset.csv")

# ======================
# ✅ 如果 CSV 文件不存在，提示用户选择
# ======================
if not os.path.exists(CSV_PATH):
    messagebox.showinfo("提示", "未检测到 mvsa_dataset.csv，请选择一个 CSV 文件")
    selected = filedialog.askopenfilename(
        title="请选择 mvsa_dataset.csv",
        filetypes=[("CSV 文件", "*.csv")]
    )
    if selected:
        CSV_PATH = selected
    else:
        messagebox.showerror("错误", "未选择任何 CSV 文件，程序退出")
        exit()

# ======================
# ✅ 读取 CSV 并初始化
# ======================
df = pd.read_csv(CSV_PATH)

# 如果没有 emotion 列，则自动添加
if 'emotion' not in df.columns:
    df['emotion'] = None

# 自动续标：找到第一个未标注的行
if df['emotion'].isna().any():
    index = df['emotion'].isna().idxmax()
else:
    index = len(df)

# ======================
# ✅ 保存 + 下一条
# ======================
def save_and_next(label):
    global index
    df.loc[index, 'emotion'] = label
    df.to_csv(CSV_PATH, index=False, encoding='utf-8-sig')
    index += 1
    show_sample()

# ======================
# ✅ 显示当前样本
# ======================
def show_sample():
    global index, img_label, text_label
    if index >= len(df):
        text_label.config(text="✅ 已完成所有标注！")
        img_label.config(image="")
        return

    row = df.iloc[index]
    path = row['image_path']
    caption = row['caption']

    text_label.config(text=f"[{index+1}/{len(df)}]\n{caption}")

    try:
        img = Image.open(path).resize((400, 400))
        tkimg = ImageTk.PhotoImage(img)
        img_label.img = tkimg
        img_label.config(image=tkimg, text="")
    except:
        img_label.config(image="", text=f"(无法加载图片 {path})")

# ======================
# ✅ GUI 界面
# ======================
root = tk.Tk()
root.title("🎨 MVSA 情绪标注工具")
root.geometry("600x550")

img_label = tk.Label(root)
img_label.pack(pady=10)

text_label = tk.Label(root, wraplength=500, justify="center", font=("Microsoft YaHei", 11))
text_label.pack(pady=10)

frame = tk.Frame(root)
frame.pack()

# 三种情绪按钮
for emotion in ["positive", "neutral", "negative"]:
    ttk.Button(frame, text=emotion, command=lambda e=emotion: save_and_next(e)).pack(side=tk.LEFT, padx=10)

# 启动
show_sample()
root.mainloop()
