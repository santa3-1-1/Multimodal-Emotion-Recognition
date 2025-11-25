# run_predict.py
# 在服务器上直接运行即可，不需要输入命令行参数

import sys
from main import main

# 修改为服务器上的路径 👇
image_path = "/home/data/xiaoyu/CLIPMulti_reduced_project/CLIPMulti_reduced/test_img/test.jpg"
text_input = """在大沙河跑步被一个女生叫住，
午间，在大沙河跑步时，
一个小妹妹骑单车停到我面前跟我说：“你好漂亮呀！”
我说：“谢谢☺️，有你的夸夸我今天心情都会很棒！”
又是被女生治愈的一天～"""

# 设置命令行参数
sys.argv = [
    "main.py",
    "--mode", "predict",
    "--image_path", image_path,
    "--text", text_input,
    "--device", "cuda"  # 使用GPU
]

# 调用主程序
if __name__ == "__main__":
    main()