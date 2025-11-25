# multi_modal_agent

多模态情绪识别 Workflow 节点（用于 Coze）

## 目标
这个节点接收 JSON 输入：
## json
{
  "image_path": "xxx.jpg",
  "text": "I feel good!"
}
返回：
## json
复制代码
{
  "scores": {"happy": 0.7, "sad": 0.1, "angry": 0.05, "calm": 0.1, "anxious": 0.05},
  "top": "happy"
}
## 结构
markdown
multi_modal_agent/
├── models/
│   ├── clip_model.py
│   ├── text_model.py
│   └── __init__.py
├── utils/
│   ├── label_map.py
│   ├── language_detect.py
│   └── __init__.py
├── fusion.py
├── inference.py
├── agent_entry.py
├── requirements.txt
└── README.md
使用
安装依赖：

## bash
复制代码
pip install -r requirements.txt
在 Coze 中上传此项目并将 agent_entry.handler 配置为节点入口。

示例本地测试：

python
复制代码
from multi_modal_agent.agent_entry import handler

req = {"image_path": "/absolute/path/to/image.jpg", "text": "I feel amazing today!"}
print(handler(req))
## 注意：

请确保 image_path 是 Coze 执行环境中可访问的绝对路径或相对路径（相对于运行目录）。

transformers 会在第一次运行时下载模型，请确保环境能联网，或提前把模型缓存到目标环境。

设计说明
图像情绪：使用 CLIP (openai/clip-vit-base-patch16) 对候选情绪标签做对比并 softmax 得到概率。

文本情绪：使用 transformers pipeline("sentiment-analysis")，将 label+score 映射为对候选标签的软分布（由 utils/label_map.py 控制）。

融合策略：final = 0.6 * clip + 0.4 * text（可在 fusion.py 调整）。

---

### 重要注意事项与使用提示
1. 所有路径均为标准导入（包内导入）。在 Coze 中请确保工作目录包含 `multi_modal_agent` 文件夹或将其打包上传为项目根。
2. `agent_entry.handler` 是默认入口，返回值是 JSON-serializable 的 `dict`。
3. 模型首次下载需要网络，若 Coze 环境无法联网，请在本地预下载 transformers 模型并导出到 Coze 环境的模型缓存目录（通常是 `~/.cache/huggingface`）。
4. 如果希望把候选情绪（DEFAULT_CANDIDATES）增加或修改，请同时修改 `utils/label_map.py` 中的 `DEFAULT_CANDIDATES` 与 `LABEL_MAP` 中的映射顺序和值，确保一致性。
5. 你可以根据需要修改 `fusion.DEFAULT_CLIP_WEIGHT` 和 `DEFAULT_TEXT_WEIGHT` 的默认值。

---

如果你希望我把这些文件直接打包成一个 ZIP（我这里无法直接创建上传文件，但可以给