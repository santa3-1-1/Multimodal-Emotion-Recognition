# multi_modal_agent/agent_entry.py
"""
Coze workflow 节点入口。Coze 会调用 handler(request)。
要求：request 是一个 dict，包含 "image_path" 和 "text" 字段。

示例 request:
{
  "image_path": "/path/to/img.jpg",
  "text": "I feel good!"
}

handler 返回 predict 的结果（JSON-serializable dict）。
"""
from typing import Dict, Any

from .inference import predict


def handler(request: Dict[str, Any]) -> Dict[str, Any]:
    """
    Coze workflow 节点默认入口函数。
    只做：接收 json -> 调用 predict -> 返回结果
    """
    if not isinstance(request, dict):
        return {"error": "Invalid request format, expected JSON object/dict."}

    image_path = request.get("image_path")
    text = request.get("text", "")

    if not image_path:
        return {"error": "Missing 'image_path' in request."}

    try:
        result = predict(image_path, text)
        return result
    except Exception as e:
        return {"error": f"Exception during prediction: {str(e)}"}
