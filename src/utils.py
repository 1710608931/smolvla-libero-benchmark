import numpy as np


def decode_action(action_text):

    action_text = action_text.lower()

    if "left" in action_text:
        return np.array([0, 0.3, 0, 0])

    if "right" in action_text:
        return np.array([0, -0.3, 0, 0])

    if "forward" in action_text:
        return np.array([0.3, 0, 0, 0])

    return np.zeros(4)
# import numpy as np
# import re

# def decode_action(action_text, action_dim=7):
#     """
#     针对 LIBERO 的连续动作解码
#     action_text 示例: "Action: 0.1, -0.2, 0.5, 0.0, 0.0, 0.0, 1.0"
#     """
#     action_text = action_text.lower()
    
#     # 尝试提取文本中的数字
#     numbers = re.findall(r"[-+]?\d*\.\d+|\d+", action_text)
    
#     if len(numbers) >= action_dim:
#         return np.array([float(n) for n in numbers[:action_dim]], dtype=np.float32)
    
#     # 如果模型确实只输出方向词（逻辑回退）
#     action = np.zeros(action_dim, dtype=np.float32)
#     if "left" in action_text:
#         action[1] = 0.3  # 假设 index 1 是左右
#     elif "forward" in action_text:
#         action[0] = 0.3  # 假设 index 0 是前后
    
#     # 注意：LIBERO 的夹持器状态通常在最后一位
#     if "open" in action_text: action[-1] = 1.0
#     if "close" in action_text: action[-1] = -1.0
    
#     return action