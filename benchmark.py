import os
import time
import torch
import numpy as np
from PIL import Image

from LIBERO.libero.libero import benchmark
from LIBERO.libero.libero.envs import OffScreenRenderEnv
from lerobot.policies.factory import make_pre_post_processors
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
import torch.nn.functional as F
from huggingface_hub import snapshot_download

# ----------------- 初始化 SmolVLA -----------------
os.environ["HF_HUB_URL"] = "https://hf-mirror.com"
# 指定缓存目录（可选）
cache_dir = "/home/wushi/smolvla/huggingface_cache"

# 下载模型到本地
local_model_path = snapshot_download(
    repo_id="lerobot/smolvla_base",
    cache_dir=cache_dir,
    local_files_only=False  # 如果已经缓存，可用 True
)

model_id = local_model_path
# model_id = "/home/wushi/.cache/huggingface/hub/models--HuggingFaceTB--SmolVLM2-500M-Video-Instruct/snapshots/7b375e1b73b11138ff12fe22c8f2822d8fe03467"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

policy = SmolVLAPolicy.from_pretrained(model_id).to(device).eval()
# print(11111111111111111111111111111111111111)
# print(policy.config.image_features)
# print(11111111111111111111111111111111111111)
# time.sleep(2)

preprocess, postprocess = make_pre_post_processors(
    policy.config,
    model_id,
    preprocessor_overrides={"device_processor": {"device": str(device)}}
)

max_state_dim = policy.config.max_state_dim  # 32
resize_h, resize_w = policy.config.resize_imgs_with_padding

# ----------------- frame preprocess -----------------
def prepare_frame_for_policy(obs, instruction, device=device):

    def hwc_to_bchw(img):
        img = np.transpose(img,(2,0,1))
        img = torch.from_numpy(img).unsqueeze(0).float().to(device)
        img = F.interpolate(img, size=(256,256), mode="bilinear", align_corners=False)
        return img

    # concat robot + object state
    state = np.concatenate([obs['robot0_proprio-state'], obs['object-state']])
    state = torch.from_numpy(state).unsqueeze(0).float().to(device)

    if state.shape[-1] > max_state_dim:
        state = state[:, :max_state_dim]
    elif state.shape[-1] < max_state_dim:
        pad = max_state_dim - state.shape[-1]
        state = torch.nn.functional.pad(state, (0, pad))

    img1 = hwc_to_bchw(obs['agentview_image'])
    img2 = hwc_to_bchw(obs['robot0_eye_in_hand_image'])
    img3 = hwc_to_bchw(obs['agentview_image'])

    transition = {

        "observation.state": state,

        "observation.images.camera1": img1,
        "observation.images.camera2": img2,
        "observation.images.camera3": img3,

        "task": instruction
    }

    processed = preprocess(transition)
    return processed

def process_action(action):
    """
    将 SmolVLA 输出动作处理成 LIBERO 环境可接受的 7 维动作
    - 前 6 维为末端位置/姿态
    - 第 7 维为抓手，固定闭合 (1.0)
    """
    action = action.detach().cpu().numpy()

    # 去掉 batch / chunk 维度
    while action.ndim > 1:
        action = action[0]

    # 保留前 6 维
    if action.shape[0] >= 6:
        action6 = action[:6]
    else:
        # 如果输出 <6维，用 0 填充
        pad_len = 6 - action.shape[0]
        action6 = np.concatenate([action, np.zeros(pad_len)])

    # 第 7 维固定闭合抓手
    gripper_val = 1.0  # 1.0 = fully closed, -1.0 = fully open
    action7 = np.array([gripper_val])

    # 拼成 7 维动作
    action7d = np.concatenate([action6, action7])

    return action7d

# ----------------- 运行一个 episode -----------------
def run_episode(env, instruction, max_steps=300):
    obs = env.reset()
    total_reward = 0
    latency_list = []

    for step in range(max_steps):
        frame = prepare_frame_for_policy(obs, instruction, device=device)

        # print(frame.keys())

        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()

        # policy 输出动作
        action = policy.select_action(frame)
        action = postprocess(action)
        # print("FINAL ACTION Numpy2:", action)
        # print("FINAL ACTION SHAPE2:", action.shape)
        # time.sleep(2)

        action = process_action(action)
        # print("FINAL ACTION Numpy3:", action)
        # print("FINAL ACTION SHAPE3:", action.shape)
        # time.sleep(2)

        end.record()
        torch.cuda.synchronize()
        latency_list.append(start.elapsed_time(end) / 1000)  # 秒

        obs, reward, done, info = env.step(action)
        total_reward += reward

        if done:
            break

    return total_reward, latency_list, step + 1

# ----------------- LIBERO_10 benchmark 循环 -----------------
benchmark_dict = benchmark.get_benchmark_dict()
task_suite = benchmark_dict["libero_10"]()

total_reward = 0
total_latency = []
total_steps = 0
success = 0

for task_id in range(len(task_suite.tasks)):
    task = task_suite.get_task(task_id)
    instruction = task.language

    base_path = os.path.expanduser("~/smolvla/benchmark/LIBERO/libero/libero/bddl_files")
    bddl_file = os.path.join(base_path, task.problem_folder, task.bddl_file)

    env = OffScreenRenderEnv(
        bddl_file_name=bddl_file,
        camera_heights=128,
        camera_widths=128
    )

    # 初始化状态
    env.set_init_state(task_suite.get_task_init_states(task_id)[0])

    reward, latency, steps = run_episode(env, instruction)

    total_reward += reward
    total_latency.extend(latency)
    total_steps += steps
    if reward > 0:
        success += 1

    print(f"Task {task_id} | Reward: {reward} | Steps: {steps}")

# ----------------- 输出指标 -----------------
success_rate = success / len(task_suite.tasks)
avg_reward = total_reward / len(task_suite.tasks)
avg_latency = np.mean(total_latency)
fps = total_steps / sum(total_latency)

print("\n========== RESULT ==========")
print("Success Rate:", success_rate)
print("Average Reward:", avg_reward)
print("Average Latency (s):", avg_latency)
print("FPS:", fps)