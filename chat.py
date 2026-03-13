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

# ----------------- 初始化 SmolVLA -----------------
model_id = "lerobot/smolvla_base"
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

        img = F.interpolate(
            img,
            size=(256,256),
            mode="bilinear",
            align_corners=False
        )

        return img

    # state = np.concatenate([
    #     obs['robot0_proprio-state'],
    #     obs['object-state']
    # ])
    state = obs["robot0_proprio-state"]

    # if len(state) < max_state_dim:
    #     state = np.pad(state,(0,max_state_dim-len(state)))

    state = torch.from_numpy(state).unsqueeze(0).float().to(device)

    # 如果超过 max_state_dim -> 裁剪
    if state.shape[-1] > max_state_dim:
        state = state[:, :max_state_dim]

    # 如果不足 -> padding
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

    action = action.detach().cpu().numpy()

    # 去掉 batch / chunk
    while action.ndim > 1:
        action = action[0]

    # 如果是8维 → 去掉 terminate
    if action.shape[0] == 8:
        action = action[:7]

    # 如果是6维 → 补 gripper
    elif action.shape[0] == 6:
        action = np.concatenate([action, [0.0]])

    return action

# ----------------- 运行一个 episode -----------------
def run_episode(env, instruction, max_steps=200):
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

        action = process_action(action)

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

    base_path = os.path.expanduser("~/smallVLA/benchmark/LIBERO/libero/libero/bddl_files")
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