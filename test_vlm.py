import argparse
import time
import numpy as np
import torch
from PIL import Image
import os

from transformers import AutoProcessor, AutoModelForVision2Seq, BitsAndBytesConfig

from libero.libero import benchmark
from libero.libero.envs import OffScreenRenderEnv



def load_model(model_name, quant):

    if quant == "fp16":

        model = AutoModelForVision2Seq.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            device_map="auto"
        )

    elif quant == "int8":

        quant_config = BitsAndBytesConfig(
            load_in_8bit=True
        )

        model = AutoModelForVision2Seq.from_pretrained(
            model_name,
            quantization_config=quant_config,
            device_map="auto"
        )

    elif quant == "int4":

        quant_config = BitsAndBytesConfig(
            load_in_4bit=True
        )

        model = AutoModelForVision2Seq.from_pretrained(
            model_name,
            quantization_config=quant_config,
            device_map="auto"
        )

    else:

        model = AutoModelForVision2Seq.from_pretrained(
            model_name,
            device_map="auto"
        )

    processor = AutoProcessor.from_pretrained(model_name)

    return model, processor


def parse_action(text):

    try:

        nums = [float(x) for x in text.replace(",", " ").split()]

        if len(nums) >= 7:

            return np.array(nums[:7])

    except:

        pass

    return np.zeros(7)


def run_episode(env, model, processor, instruction, max_steps=200):

    obs = env.reset()

    total_reward = 0
    latency_list = []

    for step in range(max_steps):

        image = obs["agentview_image"]

        image = Image.fromarray(image)

        prompt = f"<image>\n{instruction}"
        # 打印 prompt，方便调试
        # print(f"[Step {step}] Prompt:", repr(prompt))
        # time.sleep(1)

        start = time.time()

        inputs = processor(
            images=image,
            text=prompt,
            return_tensors="pt"
        ).to(model.device)
        # print(inputs)

        outputs = model.generate(**inputs, max_new_tokens=20)
        
        # action_text = processor.decode(outputs[0])
        action_text = processor.decode(outputs[0], skip_special_tokens=True)
        # print(action_text)
        # time.sleep(1)
        # —— 打印每步动作 —— #
        # print(f"[Step {step}] Action: {action_text}")
        # time.sleep(1)

        latency = time.time() - start

        latency_list.append(latency)

        action = parse_action(action_text)

        # print(f"[Step {step}] Parsed Action:", action)

        obs, reward, done, info = env.step(action)
        # print(f"[Step {step}] Reward:", reward)
        # print(f"[Step {step}] Done:", done)
        # print(f"[Step {step}] Info:", info)
        # time.sleep(1)
        
        total_reward += reward

        if done:

            return True, total_reward, latency_list, step + 1

    return False, total_reward, latency_list, max_steps


def create_env(task):

    base_path = os.path.expanduser(
        "~/smallVLA/benchmark/LIBERO/libero/libero/bddl_files"
    )

    task_suite = task.problem_folder

    bddl_file = os.path.join(
        base_path,
        task_suite,
        task.bddl_file
    )

    env = OffScreenRenderEnv(
        bddl_file_name=bddl_file,
        camera_heights=128,
        camera_widths=128
    )

    return env


def main():  

    parser = argparse.ArgumentParser()

    parser.add_argument("--model", default="HuggingFaceTB/SmolVLM-256M-Instruct")
    parser.add_argument("--quant", default="fp16")

    args = parser.parse_args()

    print("Loading model...")

    model, processor = load_model(args.model, args.quant)

    benchmark_dict = benchmark.get_benchmark_dict()

    task_suite = benchmark_dict["libero_10"]()

    success = 0
    total_reward = 0
    total_latency = []
    total_steps = 0

    print("Running LIBERO_10 benchmark...")

    for task_id in range(len(task_suite.tasks)):

        task = task_suite.get_task(task_id)

        instruction = task.language

        env = create_env(task)

        init_states = task_suite.get_task_init_states(task_id)

        env.reset()

        env.set_init_state(init_states[0])

        result, reward, latency, steps = run_episode(
            env,
            model,
            processor,
            instruction,
        )

        total_reward += reward
        total_latency.extend(latency)
        total_steps += steps

        if result:

            success += 1

        print(
            f"Task {task_id} | Success: {result} | Reward: {reward} | Steps: {steps}"
        )

    success_rate = success / len(task_suite.tasks)

    avg_reward = total_reward / len(task_suite.tasks)

    avg_latency = np.mean(total_latency)

    fps = total_steps / sum(total_latency)

    print("\n========== RESULT ==========")

    print("Quantization:", args.quant)

    print("Success Rate:", success_rate)

    print("Average Reward:", avg_reward)

    print("Average Latency:", avg_latency)

    print("FPS:", fps)


if __name__ == "__main__":

    main()