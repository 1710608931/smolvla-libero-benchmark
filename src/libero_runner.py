from tqdm import tqdm
from LIBERO.libero.libero import benchmark



def run_libero(policy, benchmark_name="libero_spatial", episodes=20):

    benchmark_dict = benchmark.get_benchmark_dict()

    tasks = benchmark_dict[benchmark_name]()

    total_success = 0
    total_trials = 0

    for task_id in range(tasks.n_tasks): 
        task = tasks.get_task(task_id)
        task_name = tasks.get_task_names()[task_id]
        print(f"正在运行任务 {task_id}: {task_name}")

        env = tasks.get_env(task_id)

        instruction = tasks.get_task(task_id).language

        for ep in tqdm(range(episodes)):

            obs = env.reset()

            done = False

            while not done:

                action = policy.predict(obs, instruction)

                obs, reward, done, info = env.step(action)

            total_success += info["success"]
            total_trials += 1

    success_rate = total_success / total_trials

    return success_rate