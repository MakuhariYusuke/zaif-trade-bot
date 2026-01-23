import os

import numpy as np

eval_log_path = "eval_logs/evaluations.npz"

if os.path.exists(eval_log_path):
    data = np.load(eval_log_path, allow_pickle=True)
    print("Keys in npz:", data.files)

    # Assuming standard stable-baselines3 eval log structure
    if "results" in data:
        results = data["results"]
        timesteps = data["timesteps"]
        ep_lengths = data["ep_lengths"]

        print(f"\nLoaded {len(timesteps)} evaluation points.")

        for i, step in enumerate(timesteps):
            mean_reward = np.mean(results[i])
            std_reward = np.std(results[i])
            mean_len = np.mean(ep_lengths[i])
            print(
                f"Step {step}: Reward = {mean_reward:.4f} +/- {std_reward:.4f}, Length = {mean_len:.1f}"
            )

    else:
        print("Unknown structure:", data.files)
else:
    print(f"File not found: {eval_log_path}")
