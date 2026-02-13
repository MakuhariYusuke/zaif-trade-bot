import json
from pathlib import Path

# Load the training summary
summary_path = Path("results/sac_v445.3_strong_selling_optimized_training_summary.json")
if summary_path.exists():
    with open(summary_path, "r") as f:
        summary = json.load(f)
    print("=== Training Summary ===")
    for key, value in summary.items():
        print(f"{key}: {value}")

    # Check for evaluation results
    eval_files = list(Path("results").glob("*evaluation*"))
    if eval_files:
        print("\n=== Evaluation Results ===")
        for eval_file in eval_files:
            if "sac_v445.3" in str(eval_file):
                with open(eval_file, "r") as f:
                    eval_data = json.load(f)
                print(f"File: {eval_file.name}")
                if "mean_reward" in eval_data:
                    print(f'Mean Reward: {eval_data["mean_reward"]:.2f}')
                if "std_reward" in eval_data:
                    print(f'Std Reward: {eval_data["std_reward"]:.2f}')
                print()

    # Check TensorBoard logs
    tensorboard_path = Path("tensorboard/sac_v445.3_strong_selling_optimized")
    if tensorboard_path.exists():
        print("=== TensorBoard Logs ===")
        print(f"TensorBoard path: {tensorboard_path}")
        ppo_dirs = list(tensorboard_path.glob("PPO_*"))
        if ppo_dirs:
            print(f"Found {len(ppo_dirs)} PPO training runs")
            for ppo_dir in sorted(ppo_dirs):
                print(f"  - {ppo_dir.name}")

print("\n=== Analysis Complete ===")
