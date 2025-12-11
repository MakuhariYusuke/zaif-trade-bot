import json
import os

file_path = r"c:\Users\Admin\dev\zaif-trade-bot\config\v446\sac_v446_multitimeframe_shortterm_optimized.json"

if not os.path.exists(file_path):
    print(f"File not found: {file_path}")
    exit(1)

with open(file_path, "r", encoding="utf-8") as f:
    data = json.load(f)

if "training" in data and "environment" in data["training"]:
    print("Moving environment from training to top level")
    env_config = data["training"].pop("environment")
    data["environment"] = env_config

    # Ensure max_action_threshold is set correctly
    data["environment"]["max_action_threshold"] = 1.0

    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    print("Done")
else:
    print("Environment not found in training or already at top level")
    if "environment" in data:
        print("Environment is at top level")
        # Ensure max_action_threshold is set correctly
        data["environment"]["max_action_threshold"] = 1.0
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
