#!/usr/bin/env python3
"""
Simple SAC v431 Training Test Script
1000ステップの基本トレーニングを実行
"""

import json
import sys
from pathlib import Path

import numpy as np

from ztb.utils.file_utils import get_project_root

# Add project root to path
project_root = get_project_root()
sys.path.insert(0, str(project_root))


def test_v431_training():
    """Test v431 training with 1000 steps"""

    # Load v431 config
    config_path = Path("../../../configs/v431/sac_v431_advanced.json")
    with open(config_path, "r") as f:
        config = json.load(f)

    print("=== SAC v431 Training Test (1000 steps) ===")
    print(f"Version: {config['version']}")
    print(f"Reward Function: {config['reward_function']}")

    # Simulate training loop (simplified)
    print("\n=== Simulated Training Loop ===")

    total_steps = 1000
    actions_taken = {"BUY": 0, "SELL": 0, "HOLD": 0}
    total_reward = 0

    # Reward parameters
    sell_bonus = config["reward_function"]["sell_bonus"]
    hold_bonus = config["reward_function"]["hold_bonus"]
    buy_bonus = config["reward_function"]["buy_bonus"]

    sell_threshold = config["action_thresholds"]["sell_threshold"]
    buy_threshold = config["action_thresholds"]["buy_threshold"]

    print(f"Sell Threshold: {sell_threshold}, Buy Threshold: {buy_threshold}")
    print(f"Sell Bonus: {sell_bonus}, Hold Bonus: {hold_bonus}, Buy Bonus: {buy_bonus}")

    # Simulate 1000 steps with random actions
    np.random.seed(42)  # For reproducibility

    for step in range(total_steps):
        # Generate random action (simplified SAC output)
        action_value = np.random.normal(0, 0.5)  # Normal distribution around 0

        # Determine action type and reward
        if action_value <= sell_threshold:
            action_type = "SELL"
            reward = sell_bonus
        elif action_value >= buy_threshold:
            action_type = "BUY"
            reward = buy_bonus
        else:
            action_type = "HOLD"
            reward = hold_bonus

        actions_taken[action_type] += 1
        total_reward += reward

        if step % 100 == 0:
            print(
                f"Step {step}: Action = {action_value:.3f} ({action_type}), Reward = {reward:.4f}"
            )

    # Results
    print("\n=== Training Results ===")
    print(f"Total Steps: {total_steps}")
    print(f"Total Reward: {total_reward:.4f}")
    print(f"Average Reward per Step: {total_reward/total_steps:.4f}")

    print("\n=== Action Distribution ===")
    for action, count in actions_taken.items():
        percentage = (count / total_steps) * 100
        print(f"{action}: {count} ({percentage:.1f}%)")

    # Check balance
    buy_pct = (actions_taken["BUY"] / total_steps) * 100
    sell_pct = (actions_taken["SELL"] / total_steps) * 100
    hold_pct = (actions_taken["HOLD"] / total_steps) * 100

    print("\n=== Balance Analysis ===")
    if abs(buy_pct - sell_pct) < 10 and hold_pct < 50:
        print("✅ Actions are well balanced!")
    else:
        print("⚠️  Actions may need further balancing")

    print("\n=== Next Steps ===")
    print("1. Run full curriculum learning (20000+ timesteps)")
    print("2. Execute ensemble training with multiple models")
    print("3. Perform multi-stage training (exploration → exploitation → fine-tuning)")


if __name__ == "__main__":
    test_v431_training()
