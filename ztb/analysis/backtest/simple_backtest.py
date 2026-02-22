#!/usr/bin/env python3

"""
Simple Backtest Script for PPO Model
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

import numpy as np

from ztb.trading.environment.environment import HeavyTradingEnv
from ztb.training.policy_utils import predict_with_masks
from ztb.io.data_loader import DataLoader
from ztb.metrics.metrics import action_distribution as calculate_action_distribution


def run_simple_backtest(
    model_path="models/progress_bar_test.zip",
    data_path="ml-dataset-enhanced.csv",
    episodes=5,
):
    """Run a simple backtest"""

    print(f"Loading model from {model_path}")

    # Load data
    df = DataLoader.load_csv_optimized(data_path)
    print(f"Loaded {len(df)} rows of data")

    # Create environment
    config = {
        "reward_scaling": 0.01,  # Optimized value
        "transaction_cost": 0.00505,  # Optimized value
        "max_position_size": 1.05,  # Optimized value
        "risk_free_rate": 0.05,  # Optimized value
    }

    env = HeavyTradingEnv(df=df, config=config)

    # Load model
    try:
        # Try loading with MaskablePPO
        from sb3_contrib import MaskablePPO

        model = MaskablePPO.load(model_path)
        print("Model loaded successfully with MaskablePPO")
    except Exception as e:
        print(f"Error loading model: {e}")
        return None

    total_reward = 0.0
    total_pnl = 0.0
    all_actions = []
    total_rewards = []
    total_pnls = []

    for episode in range(episodes):
        obs, info = env.reset()
        done = False
        steps = 0
        episode_reward = 0.0
        episode_pnl = 0.0

        while not done:
            action, _states = predict_with_masks(model, obs, env, deterministic=True)
            if isinstance(action, np.ndarray):
                action = action.item()
            obs, reward, done, truncated, info = env.step(action)

            episode_reward += reward
            episode_pnl += info.get("pnl", 0.0)

            # Collect actions
            all_actions.append(action)

            steps += 1

        total_reward += episode_reward
        total_pnl += episode_pnl
        total_rewards.append(episode_reward)
        total_pnls.append(episode_pnl)

        print(
            f"Episode {episode + 1}: Reward={episode_reward:.2f}, PnL={episode_pnl:.6f}, Steps={steps}"
        )

    avg_reward = total_reward / episodes
    avg_pnl = total_pnl / episodes

    # Calculate statistics
    print("\n=== Backtest Results ===")
    avg_reward = np.mean(total_rewards)
    avg_pnl = np.mean(total_pnls)

    print(f"Episodes: {episodes}")
    std_reward = np.std(total_rewards)
    std_pnl = np.std(total_pnls)

    print(f"Average Reward: {avg_reward:.4f} ± {std_reward:.4f}")
    print(f"Average PnL: {avg_pnl:.6f} ± {std_pnl:.6f}")

    action_distribution = calculate_action_distribution(all_actions)
    total_actions = len(all_actions)
    print(f"Total Return: {sum(total_pnls):.6f}")

    print("\nAction Distribution:")
    for action_name, ratio in action_distribution.items():
        count = int(ratio * total_actions)
        percentage = ratio * 100
        print(f"  {action_name}: {count} ({percentage:.1f}%)")

    return {
        "avg_reward": avg_reward,
        "avg_pnl": avg_pnl,
        "total_pnl": total_pnl,
        "action_distribution": action_distribution,
    }


if __name__ == "__main__":
    model_path = "ztb/training/models/scalping_15s_balance_quick_test_final.zip"
    data_path = "ml-dataset-enhanced.csv"

    results = run_simple_backtest(model_path, data_path, episodes=5)
    if results:
        print("\nBacktest completed successfully!")
    else:
        print("\nBacktest failed!")
