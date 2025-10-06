content = '''#!/usr/bin/env python3

"""
Simple Backtest Script for PPO Model
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

import numpy as np
import pandas as pd
from stable_baselines3 import PPO
from ztb.trading.environment.environment import HeavyTradingEnv

def run_simple_backtest(model_path="models/progress_bar_test.zip", data_path="ml-dataset-enhanced.csv", episodes=5):
    """Run a simple backtest"""

    print(f"Loading model from {model_path}")

    # Load data
    df = pd.read_csv(data_path)
    print(f"Loaded {len(df)} rows of data")

    # Create environment
    config = {
        "reward_scaling": 0.01,       # Optimized value
        "transaction_cost": 0.00505,  # Optimized value
        "max_position_size": 1.05,    # Optimized value
        "risk_free_rate": 0.05,       # Optimized value
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
    action_counts = {"HOLD": 0, "BUY": 0, "SELL": 0}
    total_rewards = []
    total_pnls = []

    for episode in range(episodes):
        obs, info = env.reset()
        done = False
        steps = 0
        episode_reward = 0.0
        episode_pnl = 0.0

        while not done:
            action, _states = model.predict(obs, deterministic=True)
            if isinstance(action, np.ndarray):
                action = action.item()
            obs, reward, done, truncated, info = env.step(action)

            episode_reward += reward
            episode_pnl += info.get('pnl', 0.0)

            # Count actions
            if action == 0:
                action_counts["HOLD"] += 1
            elif action == 1:
                action_counts["BUY"] += 1
            else:
                action_counts["SELL"] += 1

            steps += 1

        total_reward += episode_reward
        total_pnl += episode_pnl
        total_rewards.append(episode_reward)
        total_pnls.append(episode_pnl)

        print(f"Episode {episode + 1}: Reward={episode_reward:.2f}, PnL={episode_pnl:.6f}, Steps={steps}")

    avg_reward = total_reward / episodes
    avg_pnl = total_pnl / episodes

    # Calculate statistics
    print("\\n=== Backtest Results ===")
    avg_reward = np.mean(total_rewards)
    avg_pnl = np.mean(total_pnls)

    print(f"Episodes: {episodes}")
    std_reward = np.std(total_rewards)
    std_pnl = np.std(total_pnls)

    print(f"Average Reward: {avg_reward:.4f} ± {std_reward:.4f}")
    print(f"Average PnL: {avg_pnl:.6f} ± {std_pnl:.6f}")

    total_actions = sum(action_counts.values())
    print(f"Total Return: {sum(total_pnls):.6f}")

    print("\\nAction Distribution:")
    for action, count in action_counts.items():
        percentage = (count / total_actions) * 100 if total_actions > 0 else 0
        print(f"  {action}: {count} ({percentage:.1f}%)")

    return {
        "avg_reward": avg_reward,
        "avg_pnl": avg_pnl,
        "total_pnl": total_pnl,
        "action_counts": action_counts
    }

if __name__ == "__main__":
    model_path = "models/progress_bar_test.zip"
    data_path = "ml-dataset-enhanced.csv"

    results = run_simple_backtest(model_path, data_path, episodes=5)
    if results:
        print("\\nBacktest completed successfully!")
    else:
        print("\\nBacktest failed!")
'''

with open('simple_backtest.py', 'w', encoding='utf-8') as f:
    f.write(content)