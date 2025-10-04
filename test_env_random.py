#!/usr/bin/env python3
"""
Test HeavyTradingEnv with random policy for 100 steps
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
from collections import Counter

# Add project root to path
PROJECT_ROOT = Path(__file__).parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ztb.trading.environment import HeavyTradingEnv

def test_random_policy():
    """Test environment with random policy for 100 steps"""

    # Load data
    data_path = PROJECT_ROOT / "ml-dataset-enhanced.csv"
    df = pd.read_csv(data_path)

    # Create environment
    env_config = {
        "reward_scaling": 1.0,
        "transaction_cost": 0.001,
        "max_position_size": 1.0,
        "risk_free_rate": 0.02,
        "feature_set": "full",
        "initial_portfolio_value": 1000000.0,
    }

    env = HeavyTradingEnv(
        df=df,
        config=env_config,
        streaming_pipeline=None,
        stream_batch_size=1000,
        max_features=68
    )

    # Reset environment
    obs, info = env.reset()
    initial_portfolio_value = info.get('portfolio_value', 1000000.0)

    print(f"Initial portfolio value: {initial_portfolio_value}")
    print("Testing random policy for 100 steps...")

    actions_taken = []
    portfolio_values = [initial_portfolio_value]
    rewards = []

    for step in range(100):
        # Random action
        action = np.random.randint(0, 3)  # 0=Hold, 1=Buy, 2=Sell
        actions_taken.append(action)

        # Step
        obs, reward, done, truncated, info = env.step(action)
        portfolio_value = info.get('portfolio_value', portfolio_values[-1])

        portfolio_values.append(portfolio_value)
        rewards.append(reward)

        if step < 10 or step % 20 == 0:  # Log first 10 and every 20th step
            print(f"Step {step}: Action={action}, Reward={reward:.6f}, Portfolio={portfolio_value:.2f}, Position={info.get('position', 0)}")

        if done:
            break

    # Final statistics
    final_portfolio_value = portfolio_values[-1]
    total_return = (final_portfolio_value - initial_portfolio_value) / initial_portfolio_value * 100

    action_counts = Counter(actions_taken)
    action_distribution = {
        0: action_counts.get(0, 0),  # Hold
        1: action_counts.get(1, 0),  # Buy
        2: action_counts.get(2, 0),  # Sell
    }

    print("\n=== Results ===")
    print(f"Initial Portfolio: {initial_portfolio_value:.2f}")
    print(f"Final Portfolio: {final_portfolio_value:.2f}")
    print(f"Total Return: {total_return:.4f}%")
    print(f"Action Distribution: Hold={action_distribution[0]}, Buy={action_distribution[1]}, Sell={action_distribution[2]}")
    print(f"Total Steps: {len(actions_taken)}")
    print(f"Average Reward: {np.mean(rewards):.6f}")
    print(f"Reward Std: {np.std(rewards):.6f}")

    env.close()

if __name__ == "__main__":
    test_random_policy()