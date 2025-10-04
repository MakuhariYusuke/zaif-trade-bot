#!/usr/bin/env python3
"""
Random action test to verify reward function works correctly.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

import pandas as pd
from ztb.trading.environment import HeavyTradingEnv
import numpy as np

def test_random_actions():
    """Test random actions to verify reward function."""

    # Load data
    df = pd.read_csv('ml-dataset-enhanced.csv')

    # Create environment
    env_config = {
        'reward_scaling': 6.0,
        'transaction_cost': 0.001,
        'max_position_size': 1.0,
        'risk_free_rate': 0.02,
        'feature_set': 'full',
        'initial_portfolio_value': 1000000.0,
        'curriculum_stage': 'simple_portfolio',
        'reward_settings': {
            'enable_forced_diversity': False,
            'profit_bonus_multipliers': [1.0, 1.0, 1.0]
        },
    }

    env = HeavyTradingEnv(
        df=df,
        config=env_config,
        streaming_pipeline=None,
        stream_batch_size=1000,
        max_features=68
    )

    print('=== Random Action Test ===')
    env.reset()
    total_reward = 0
    actions_taken = []

    for i in range(100):
        action = np.random.choice([0, 1, 2])
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        actions_taken.append(action)

        if terminated or truncated:
            break

    print(f'Total steps: {len(actions_taken)}')
    print(f'Actions: H={actions_taken.count(0)}, B={actions_taken.count(1)}, S={actions_taken.count(2)}')
    print(f'Total reward: {total_reward:.2f}')
    print(f'Average reward per step: {total_reward/len(actions_taken):.4f}')

    # Expected reward calculation
    expected_avg = (actions_taken.count(0) * -1.0 + actions_taken.count(1) * -0.5 + actions_taken.count(2) * 2.0) / len(actions_taken)
    print(f'Expected average reward: {expected_avg:.4f}')

    env.close()

if __name__ == "__main__":
    test_random_actions()