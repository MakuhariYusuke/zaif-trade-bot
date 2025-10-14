#!/usr/bin/env python3
"""
Quick action distribution test for SAC models
"""

import sys
import os
import numpy as np
from pathlib import Path
import pandas as pd
import argparse
import json

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from stable_baselines3 import SAC
from ztb.trading.environment.heavy_env import HeavyTradingEnv
from ztb.trading.environment.utils.config import EnvironmentConfig

def test_action_distribution(config_path: str):
    """Test action distribution for SAC model"""

    # Load config
    with open(config_path, 'r') as f:
        config_dict = json.load(f)

    model_name = 'sac_v414_balanced_trading_final'  # Use the existing working v414 model
    model_path = f"checkpoints/sac_session/{model_name}.zip"

    print(f"Loading model: {model_path}")

    # Load model
    model = SAC.load(model_path)

    # Load data
    data_path = config_dict.get('data_path', 'btc_jpy_real_dataset.csv')
    df = pd.read_csv(data_path)

    # Create environment config from dict
    env_config_dict = config_dict.get('environment', {})
    reward_settings = config_dict.get('reward_settings', {})

    config = EnvironmentConfig.from_dict({
        **env_config_dict,
        'reward_settings': reward_settings
    })

    # Create environment
    env = HeavyTradingEnv(config=config, df=df)

    # Test action distribution
    actions = []
    discrete_actions = []

    print("Testing action distribution for SAC v402...")
    print("=" * 50)

    # Run multiple episodes to get action distribution
    for episode in range(10):
        obs, _ = env.reset()
        done = False
        episode_actions = []
        episode_discrete = []

        while not done:
            # Get action from model
            action, _ = model.predict(obs, deterministic=True)
            actions.append(action[0])

            # Convert to discrete for counting (using same threshold as environment: 0.1)
            if action[0] < -0.1:
                discrete_action = 2  # SELL
            elif action[0] > 0.1:
                discrete_action = 1  # BUY
            else:
                discrete_action = 0  # HOLD

            discrete_actions.append(discrete_action)
            episode_actions.append(action[0])
            episode_discrete.append(discrete_action)

            # Step environment
            obs, reward, done, truncated, info = env.step(action)
            done = done or truncated

        print(f"Episode {episode+1}: Actions - BUY:{episode_discrete.count(1)}, SELL:{episode_discrete.count(2)}, HOLD:{episode_discrete.count(0)}")

    # Overall statistics
    total_actions = len(discrete_actions)
    buy_count = discrete_actions.count(1)
    sell_count = discrete_actions.count(2)
    hold_count = discrete_actions.count(0)

    buy_ratio = buy_count / total_actions * 100
    sell_ratio = sell_count / total_actions * 100
    hold_ratio = hold_count / total_actions * 100

    print("\nOverall Action Distribution:")
    print("=" * 30)
    print(f"Total Actions: {total_actions}")
    print(f"BUY:  {buy_count} ({buy_ratio:.1f}%)")
    print(f"SELL: {sell_count} ({sell_ratio:.1f}%)")
    print(f"HOLD: {hold_count} ({hold_ratio:.1f}%)")

    # Continuous action statistics
    actions_array = np.array(actions)
    print("\nContinuous Action Statistics:")
    print("=" * 35)
    print(f"Mean: {actions_array.mean():.3f}")
    print(f"Std:  {actions_array.std():.3f}")
    print(f"Min:  {actions_array.min():.3f}")
    print(f"Max:  {actions_array.max():.3f}")

    # Check balance
    if abs(buy_ratio - sell_ratio) < 10:
        print("\n✅ BUY/SELL actions are well balanced!")
    else:
        print(f"\n⚠️  BUY/SELL imbalance detected. Difference: {abs(buy_ratio - sell_ratio):.1f}%")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Test action distribution for SAC model')
    parser.add_argument('--config', type=str, required=True, help='Path to config JSON file')
    args = parser.parse_args()
    
    test_action_distribution(args.config)