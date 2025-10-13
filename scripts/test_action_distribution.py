#!/usr/bin/env python3
"""
Quick action distribution test for SAC v402
"""

import sys
import os
import numpy as np
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from stable_baselines3 import SAC
from ztb.trading.environment.heavy_trading_env import HeavyTradingEnv
from ztb.trading.environment.utils.config import EnvironmentConfig

def test_action_distribution():
    """Test action distribution for SAC v402 model"""

    # Load model
    model_path = "checkpoints/sac_session/sac_v402_equal_actions_final.zip"
    model = SAC.load(model_path)

    # Create environment config
    config = EnvironmentConfig(
        initial_balance=200000.0,
        transaction_cost=0.0005,
        max_position_size=0.01,
        use_continuous_actions=True,
        use_standardized_observations=True,
        reward_settings={
            "use_simple_reward": True,
            "reward_scale": 2000.0,
            "reward_clip_min": -20.0,
            "reward_clip_max": 20.0,
            "enable_inactivity_penalty": True,
            "inactivity_penalty_rate": 0.0005,
            "enable_opportunity_cost": True,
            "opportunity_cost_rate": 0.0005,
            "enable_trade_execution_bonus": True,
            "trade_execution_bonus_rate": 0.1,
            "buy_action_penalty": -1.0,
            "sell_action_penalty": -1.0,
            "action_threshold_buy": 0.2,
            "action_threshold_sell": -0.2,
        }
    )

    # Create environment
    env = HeavyTradingEnv(config=config)

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

            # Convert to discrete for counting
            if action[0] < -0.2:
                discrete_action = 2  # SELL
            elif action[0] > 0.2:
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
    test_action_distribution()