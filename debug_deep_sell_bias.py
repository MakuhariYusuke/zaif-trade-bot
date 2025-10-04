#!/usr/bin/env python3
"""
Debug script to investigate why models prefer SELL action even with neutral rewards.
Tests different actions and compares their effects on environment state.
"""

import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent / 'src'))

import numpy as np
import pandas as pd
from ztb.trading.environment import HeavyTradingEnv


def debug_action_effects():
    """Debug what happens when different actions are taken"""

    print("=== Deep Debug: Action Effects Analysis ===")

    # Load data
    data_path = Path(__file__).parent / "ml-dataset-enhanced.csv"
    df = pd.read_csv(data_path)

    # Create environment with neutral rewards
    env_config = {
        "reward_scaling": 6.0,
        "transaction_cost": 0.001,
        "max_position_size": 1.0,
        "risk_free_rate": 0.02,
        "feature_set": "full",
        "initial_portfolio_value": 1000000.0,
        "curriculum_stage": "simple_portfolio",
        "reward_settings": {
            "custom_reward_params": {
                "hold_penalty": 0.0,
                "buy_penalty": 0.0,
                "sell_reward": 0.0
            }
        }
    }

    env = HeavyTradingEnv(df=df, config=env_config)

    # Reset environment
    obs, info = env.reset()
    print(f"Initial position: {env.position}")
    print(f"Initial portfolio value: {env.portfolio_value}")
    print(f"Observation shape: {obs.shape}")
    print(f"Features count: {len(env.features)}")
    print(f"First 10 features: {env.features[:10]}")

    # Test each action and compare effects
    actions = [0, 1, 2]  # HOLD, BUY, SELL
    action_names = ["HOLD", "BUY", "SELL"]

    for action, name in zip(actions, action_names):
        print(f"\n--- Testing {name} action ---")

        # Save state before action
        old_position = env.position
        old_portfolio = env.portfolio_value
        old_obs = obs.copy()

        # Take action
        obs, reward, done, truncated, info = env.step(action)

        # Compare states
        print(f"Action: {action} ({name})")
        print(f"Reward: {reward}")
        print(f"Position change: {old_position} -> {env.position}")
        print(f"Portfolio change: {old_portfolio:.2f} -> {env.portfolio_value:.2f}")
        print(f"PnL: {info.get('pnl', 'N/A')}")

        # Check observation differences
        obs_diff = obs - old_obs
        significant_changes = np.abs(obs_diff) > 1e-6
        if np.any(significant_changes):
            print(f"Observation changes at indices: {np.where(significant_changes)[0]}")
            print(f"Change values: {obs_diff[significant_changes]}")
        else:
            print("No significant observation changes")

        # Reset for next test
        obs, info = env.reset()

    env.close()


def debug_state_patterns():
    """Debug patterns in state that might bias toward SELL"""

    print("\n=== Deep Debug: State Pattern Analysis ===")

    # Load data
    data_path = Path(__file__).parent / "ml-dataset-enhanced.csv"
    df = pd.read_csv(data_path)

    # Create environment with neutral rewards
    env_config = {
        "reward_scaling": 6.0,
        "transaction_cost": 0.001,
        "max_position_size": 1.0,
        "risk_free_rate": 0.02,
        "feature_set": "full",
        "initial_portfolio_value": 1000000.0,
        "curriculum_stage": "simple_portfolio",
        "reward_settings": {
            "custom_reward_params": {
                "hold_penalty": 0.0,
                "buy_penalty": 0.0,
                "sell_reward": 0.0
            }
        }
    }

    env = HeavyTradingEnv(df=df, config=env_config)

    # Collect states from multiple steps
    states = []
    actions_taken = []

    obs, info = env.reset()

    for step in range(100):  # Sample 100 steps
        # Random action to explore different states
        action = np.random.choice([0, 1, 2])
        obs, reward, done, truncated, info = env.step(action)

        states.append(obs.copy())
        actions_taken.append(action)

        if done:
            obs, info = env.reset()

    states = np.array(states)
    actions_taken = np.array(actions_taken)

    print(f"Collected {len(states)} state samples")
    print(f"Action distribution: H={np.sum(actions_taken==0)}, B={np.sum(actions_taken==1)}, S={np.sum(actions_taken==2)}")

    # Analyze state patterns for different actions
    for action, name in [(0, "HOLD"), (1, "BUY"), (2, "SELL")]:
        action_states = states[actions_taken == action]
        if len(action_states) > 0:
            print(f"\n{name} action states ({len(action_states)} samples):")
            print(f"  Mean values (first 10 features): {np.mean(action_states[:, :10], axis=0)}")
            print(f"  Std values (first 10 features): {np.std(action_states[:, :10], axis=0)}")

    env.close()


if __name__ == "__main__":
    debug_action_effects()
    debug_state_patterns()