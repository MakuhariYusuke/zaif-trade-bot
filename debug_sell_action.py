#!/usr/bin/env python3
"""
Debug script to test if SELL actions are executable in the environment.
"""

import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent / 'src'))

import pandas as pd
from ztb.trading.environment import HeavyTradingEnv

def test_sell_action():
    """Test if SELL action can be executed properly."""

    # Load data
    data_path = Path(__file__).parent / "ml-dataset-enhanced.csv"
    df = pd.read_csv(data_path)

    # Create environment with simple reward
    env_config = {
        "reward_scaling": 6.0,
        "transaction_cost": 0.001,
        "max_position_size": 1.0,
        "risk_free_rate": 0.02,
        "feature_set": "full",
        "initial_portfolio_value": 1000000.0,
        "curriculum_stage": "simple_portfolio",  # Use simple reward
        "reward_settings": {
            "enable_forced_diversity": False,  # Disable forced diversity for this test
            "profit_bonus_multipliers": [1.0, 1.0, 1.0],
        },
    }

    env = HeavyTradingEnv(
        df=df,
        config=env_config,
        streaming_pipeline=None,
        stream_batch_size=1000,
        max_features=68
    )

    print("=== Testing SELL Action Execution ===")

    # Reset environment
    env.reset()
    print(f"Initial position: {env.position}")
    print(f"Initial portfolio value: {env.portfolio_value}")

    # Test HOLD action
    print("\n--- Testing HOLD action ---")
    obs, reward, terminated, truncated, info = env.step(0)  # HOLD
    print(f"After HOLD - Position: {env.position}, Reward: {reward}")

    # Test BUY action
    print("\n--- Testing BUY action ---")
    obs, reward, terminated, truncated, info = env.step(1)  # BUY
    print(f"After BUY - Position: {env.position}, Reward: {reward}")

    # Test SELL action (should work if we have a position)
    print("\n--- Testing SELL action ---")
    obs, reward, terminated, truncated, info = env.step(2)  # SELL
    print(f"After SELL - Position: {env.position}, Reward: {reward}")

    # Test SELL again (should work if flat)
    print("\n--- Testing SELL action again ---")
    obs, reward, terminated, truncated, info = env.step(2)  # SELL
    print(f"After SELL again - Position: {env.position}, Reward: {reward}")

    print("\n=== Reward Function Test ===")
    print("Expected rewards for simple_portfolio stage:")
    print("HOLD (0): -1.0")
    print("BUY (1): -0.5")
    print("SELL (2): 2.0")

    # Test reward calculation by simulating the step method
    print("\n--- Testing reward calculation ---")
    for action in [0, 1, 2]:
        # Simulate step to get reward
        env.reset()
        obs, reward, terminated, truncated, info = env.step(action)
        print(f"Action {action} reward: {reward}")

    env.close()

if __name__ == "__main__":
    test_sell_action()