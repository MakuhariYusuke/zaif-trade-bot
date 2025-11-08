#!/usr/bin/env python3
"""
Debug script to investigate SELL bias in SAC v444 training.
Tests reward function behavior for different actions.
"""

import json
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from ztb.trading.environment.heavy_env.core import HeavyTradingEnv
from ztb.trading.constants import ACTION_BUY, ACTION_HOLD, ACTION_SELL
import numpy as np
import pandas as pd

def test_reward_function():
    """Test reward function for different actions to identify SELL bias."""

    # Load config
    config_path = "config/sac_v444_improved_balance_penalty.json"
    with open(config_path, "r") as f:
        config = json.load(f)

    # Create minimal test data
    dates = pd.date_range("2023-01-01", periods=100, freq="1H")
    test_data = pd.DataFrame({
        "timestamp": dates,
        "open": np.random.uniform(5000000, 5100000, 100),
        "high": np.random.uniform(5050000, 5150000, 100),
        "low": np.random.uniform(4950000, 5050000, 100),
        "close": np.random.uniform(5000000, 5100000, 100),
        "volume": np.random.uniform(100, 1000, 100),
    })

    # Set close prices with slight upward trend to test SELL behavior
    test_data["close"] = 5000000 + np.arange(100) * 100  # Slight upward trend

    # Create environment
    env = HeavyTradingEnv(df=test_data, config=config)

    print("=== Reward Function Debug Test ===")
    print(f"Config curriculum_stage: {config['environment']['curriculum_stage']}")
    print(f"Environment curriculum_stage: {env.config.curriculum_stage}")
    print(f"Reward calculator curriculum_stage: {env.reward_calculator.config.curriculum_stage}")
    print()

    # Test different scenarios
    test_scenarios = [
        {"name": "Initial state (no position)", "position": 0.0, "old_position": 0.0, "pnl": 0.0},
        {"name": "After BUY (small profit)", "position": 0.1, "old_position": 0.0, "pnl": 1000.0},
        {"name": "After SELL (small profit)", "position": -0.1, "old_position": 0.0, "pnl": 1000.0},
        {"name": "After HOLD (no change)", "position": 0.0, "old_position": 0.0, "pnl": 0.0},
        {"name": "After BUY (small loss)", "position": 0.1, "old_position": 0.0, "pnl": -500.0},
        {"name": "After SELL (small loss)", "position": -0.1, "old_position": 0.0, "pnl": -500.0},
    ]

    current_price = 5010000.0
    atr = 1000.0

    for scenario in test_scenarios:
        print(f"--- {scenario['name']} ---")

        rewards = {}
        for action_name, action in [("BUY", ACTION_BUY), ("HOLD", ACTION_HOLD), ("SELL", ACTION_SELL)]:
            reward = env.reward_calculator.calculate_reward(
                action=action,
                current_price=current_price,
                position=scenario["position"],
                portfolio_value=200000.0,
                atr=atr,
                transaction_cost=0.001,
                reward_scaling=1.0,
                pnl=scenario["pnl"],
                old_position=scenario["old_position"],
                step=50,  # After forced_balance should be active
                observation=None,
                reward_history=[],
                portfolio_value_history=[200000.0],
            )
            rewards[action_name] = reward
            print(".4f")

        # Check for SELL bias
        sell_vs_buy = rewards["SELL"] - rewards["BUY"]
        sell_vs_hold = rewards["SELL"] - rewards["HOLD"]

        print(".4f")
        print(".4f")

        if rewards["SELL"] > rewards["BUY"] and rewards["SELL"] > rewards["HOLD"]:
            print("⚠️  SELL bias detected!")
        else:
            print("✅ No SELL bias in this scenario")

        print()

    # Test forced_balance specifically
    print("=== Forced Balance Test ===")
    print("Testing forced_balance reward calculation...")

    # Simulate imbalanced actions (mostly SELL)
    env.reward_calculator._action_counts = [10, 10, 70]  # [HOLD, BUY, SELL]

    for action_name, action in [("BUY", ACTION_BUY), ("HOLD", ACTION_HOLD), ("SELL", ACTION_SELL)]:
        forced_reward = env.reward_calculator._calculate_forced_balance_reward(action, 50)
        print(".4f")

    print()
    print("Action counts:", env.reward_calculator._action_counts)
    total_actions = sum(env.reward_calculator._action_counts)
    ratios = [count / total_actions for count in env.reward_calculator._action_counts]
    print(".3f")

if __name__ == "__main__":
    test_reward_function()