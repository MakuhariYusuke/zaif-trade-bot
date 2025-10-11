#!/usr/bin/env python3
"""
Debug environment test script.
"""

import pandas as pd

from ztb.trading.environment import HeavyTradingEnv

# Load test data
df = pd.read_csv("ml-dataset-enhanced.csv")
print(f"Loaded {len(df)} rows")

# Create environment
env_config = {
    "reward_scaling": 1.0,
    "transaction_cost": 0.0,
    "max_position_size": 1.0,
    "feature_set": "full",
    "curriculum_stage": "full",
}

env = HeavyTradingEnv(df=df, config=env_config, random_start=False)
obs, info = env.reset()
print(f"Initial observation shape: {obs.shape}")
print(f"Initial position: {env.position}")
print(f"Initial portfolio value: {env.portfolio_value}")

# Test a few steps with different actions
for action in [0, 1, 2, 1, 0]:  # HOLD, BUY, SELL, BUY, HOLD
    obs, reward, terminated, truncated, info = env.step(action)
    pnl = info.get("pnl", 0.0)
    print(
        f"Action: {action}, Reward: {reward:.4f}, PnL: {pnl:.4f}, Position: {env.position:.4f}, Portfolio: {env.portfolio_value:.2f}"
    )
    if terminated or truncated:
        break
