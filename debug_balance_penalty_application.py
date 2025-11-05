#!/usr/bin/env python
"""Debug script to verify balance_penalty is actually being applied during reward calculation."""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
import pandas as pd

from ztb.trading.environment.heavy_env.core import HeavyTradingEnv
from ztb.trading.constants import ACTION_BUY, ACTION_HOLD, ACTION_SELL
from ztb.utils.v4xx_config_converter import V4XXConfigConverter

# Create minimal dataframe
def create_sample_data():
    dates = pd.date_range("2023-01-01", periods=100, freq="1h")
    base_price = 5000000
    price_changes = np.random.normal(0, 0.005, 100).cumsum()
    close = pd.Series(base_price * (1 + price_changes), index=dates)
    
    df = pd.DataFrame({
        "open": close.shift(1).fillna(close.iloc[0]),
        "high": close * 1.01,
        "low": close * 0.99,
        "close": close,
        "volume": np.random.uniform(1000, 10000, 100),
        "timestamp": dates,
    })
    
    # Add technical indicators
    df["SMA_20"] = df["close"].rolling(20).mean()
    df["SMA_50"] = df["close"].rolling(50).mean()
    df["RSI"] = 50
    df["MACD"] = df["close"].ewm(span=12).mean() - df["close"].ewm(span=26).mean()
    df["BB_Upper"] = df["close"].rolling(20).mean() + 2 * df["close"].rolling(20).std()
    df["BB_Lower"] = df["close"].rolling(20).mean() - 2 * df["close"].rolling(20).std()
    
    return df.ffill().bfill()

print("="*80)
print("DEBUG: Verify balance_penalty is applied correctly")
print("="*80)

# Load config
print("\n1. Loading configuration...")
config = json.load(open('config/sac_v444_3_balanced_penalty_scale_200.json'))
converted = V4XXConfigConverter.convert_v444_to_unified(config)
env_dict = converted.get('training', {}).get('environment', {})

print(f"   curriculum_stage: {env_dict.get('curriculum_stage')}")
print(f"   balance_penalty: {env_dict.get('balance_penalty')}")
print(f"   buy_action_bonus: {env_dict.get('buy_action_bonus')}")
print(f"   sell_action_bonus: {env_dict.get('sell_action_bonus')}")
print(f"   hold_action_bonus: {env_dict.get('hold_action_bonus')}")

# Create environment
print("\n2. Creating HeavyTradingEnv...")
df = create_sample_data()
env = HeavyTradingEnv(df, env_dict)

print(f"   env.config.curriculum_stage: {env.config.curriculum_stage}")
print(f"   env.reward_calculator.config.curriculum_stage: {env.reward_calculator.config.curriculum_stage}")

# Initialize environment
obs, info = env.reset()

print("\n3. Testing reward calculation with different actions...")
print("   (Taking actions and observing how balance_penalty affects rewards)\n")

action_names = {1.0: "BUY", 0.0: "HOLD", -1.0: "SELL"}
test_actions = [1.0, 0.0, -1.0]

for action_value in test_actions:
    # Take action (convert to numpy array for continuous action space)
    obs, reward, terminated, truncated, info = env.step(np.array([action_value]))
    
    action_name = action_names.get(action_value, f"UNKNOWN({action_value})")
    print(f"   Action: {action_name:6} | Reward: {reward:10.6f} | Position: {env.position_manager.position:8.4f}")

print("\n" + "="*80)
print("4. Analyzing reward settings in calculator...")
print("="*80)

# Check reward settings
reward_calc = env.reward_calculator
print(f"\n   config.curriculum_stage: {reward_calc.config.curriculum_stage}")
print(f"   config.reward_balance_penalty_scale: {getattr(reward_calc.config, 'reward_balance_penalty_scale', 'NOT SET')}")
print(f"   config.balance_penalty: {getattr(reward_calc.config, 'balance_penalty', 'NOT SET')}")
print(f"   config.action_balance_target: {getattr(reward_calc.config, 'action_balance_target', 'NOT SET')}")

# Check if balanced_penalty method exists and is callable
print(f"\n   Has calculate_balanced_penalty_penalty: {hasattr(reward_calc, 'calculate_balanced_penalty_penalty')}")
print(f"   Has _calculate_balance_penalty: {hasattr(reward_calc, '_calculate_balance_penalty')}")

# Look for any method that handles balance penalty
balance_methods = [m for m in dir(reward_calc) if 'balance' in m.lower() or 'penalty' in m.lower()]
print(f"\n   Methods containing 'balance' or 'penalty':")
for method in balance_methods:
    print(f"      - {method}")

print("\n" + "="*80)
