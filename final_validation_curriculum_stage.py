#!/usr/bin/env python
"""Final validation test - ensure curriculum_stage is loaded correctly in environment."""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
import pandas as pd

from ztb.trading.environment.heavy_env.core import HeavyTradingEnv
from ztb.utils.v4xx_config_converter import V4XXConfigConverter

# Create minimal dataframe
def create_sample_data():
    dates = pd.date_range("2023-01-01", periods=200, freq="1h")
    base_price = 5000000
    price_changes = np.random.normal(0, 0.005, 200).cumsum()
    close = pd.Series(base_price * (1 + price_changes), index=dates)
    
    df = pd.DataFrame({
        "open": close.shift(1).fillna(close.iloc[0]),
        "high": close * 1.01,
        "low": close * 0.99,
        "close": close,
        "volume": np.random.uniform(1000, 10000, 200),
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

print("="*70)
print("FINAL VALIDATION TEST: curriculum_stage in HeavyTradingEnv")
print("="*70)

# Load and convert config
print("\n1. Loading config file...")
config = json.load(open('config/sac_v444_3_balanced_penalty_scale_200.json'))
converted = V4XXConfigConverter.convert_v444_to_unified(config)
env_dict = converted.get('training', {}).get('environment', {})
print(f"   ✓ env_dict['curriculum_stage'] = {env_dict.get('curriculum_stage')}")

# Create environment
print("\n2. Creating HeavyTradingEnv...")
df = create_sample_data()
env = HeavyTradingEnv(df, env_dict)
print(f"   ✓ env.config.curriculum_stage = {env.config.curriculum_stage}")

# Check reward calculator
print("\n3. Checking RewardCalculator...")
print(f"   ✓ env.reward_calculator.config.curriculum_stage = {env.reward_calculator.config.curriculum_stage}")

# Final verdict
if env.config.curriculum_stage == 'balanced_penalty':
    print("\n" + "="*70)
    print("✅ SUCCESS! curriculum_stage is correctly loaded and available")
    print("="*70)
else:
    print("\n" + "="*70)
    print(f"❌ FAILED! curriculum_stage = {env.config.curriculum_stage}")
    print("="*70)
