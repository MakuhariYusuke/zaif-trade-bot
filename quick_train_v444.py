#!/usr/bin/env python3
"""
Quick training script for V444 backtest
"""

import sys
from pathlib import Path
import json
import pandas as pd
import numpy as np

# Add project root to path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from stable_baselines3 import SAC
from ztb.trading.environment.heavy_env.core import HeavyTradingEnv

def create_sample_data():
    """Create sample data for quick training"""
    np.random.seed(42)
    dates = pd.date_range('2023-01-01', periods=1000, freq='1h')

    # Generate sample price data
    base_price = 5000000
    price_changes = np.random.normal(0, 0.005, 1000).cumsum()
    close = pd.Series(base_price * (1 + price_changes), index=dates)

    # Generate OHLCV
    high = close * (1 + np.abs(np.random.normal(0, 0.002, 1000)))
    low = close * (1 - np.abs(np.random.normal(0, 0.002, 1000)))
    open_price = close.shift(1).fillna(close.iloc[0])
    volume = pd.Series(np.random.uniform(1000, 10000, 1000), index=dates)

    # Add some basic features
    df = pd.DataFrame({
        'open': open_price,
        'high': high,
        'low': low,
        'close': close,
        'volume': volume,
        'timestamp': dates
    })

    # Add basic technical indicators
    df['SMA_20'] = df['close'].rolling(20).mean()
    df['SMA_50'] = df['close'].rolling(50).mean()
    df['RSI'] = 50  # Simple placeholder
    df['MACD'] = df['close'].ewm(span=12).mean() - df['close'].ewm(span=26).mean()
    df['BB_Upper'] = df['close'].rolling(20).mean() + 2 * df['close'].rolling(20).std()
    df['BB_Lower'] = df['close'].rolling(20).mean() - 2 * df['close'].rolling(20).std()

    return df.fillna(method='bfill').fillna(method='ffill')

def quick_train():
    """Quick training for backtest"""
    print("Creating sample data...")
    df = create_sample_data()

    print("Setting up environment...")
    config = {
        "initial_balance": 100000.0,
        "commission": 0.001,
        "max_position_size": 1.0,
        "reward_scaling": 1.0,
        "action_space_type": "continuous",
        "use_continuous_actions": True,
        "feature_set": "basic",
        "data": df
    }

    env = HeavyTradingEnv(config)

    print("Creating SAC model...")
    model = SAC(
        "MlpPolicy",
        env,
        learning_rate=0.001,
        buffer_size=10000,
        learning_starts=100,
        batch_size=64,
        tau=0.005,
        gamma=0.99,
        ent_coef=0.1,
        target_update_interval=1,
        target_entropy=-2.0,
        verbose=1
    )

    print("Training model (1000 steps)...")
    model.learn(total_timesteps=1000)

    # Save model
    model_path = "models/quick_v444_model.zip"
    model.save(model_path)
    print(f"Model saved to {model_path}")

    # Save data
    data_path = "data/quick_training_data.csv"
    df.to_csv(data_path, index=False)
    print(f"Data saved to {data_path}")

    return model_path, data_path

if __name__ == "__main__":
    try:
        model_path, data_path = quick_train()
        print("Quick training completed!")
        print(f"Model: {model_path}")
        print(f"Data: {data_path}")
    except Exception as e:
        print(f"Training failed: {e}")
        import traceback
        traceback.print_exc()