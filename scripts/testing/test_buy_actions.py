#!/usr/bin/env python3
"""
Test script to verify BUY action selection in trained SAC model
"""

import sys

sys.path.insert(0, ".")
from collections import Counter

import numpy as np
import pandas as pd
from stable_baselines3 import SAC


def main():
    # Load the trained model
    model_path = "models/quick_v444_model.zip"
    try:
        model = SAC.load(model_path)
        print("✅ Model loaded successfully")
    except Exception as e:
        print(f"❌ Failed to load model: {e}")
        return

    # Create test environment (minimal setup)
    from ztb.trading.environment.heavy_env.core import HeavyTradingEnv

    # Create sample data (exact same as training - 1000 steps)
    np.random.seed(42)
    dates = pd.date_range("2023-01-01", periods=1000, freq="1h")
    base_price = 5000000
    price_changes = np.random.normal(0, 0.005, 1000).cumsum()
    close = pd.Series(base_price * (1 + price_changes), index=dates)
    high = close * (1 + np.abs(np.random.normal(0, 0.002, 1000)))
    low = close * (1 - np.abs(np.random.normal(0, 0.002, 1000)))
    open_price = close.shift(1).fillna(close.iloc[0])
    volume = pd.Series(np.random.uniform(1000, 10000, 1000), index=dates)

    df = pd.DataFrame(
        {
            "open": open_price,
            "high": high,
            "low": low,
            "close": close,
            "volume": volume,
            "timestamp": dates,
        }
    )

    # Add basic technical indicators (same as training script)
    df["SMA_20"] = df["close"].rolling(20).mean()
    df["SMA_50"] = df["close"].rolling(50).mean()
    df["RSI"] = 50  # Simple placeholder
    df["MACD"] = df["close"].ewm(span=12).mean() - df["close"].ewm(span=26).mean()
    df["BB_Upper"] = df["close"].rolling(20).mean() + 2 * df["close"].rolling(20).std()
    df["BB_Lower"] = df["close"].rolling(20).mean() - 2 * df["close"].rolling(20).std()

    # Fill NaN values
    df = df.ffill().bfill()

    config = {
        "initial_balance": 100000.0,
        "commission": 0.001,
        "max_position_size": 1.0,
        "reward_scaling": 1.0,
        "action_space_type": "continuous",
        "use_continuous_actions": True,
        "feature_set": "minimal",
    }

    env = HeavyTradingEnv(df, config)

    # Test action selection
    actions_taken = []
    rewards_received = []

    obs, info = env.reset()
    print(f"Initial portfolio value: {env.portfolio_value:.2f}")
    print(f"Initial position: {env.position:.6f}")

    for step in range(50):  # Test 50 steps
        action, _ = model.predict(obs, deterministic=False)
        obs, reward, terminated, truncated, info = env.step(action)

        # Convert continuous action to discrete for logging
        if action < -0.333:
            action_type = "SELL"
        elif action > 0.333:
            action_type = "BUY"
        else:
            action_type = "HOLD"

        actions_taken.append(action_type)
        rewards_received.append(reward)

        if terminated or truncated:
            break

    # Analyze results
    action_counts = Counter(actions_taken)

    print("\n=== Action Distribution (50 steps) ===")
    print(f'BUY: {action_counts.get("BUY", 0)}')
    print(f'SELL: {action_counts.get("SELL", 0)}')
    print(f'HOLD: {action_counts.get("HOLD", 0)}')

    print(f"\nFinal portfolio value: {env.portfolio_value:.2f}")
    print(f"Final position: {env.position:.6f}")
    print(f"Total reward: {sum(rewards_received):.2f}")
    print(f"Average reward: {np.mean(rewards_received):.2f}")

    if action_counts.get("BUY", 0) > 0:
        print("✅ BUY actions are being selected!")
        print("\n🎉 SUCCESS: SELL-lock fix verified - BUY actions are now possible!")
    else:
        print("❌ No BUY actions selected - issue may still exist")


if __name__ == "__main__":
    main()
