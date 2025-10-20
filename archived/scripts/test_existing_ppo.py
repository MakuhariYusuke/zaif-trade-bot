#!/usr/bin/env python3
"""
Test Existing PPO Model - Check if PPO produces balanced actions
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from stable_baselines3 import PPO

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))


def test_existing_ppo():
    """Test existing PPO model for action distribution."""

    print("=" * 80)
    print("TESTING EXISTING PPO MODEL")
    print("=" * 80)

    # Load existing PPO model
    model_path = "models/ppo_profitable_v392_bugfix.zip"
    print(f"Loading model: {model_path}")

    try:
        model = PPO.load(model_path)
        print("✅ Model loaded successfully")
    except Exception as e:
        print(f"❌ Failed to load model: {e}")
        return

    # Load data
    df = pd.read_csv("btc_jpy_real_dataset.csv")
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values("timestamp").reset_index(drop=True)

    print(f"Testing with {len(df)} data points")

    # Test action distribution
    actions = []
    print("Sampling actions...")

    for i in range(2000):  # More samples for better statistics
        step = np.random.randint(100, len(df) - 100)
        obs = np.array(
            [
                df.iloc[step]["close"],
                df.iloc[step]["volume"] if "volume" in df.columns else 1000,
                0.0,
                0.0,
                0.0,  # position and portfolio features
            ],
            dtype=np.float32,
        )

        action, _ = model.predict(obs, deterministic=True)
        actions.append(float(action[0]))

    actions = np.array(actions)

    print("\nAction Distribution (2000 samples):")
    print(f"Mean:   {np.mean(actions):.4f}")
    print(f"Std:    {np.std(actions):.4f}")
    print(f"Min:    {np.min(actions):.4f}")
    print(f"Max:    {np.max(actions):.4f}")
    print(f"Median: {np.median(actions):.4f}")

    # Count discrete actions
    buy_threshold = 0.1
    sell_threshold = -0.1

    buy_count = sum(1 for a in actions if a > buy_threshold)
    sell_count = sum(1 for a in actions if a < sell_threshold)
    hold_count = sum(1 for a in actions if sell_threshold <= a <= buy_threshold)

    total = len(actions)
    print("\nDiscrete Action Distribution:")
    print(f"BUY:  {buy_count:4d} ({buy_count/total*100:5.1f}%)")
    print(f"SELL: {sell_count:4d} ({sell_count/total*100:5.1f}%)")
    print(f"HOLD: {hold_count:4d} ({hold_count/total*100:5.1f}%)")

    # Analyze distribution
    if buy_count == total:
        print("❌ Constant BUY bias!")
        return "BUY_BIAS"
    elif sell_count == total:
        print("❌ Constant SELL bias!")
        return "SELL_BIAS"
    elif abs(buy_count - sell_count) < total * 0.15:  # Within 15%
        print("✅ Balanced action distribution achieved!")
        return "BALANCED"
    else:
        print("⚠️ Some bias present")
        bias_ratio = (
            max(buy_count, sell_count) / min(buy_count, sell_count)
            if min(buy_count, sell_count) > 0
            else float("inf")
        )
        print(".2f")
        return "SOME_BIAS"


def test_multiple_ppo_models():
    """Test multiple PPO models to see if any produce balanced actions."""

    print("\n" + "=" * 80)
    print("TESTING MULTIPLE PPO MODELS")
    print("=" * 80)

    models_to_test = [
        "models/ppo_profitable_v392_bugfix.zip",
        "models/ppo_profitable_v391_optimized.zip",
        "models/ppo_profitable_v390_hybrid.zip",
        "models/ppo_balanced_mem_optimized.zip",
        "models/ppo_100k_balanced.zip",
    ]

    results = {}

    for model_path in models_to_test:
        print(f"\n--- Testing {Path(model_path).name} ---")
        try:
            result = test_existing_ppo_model(model_path)
            results[Path(model_path).name] = result
        except Exception as e:
            print(f"❌ Error testing {model_path}: {e}")
            results[Path(model_path).name] = "ERROR"

    print("\n" + "=" * 80)
    print("SUMMARY OF PPO MODEL TESTS")
    print("=" * 80)

    for model, result in results.items():
        print("12")

    balanced_count = sum(1 for r in results.values() if r == "BALANCED")
    print(f"\nBalanced models: {balanced_count}/{len(results)}")

    if balanced_count > 0:
        print("✅ Some PPO models achieve balanced actions!")
    else:
        print("❌ All PPO models show bias - issue may be in environment/reward design")


def test_existing_ppo_model(model_path):
    """Test a single PPO model."""

    try:
        model = PPO.load(model_path)
    except Exception as e:
        raise e

    df = pd.read_csv("btc_jpy_real_dataset.csv")
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values("timestamp").reset_index(drop=True)

    actions = []
    for i in range(1000):
        step = np.random.randint(100, len(df) - 100)
        obs = np.array(
            [
                df.iloc[step]["close"],
                df.iloc[step]["volume"] if "volume" in df.columns else 1000,
                0.0,
                0.0,
                0.0,
            ],
            dtype=np.float32,
        )

        action, _ = model.predict(obs, deterministic=True)
        actions.append(float(action[0]))

    actions = np.array(actions)

    buy_threshold = 0.1
    sell_threshold = -0.1

    buy_count = sum(1 for a in actions if a > buy_threshold)
    sell_count = sum(1 for a in actions if a < sell_threshold)
    hold_count = sum(1 for a in actions if sell_threshold <= a <= buy_threshold)

    total = len(actions)

    if buy_count == total:
        return "BUY_BIAS"
    elif sell_count == total:
        return "SELL_BIAS"
    elif abs(buy_count - sell_count) < total * 0.15:
        return "BALANCED"
    else:
        return "SOME_BIAS"


if __name__ == "__main__":
    test_multiple_ppo_models()
