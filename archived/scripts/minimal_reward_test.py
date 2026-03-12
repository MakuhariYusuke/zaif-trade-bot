#!/usr/bin/env python3
"""
Simple Reward Function Test - SAC SELL Bias Root Cause Investigation

Tests SAC with a minimal reward function to isolate the bias cause.
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from stable_baselines3 import SAC

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from ztb.trading.environment.heavy_env.core import HeavyTradingEnv
from ztb.trading.environment.utils.config import EnvironmentConfig


def create_minimal_reward_env():
    """Create environment with minimal reward function."""
    config = EnvironmentConfig(
        max_position_size=0.01,
        transaction_cost=0.0,
        reward_scaling=1.0,  # Minimal scaling
        reward_clip_value=1.0,  # Minimal clipping
        reward_settings={
            "use_simple_reward": True,
            "reward_scale": 1.0,
            "reward_clip_min": -1.0,
            "reward_clip_max": 1.0,
            # No action penalties or bonuses - completely neutral
            "buy_action_penalty": 0.0,
            "sell_action_penalty": 0.0,
            "hold_action_penalty": 0.0,
            "profit_bonus_multipliers": [1.0, 1.0, 1.0],  # Equal for all actions
        },
    )

    # Load sample data
    df = pd.read_csv("btc_jpy_real_dataset.csv")
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values("timestamp").reset_index(drop=True)

    env = HeavyTradingEnv(df=df, config=config)
    return env


def test_minimal_reward_function():
    """Test reward function with minimal settings."""

    print("=" * 80)
    print("MINIMAL REWARD FUNCTION TEST")
    print("=" * 80)

    env = create_minimal_reward_env()

    # Test scenarios
    test_cases = [
        {
            "action": 1,
            "pnl": 100.0,
            "position": 0.01,
            "old_position": 0.0,
            "desc": "BUY with profit",
        },
        {
            "action": 2,
            "pnl": 100.0,
            "position": 0.0,
            "old_position": 0.01,
            "desc": "SELL with profit",
        },
        {
            "action": 0,
            "pnl": 0.0,
            "position": 0.01,
            "old_position": 0.01,
            "desc": "HOLD neutral",
        },
        {
            "action": 1,
            "pnl": -50.0,
            "position": 0.01,
            "old_position": 0.0,
            "desc": "BUY with loss",
        },
        {
            "action": 2,
            "pnl": -50.0,
            "position": 0.0,
            "old_position": 0.01,
            "desc": "SELL with loss",
        },
    ]

    results = []

    for case in test_cases:
        reward = env.reward_calculator.calculate_reward(
            action=case["action"],
            current_price=5000000.0,
            position=case["position"],
            portfolio_value=200000.0,
            atr=1.0,
            transaction_cost=0.0,
            reward_scaling=1.0,
            pnl=case["pnl"],
            old_position=case["old_position"],
            step=1,
            observation=None,
            reward_history=[],
            portfolio_value_history=[200000.0] * 30,
        )

        print(f"{case['desc']:20}: {reward:+.6f}")

        results.append(
            {
                "description": case["desc"],
                "action": case["action"],
                "pnl": case["pnl"],
                "reward": reward,
            }
        )

    # Check for symmetry
    buy_profit_reward = results[0]["reward"]
    sell_profit_reward = results[1]["reward"]
    buy_loss_reward = results[3]["reward"]
    sell_loss_reward = results[4]["reward"]

    print("\nSymmetry Analysis:")
    print(f"BUY profit reward:  {buy_profit_reward:+.6f}")
    print(f"SELL profit reward: {sell_profit_reward:+.6f}")
    print(f"Difference:         {abs(buy_profit_reward - sell_profit_reward):.6f}")

    print(f"BUY loss reward:    {buy_loss_reward:+.6f}")
    print(f"SELL loss reward:   {sell_loss_reward:+.6f}")
    print(f"Difference:         {abs(buy_loss_reward - sell_loss_reward):.6f}")

    if (
        abs(buy_profit_reward - sell_profit_reward) < 1e-6
        and abs(buy_loss_reward - sell_loss_reward) < 1e-6
    ):
        print("✅ Reward function is SYMMETRIC between BUY and SELL")
    else:
        print("❌ Reward function has ASYMMETRY between BUY and SELL")

    return results


def train_minimal_sac():
    """Train SAC with minimal reward function."""

    print("\n" + "=" * 80)
    print("TRAINING SAC WITH MINIMAL REWARD FUNCTION")
    print("=" * 80)

    from ztb.training.core.algorithm_trainer import AlgorithmTrainer
    from ztb.training.core.config_manager import ConfigManager

    # Minimal config
    config = {
        "model_name": "sac_v407_minimal_reward",
        "algorithm": "sac",
        "total_timesteps": 5000,  # Short training for testing
        "data_source": "csv",
        "data_path": "btc_jpy_real_dataset.csv",
        "sac_hyperparameters": {
            "learning_rate": 0.0003,
            "buffer_size": 10000,
            "learning_starts": 100,
            "batch_size": 64,
            "tau": 0.005,
            "gamma": 0.99,
            "train_freq": 1,
            "gradient_steps": 1,
            "ent_coef": 0.1,  # Higher entropy for exploration
            "target_update_interval": 1,
            "target_entropy": -1.0,
        },
        "environment": {
            "initial_balance": 200000,
            "transaction_cost": 0.0,
            "max_position_size": 0.01,
            "reward_scaling": 1.0,
            "reward_clip_value": 1.0,
            "reward_settings": {
                "use_simple_reward": True,
                "reward_scale": 1.0,
                "reward_clip_min": -1.0,
                "reward_clip_max": 1.0,
                "buy_action_penalty": 0.0,
                "sell_action_penalty": 0.0,
                "hold_action_penalty": 0.0,
                "profit_bonus_multipliers": [1.0, 1.0, 1.0],
            },
        },
        "checkpoint_interval": 1000,
    }

    config_manager = ConfigManager(config)
    trainer = AlgorithmTrainer(config_manager, progress_bar_enabled=True)

    print("Starting minimal SAC training...")
    result = trainer.train("sac", config)

    if result and result.get("success"):
        print("✅ Minimal training completed successfully!")
        model_path = result.get("model_path")
        print(f"Model saved to: {model_path}")

        # Test the trained model
        test_minimal_model(model_path)

        return model_path
    else:
        print("❌ Minimal training failed!")
        return None


def test_minimal_model(model_path):
    """Test the minimally trained model."""

    print("\n" + "=" * 80)
    print("TESTING MINIMAL SAC MODEL")
    print("=" * 80)

    # Load model
    model = SAC.load(model_path)

    # Load data
    df = pd.read_csv("btc_jpy_real_dataset.csv")
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values("timestamp").reset_index(drop=True)

    # Test action distribution
    actions = []
    for i in range(1000):
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

    print("Action Distribution (1000 samples):")
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

    if buy_count == total:
        print("❌ Still constant BUY bias!")
    elif sell_count == total:
        print("❌ SELL bias detected!")
    elif abs(buy_count - sell_count) < total * 0.1:  # Within 10%
        print("✅ Balanced action distribution achieved!")
    else:
        print("⚠️ Some bias still present")


def main():
    """Main test function."""

    # First test the reward function symmetry
    reward_results = test_minimal_reward_function()

    # Then train and test minimal SAC
    model_path = train_minimal_sac()

    # Save results
    results = {
        "reward_function_test": reward_results,
        "minimal_training": {
            "model_path": model_path,
            "success": model_path is not None,
        },
    }

    with open("results/minimal_reward_test.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print("\nResults saved to: results/minimal_reward_test.json")


if __name__ == "__main__":
    main()
