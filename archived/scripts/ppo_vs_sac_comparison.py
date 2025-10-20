#!/usr/bin/env python3
"""
PPO vs SAC Comparison - Testing Different RL Algorithms

Tests PPO with the same neutral reward function to see if the bias is SAC-specific.
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from ztb.trading.environment.heavy_env.core import HeavyTradingEnv
from ztb.trading.environment.utils.config import EnvironmentConfig


def create_neutral_env():
    """Create environment with perfectly neutral reward function."""
    config = EnvironmentConfig(
        max_position_size=0.01,
        transaction_cost=0.0,
        reward_scaling=1.0,
        reward_clip_value=1.0,
        reward_settings={
            "use_simple_reward": True,
            "reward_scale": 1.0,
            "reward_clip_min": -1.0,
            "reward_clip_max": 1.0,
            "buy_action_penalty": 0.0,
            "sell_action_penalty": 0.0,
            "hold_action_penalty": 0.0,
            "profit_bonus_multipliers": [1.0, 1.0, 1.0],
        },
    )

    df = pd.read_csv("btc_jpy_real_dataset.csv")
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values("timestamp").reset_index(drop=True)

    env = HeavyTradingEnv(df=df, config=config)
    return env


def train_ppo():
    """Train PPO with neutral reward function."""

    print("=" * 80)
    print("TRAINING PPO WITH NEUTRAL REWARD FUNCTION")
    print("=" * 80)

    from ztb.training.core.algorithm_trainer import AlgorithmTrainer
    from ztb.training.core.config_manager import ConfigManager

    config = {
        "model_name": "ppo_v408_neutral_reward",
        "algorithm": "ppo",
        "total_timesteps": 5000,
        "data_source": "csv",
        "data_path": "btc_jpy_real_dataset.csv",
        "ppo_hyperparameters": {
            "learning_rate": 0.0003,
            "n_steps": 2048,
            "batch_size": 64,
            "n_epochs": 10,
            "gamma": 0.99,
            "gae_lambda": 0.95,
            "clip_range": 0.2,
            "ent_coef": 0.01,
            "vf_coef": 0.5,
            "max_grad_norm": 0.5,
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

    print("Starting PPO training...")
    result = trainer.train("ppo", config)

    if result and result.get("success"):
        print("✅ PPO training completed successfully!")
        model_path = result.get("model_path")
        print(f"Model saved to: {model_path}")

        test_ppo_model(model_path)

        return model_path
    else:
        print("❌ PPO training failed!")
        return None


def test_ppo_model(model_path):
    """Test the trained PPO model."""

    print("\n" + "=" * 80)
    print("TESTING PPO MODEL")
    print("=" * 80)

    from stable_baselines3 import PPO

    model = PPO.load(model_path)

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

    print("Action Distribution (1000 samples):")
    print(f"Mean:   {np.mean(actions):.4f}")
    print(f"Std:    {np.std(actions):.4f}")
    print(f"Min:    {np.min(actions):.4f}")
    print(f"Max:    {np.max(actions):.4f}")
    print(f"Median: {np.median(actions):.4f}")

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
        print("❌ BUY bias detected!")
    elif sell_count == total:
        print("❌ SELL bias detected!")
    elif abs(buy_count - sell_count) < total * 0.2:  # Within 20%
        print("✅ Balanced action distribution achieved!")
    else:
        print("⚠️ Some bias still present")


def compare_sac_vs_ppo():
    """Compare SAC vs PPO results."""

    print("\n" + "=" * 80)
    print("SAC vs PPO COMPARISON")
    print("=" * 80)

    # Load previous SAC results
    try:
        with open("results/minimal_reward_test.json", "r", encoding="utf-8") as f:
            sac_results = json.load(f)
        print("SAC Results:")
        print(
            f"  BUY:  {sac_results['sac_action_distribution']['BUY']['count']} ({sac_results['sac_action_distribution']['BUY']['percentage']}%)"
        )
        print(
            f"  SELL: {sac_results['sac_action_distribution']['SELL']['count']} ({sac_results['sac_action_distribution']['SELL']['percentage']}%)"
        )
        print(
            f"  HOLD: {sac_results['sac_action_distribution']['HOLD']['count']} ({sac_results['sac_action_distribution']['HOLD']['percentage']}%)"
        )
    except:
        print("SAC results not found")

    print("\nTraining PPO now...")

    ppo_model_path = train_ppo()

    results = {
        "ppo_training": {
            "model_path": ppo_model_path,
            "success": ppo_model_path is not None,
        }
    }

    with open("results/ppo_vs_sac_comparison.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print("\nResults saved to: results/ppo_vs_sac_comparison.json")


if __name__ == "__main__":
    compare_sac_vs_ppo()
