#!/usr/bin/env python3
"""
Zero Reward Test - Ultimate SAC Bias Root Cause Investigation

Tests SAC with completely zero rewards to isolate if bias comes from reward function or SAC itself.
"""

import sys
import json
import numpy as np
import pandas as pd
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

def create_zero_reward_env():
    """Create environment with completely zero rewards."""
    from ztb.trading.environment.heavy_env.core import HeavyTradingEnv
    from ztb.trading.environment.utils.config import EnvironmentConfig

    config = EnvironmentConfig(
        max_position_size=0.01,
        transaction_cost=0.0,
        reward_scaling=1.0,
        reward_clip_value=1.0,
        reward_settings={
            "use_simple_reward": True,
            "reward_scale": 0.0,  # ZERO reward scale
            "reward_clip_min": 0.0,  # No negative rewards
            "reward_clip_max": 0.0,  # No positive rewards
            "buy_action_penalty": 0.0,
            "sell_action_penalty": 0.0,
            "hold_action_penalty": 0.0,
            "profit_bonus_multipliers": [0.0, 0.0, 0.0],  # ZERO bonuses
        }
    )

    df = pd.read_csv("btc_jpy_real_dataset.csv")
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df = df.sort_values('timestamp').reset_index(drop=True)

    env = HeavyTradingEnv(df=df, config=config)
    return env

def test_zero_reward_function():
    """Test that reward function returns zero for all actions."""

    print("=" * 80)
    print("ZERO REWARD FUNCTION TEST")
    print("=" * 80)

    env = create_zero_reward_env()

    test_cases = [
        {"action": 1, "pnl": 100.0, "position": 0.01, "old_position": 0.0, "desc": "BUY with profit"},
        {"action": 2, "pnl": 100.0, "position": 0.0, "old_position": 0.01, "desc": "SELL with profit"},
        {"action": 0, "pnl": 0.0, "position": 0.01, "old_position": 0.01, "desc": "HOLD neutral"},
        {"action": 1, "pnl": -50.0, "position": 0.01, "old_position": 0.0, "desc": "BUY with loss"},
        {"action": 2, "pnl": -50.0, "position": 0.0, "old_position": 0.01, "desc": "SELL with loss"},
    ]

    all_zero = True
    for case in test_cases:
        reward = env.reward_calculator.calculate_reward_simple(
            pnl=case["pnl"],
            portfolio_value=200000.0,
            position=case["position"],
            old_position=case["old_position"],
            action=case["action"],
            reward_history=[],
            portfolio_value_history=[200000.0] * 30
        )

        print(f"{case['desc']:20}: {reward:+.6f}")
        if abs(reward) > 1e-10:  # Allow tiny floating point errors
            all_zero = False

    if all_zero:
        print("✅ All rewards are ZERO - perfect neutral environment")
        return True
    else:
        print("❌ Some rewards are non-zero - reward function not neutral")
        return False

def train_sac_zero_reward():
    """Train SAC with zero rewards."""

    print("\n" + "=" * 80)
    print("TRAINING SAC WITH ZERO REWARDS")
    print("=" * 80)

    from ztb.training.core.algorithm_trainer import AlgorithmTrainer
    from ztb.training.core.config_manager import ConfigManager

    config = {
        "model_name": "sac_v408_zero_reward",
        "algorithm": "sac",
        "total_timesteps": 5000,
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
            "ent_coef": 0.1,
            "target_update_interval": 1,
            "target_entropy": -1.0
        },
        "environment": {
            "initial_balance": 200000,
            "transaction_cost": 0.0,
            "max_position_size": 0.01,
            "reward_scaling": 1.0,
            "reward_clip_value": 1.0,
            "reward_settings": {
                "use_simple_reward": True,
                "reward_scale": 0.0,
                "reward_clip_min": 0.0,
                "reward_clip_max": 0.0,
                "buy_action_penalty": 0.0,
                "sell_action_penalty": 0.0,
                "hold_action_penalty": 0.0,
                "profit_bonus_multipliers": [0.0, 0.0, 0.0],
            }
        },
        "checkpoint_interval": 1000,
    }

    config_manager = ConfigManager(config)
    trainer = AlgorithmTrainer(config_manager, progress_bar_enabled=True)

    print("Starting SAC training with zero rewards...")
    result = trainer.train("sac", config)

    if result and result.get("success"):
        print("✅ Zero reward training completed successfully!")
        model_path = result.get("model_path")
        print(f"Model saved to: {model_path}")

        test_zero_reward_model(model_path)

        return model_path
    else:
        print("❌ Zero reward training failed!")
        return None

def test_zero_reward_model(model_path):
    """Test SAC model trained with zero rewards."""

    print("\n" + "=" * 80)
    print("TESTING SAC MODEL TRAINED WITH ZERO REWARDS")
    print("=" * 80)

    from stable_baselines3 import SAC

    model = SAC.load(model_path)

    df = pd.read_csv("btc_jpy_real_dataset.csv")
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df = df.sort_values('timestamp').reset_index(drop=True)

    actions = []
    for i in range(1000):
        step = np.random.randint(100, len(df) - 100)
        obs = np.array([
            df.iloc[step]['close'],
            df.iloc[step]['volume'] if 'volume' in df.columns else 1000,
            0.0, 0.0, 0.0
        ], dtype=np.float32)

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

    if np.std(actions) < 0.01:  # Very low variance
        print("❌ Constant action output despite zero rewards!")
        print("This indicates a fundamental issue with SAC implementation or initialization")
        return "CONSTANT_BIAS"
    elif abs(buy_count - sell_count) < total * 0.2:
        print("✅ Balanced action distribution with zero rewards!")
        print("SAC can learn balanced policies when rewards are neutral")
        return "BALANCED"
    else:
        print("⚠️ Some bias present even with zero rewards")
        return "SOME_BIAS"

def main():
    """Main test function."""

    # First verify zero reward function
    if not test_zero_reward_function():
        print("❌ Cannot proceed - reward function is not zero")
        return

    # Train and test SAC with zero rewards
    model_path = train_sac_zero_reward()

    results = {
        "zero_reward_test": {
            "reward_function_neutral": True,
            "model_path": model_path,
            "success": model_path is not None
        }
    }

    with open("results/zero_reward_sac_test.json", 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"\nResults saved to: results/zero_reward_sac_test.json")

if __name__ == "__main__":
    main()