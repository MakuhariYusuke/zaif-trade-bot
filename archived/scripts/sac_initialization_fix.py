#!/usr/bin/env python3
"""
SAC Initialization Fix - Test Different Initialization Methods

Tests SAC with different initialization approaches to fix the constant output bias.
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))


def test_sac_initialization_fix():
    """Test SAC with different initialization methods."""

    print("=" * 80)
    print("SAC INITIALIZATION FIX TEST")
    print("=" * 80)

    # Test different random seeds
    seeds_to_test = [0, 42, 123, 999, 2024]

    results = {}

    for seed in seeds_to_test:
        print(f"\n--- Testing SAC with seed {seed} ---")

        # Set all random seeds
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)

        model_path = train_sac_with_seed(seed)
        if model_path:
            bias_result = test_sac_model_bias(model_path)
            results[f"seed_{seed}"] = {
                "model_path": model_path,
                "bias_result": bias_result,
            }

    print("\n" + "=" * 80)
    print("INITIALIZATION TEST RESULTS")
    print("=" * 80)

    for seed_name, result in results.items():
        print("12")

    # Check if any seed produced balanced results
    balanced_seeds = [k for k, v in results.items() if v["bias_result"] == "BALANCED"]
    if balanced_seeds:
        print(f"\n✅ Found balanced results with seeds: {balanced_seeds}")
        return True
    else:
        print("\n❌ All seeds still produce bias - initialization is not the issue")
        return False


def train_sac_with_seed(seed):
    """Train SAC with specific random seed."""

    from ztb.training.core.algorithm_trainer import AlgorithmTrainer
    from ztb.training.core.config_manager import ConfigManager

    config = {
        "model_name": f"sac_v409_seed_{seed}",
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
                "reward_scale": 0.0,  # Zero rewards for neutrality test
                "reward_clip_min": 0.0,
                "reward_clip_max": 0.0,
                "buy_action_penalty": 0.0,
                "sell_action_penalty": 0.0,
                "hold_action_penalty": 0.0,
                "profit_bonus_multipliers": [0.0, 0.0, 0.0],
            },
        },
        "checkpoint_interval": 1000,
    }

    config_manager = ConfigManager(config)
    trainer = AlgorithmTrainer(config_manager, progress_bar_enabled=True)

    print(f"Training SAC with seed {seed}...")
    result = trainer.train("sac", config)

    if result and result.get("success"):
        model_path = result.get("model_path")
        print(f"✅ Training completed: {model_path}")
        return model_path
    else:
        print("❌ Training failed")
        return None


def test_sac_model_bias(model_path):
    """Test if SAC model produces balanced actions."""

    from stable_baselines3 import SAC

    try:
        model = SAC.load(model_path)
    except Exception as e:
        print(f"❌ Failed to load model: {e}")
        return "LOAD_ERROR"

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

    if np.std(actions) < 0.01:
        return "CONSTANT_BIAS"
    elif abs(buy_count - sell_count) < total * 0.2:
        return "BALANCED"
    else:
        return "SOME_BIAS"


def test_different_sac_hyperparameters():
    """Test SAC with very different hyperparameters."""

    print("\n" + "=" * 80)
    print("TESTING SAC WITH DIFFERENT HYPERPARAMETERS")
    print("=" * 80)

    hyper_configs = [
        {
            "name": "high_entropy",
            "ent_coef": 1.0,
            "learning_rate": 0.001,
            "target_entropy": -0.5,
        },
        {
            "name": "low_entropy",
            "ent_coef": 0.01,
            "learning_rate": 0.0001,
            "target_entropy": -2.0,
        },
        {
            "name": "no_entropy",
            "ent_coef": 0.0,
            "learning_rate": 0.0003,
            "target_entropy": -1.0,
        },
    ]

    results = {}

    for config in hyper_configs:
        print(f"\n--- Testing {config['name']} ---")

        model_path = train_sac_with_hyperparams(config)
        if model_path:
            bias_result = test_sac_model_bias(model_path)
            results[config["name"]] = {
                "model_path": model_path,
                "bias_result": bias_result,
                "config": config,
            }

    print("\n" + "=" * 80)
    print("HYPERPARAMETER TEST RESULTS")
    print("=" * 80)

    for config_name, result in results.items():
        print("12")

    balanced_configs = [k for k, v in results.items() if v["bias_result"] == "BALANCED"]
    if balanced_configs:
        print(f"\n✅ Found balanced results with configs: {balanced_configs}")
        return True
    else:
        print("\n❌ All hyperparameter configs still produce bias")
        return False


def train_sac_with_hyperparams(hyper_config):
    """Train SAC with specific hyperparameters."""

    from ztb.training.core.algorithm_trainer import AlgorithmTrainer
    from ztb.training.core.config_manager import ConfigManager

    config = {
        "model_name": f"sac_v410_{hyper_config['name']}",
        "algorithm": "sac",
        "total_timesteps": 5000,
        "data_source": "csv",
        "data_path": "btc_jpy_real_dataset.csv",
        "sac_hyperparameters": {
            "learning_rate": hyper_config["learning_rate"],
            "buffer_size": 10000,
            "learning_starts": 100,
            "batch_size": 64,
            "tau": 0.005,
            "gamma": 0.99,
            "train_freq": 1,
            "gradient_steps": 1,
            "ent_coef": hyper_config["ent_coef"],
            "target_update_interval": 1,
            "target_entropy": hyper_config["target_entropy"],
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
            },
        },
        "checkpoint_interval": 1000,
    }

    config_manager = ConfigManager(config)
    trainer = AlgorithmTrainer(config_manager, progress_bar_enabled=True)

    print(f"Training SAC with {hyper_config['name']} config...")
    result = trainer.train("sac", config)

    if result and result.get("success"):
        model_path = result.get("model_path")
        print(f"✅ Training completed: {model_path}")
        return model_path
    else:
        print("❌ Training failed")
        return None


def main():
    """Main test function."""

    print("Testing SAC initialization fixes...")

    # Test different random seeds
    seed_success = test_sac_initialization_fix()

    # Test different hyperparameters
    hyper_success = test_different_sac_hyperparameters()

    if seed_success or hyper_success:
        print("\n🎉 SUCCESS: Found configuration that produces balanced SAC actions!")
    else:
        print(
            "\n💥 FAILURE: SAC has fundamental bias issue that cannot be fixed with initialization or hyperparameters"
        )
        print(
            "Recommendation: Switch to different RL algorithm (PPO) or investigate SAC implementation"
        )

    results = {
        "seed_test_success": seed_success,
        "hyperparameter_test_success": hyper_success,
        "overall_success": seed_success or hyper_success,
    }

    with open("results/sac_initialization_fix_test.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print("\nResults saved to: results/sac_initialization_fix_test.json")


if __name__ == "__main__":
    main()
