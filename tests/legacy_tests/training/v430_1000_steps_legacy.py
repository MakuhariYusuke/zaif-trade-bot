#!/usr/bin/env python3
"""
SAC v430 Test Training - 1000 Steps Validation
"""

import argparse
import json
import sys
from pathlib import Path

import pytest

# Add project root to path
project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(project_root))

try:
    from sac import SACSuite
except ModuleNotFoundError:
    pytest.skip(
        "legacy v430 SAC harness depends on removed 'sac' module; kept for archive only",
        allow_module_level=True,
    )

from ztb.utils.logging_utils import get_logger

logger = get_logger(__name__)


def create_test_config():
    """Create test configuration for 1000 steps."""

    # Create ZaifTradeBotConfig format
    config = {
        "version": "1.0",
        "training": {
            "model_name": "sac_v430_test",
            "algorithm": "sac",
            "total_timesteps": 1000,
            "data_config": {
                "csv_path": "btc_jpy_real_dataset.csv",
                "use_real_data": True,
            },
            "environment": {
                "initial_balance": 200000.0,
                "transaction_cost": 0.0005,
                "max_position_size": 0.01,
                "enable_action_masking": False,
                "use_continuous_actions": True,
                "use_standardized_observations": True,
                "reward_settings": {
                    "reward_scale": 140.26367385248548,
                    "trading_bonus": 0.0041079974127759735,
                    "sell_penalty": -0.35240053723313824,
                    "buy_bonus": -0.427338600085897,
                    "action_balance_weight": 0.270731511102946,
                    "hold_penalty": 0.0052929478390304745,
                    "profit_focus": False,
                    "risk_penalty": 0.0642814422601983,
                },
            },
            "sac_hyperparameters": {
                "learning_rate": 0.00016093166779077603,
                "gamma": 0.9796652702743582,
                "tau": 0.005,
                "ent_coef": 0.01,
                "target_entropy": -2.0,
                "batch_size": 128,
                "buffer_size": 50000,
                "learning_starts": 500,
                "gradient_steps": 1,
                "train_freq": [1, "step"],
                "target_update_interval": 1,
            },
        },
        "evaluation": {
            "thresholds": {"re_evaluate": 0.05, "monitor": 0.01},
            "min_samples": 10000,
        },
        "logging": {"level": "INFO"},
    }

    # Save test config
    test_config_path = "configs/v430/sac_v430_test_1000.json"
    with open(test_config_path, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2, ensure_ascii=False)

    logger.info(f"Test configuration saved to {test_config_path}")
    return test_config_path


def run_test_training():
    """Run 1000 steps test training."""

    print("🧪 SAC v430 Test Training - 1000 Steps")
    print("=" * 60)

    # Create test config
    config_path = create_test_config()

    print("📋 Test Configuration:")
    with open(config_path, "r", encoding="utf-8") as f:
        config = json.load(f)
    print(f"   Total timesteps: {config['training']['total_timesteps']}")
    print(
        f"   Learning rate: {config['training']['sac_hyperparameters']['learning_rate']:.6f}"
    )
    print(f"   Batch size: {config['training']['sac_hyperparameters']['batch_size']}")
    print(f"   Buffer size: {config['training']['sac_hyperparameters']['buffer_size']}")
    print(
        f"   Reward scale: {config['training']['environment']['reward_settings']['reward_scale']:.1f}"
    )
    print()

    # Create mock args for SACSuite
    args = argparse.Namespace()
    args.config = config_path
    args.timesteps = 1000
    args.parallel = False
    args.validate = False

    # Initialize SAC suite
    print("🚀 Initializing SAC suite...")
    sac_suite = SACSuite()

    # Run training
    print("🎯 Starting 1000 steps test training...")
    import time

    start_time = time.time()

    success = sac_suite.run_train(args) == 0

    training_time = time.time() - start_time

    print()
    print("=" * 60)
    if success:
        print("✅ Test training completed successfully!")
        print(f"⏱️  Training time: {training_time:.2f} seconds")
        print("📁 Model saved to: models/sac_v430_test/final_model.zip")

        # Check if model file exists
        model_path = Path("models/sac_v430_test/final_model.zip")
        if model_path.exists():
            print("📊 Model file verified: EXISTS")
        else:
            print("⚠️  Model file not found - check logs for issues")

    else:
        print("❌ Test training failed!")
        print("Check logs for error details")

    print("=" * 60)

    return success


def main() -> int:
    return 0 if run_test_training() else 1


if __name__ == "__main__":
    sys.exit(main())
