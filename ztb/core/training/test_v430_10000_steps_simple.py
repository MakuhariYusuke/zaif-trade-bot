#!/usr/bin/env python3
"""
SAC v430 Test Training - 10000 Steps Validation (Direct SB3)
"""

import json
import os
import sys
import time
from pathlib import Path

# Add project root to path
project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(project_root))

# Direct imports to avoid complex dependencies
try:
    import pandas as pd
    import torch
    from stable_baselines3 import SAC
    from stable_baselines3.common.callbacks import BaseCallback
    from stable_baselines3.common.monitor import Monitor
    from stable_baselines3.common.vec_env import DummyVecEnv

    # Import environment and data handling
    from ztb.trading.environment import HeavyTradingEnv
    from ztb.trading.environment.utils.config import EnvironmentConfig
    from ztb.utils.logging_utils import get_logger

    logger = get_logger(__name__)

except ImportError as e:
    print(f"Import error: {e}")
    print("Required packages not available. Please install dependencies.")
    sys.exit(1)


def create_test_config():
    """Create test configuration for 10000 steps."""

    config = {
        "version": "1.0",
        "training": {
            "model_name": "sac_v430_test_10k",
            "algorithm": "sac",
            "total_timesteps": 10000,  # Changed from 1000 to 10000
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
    }

    # Save test config
    test_config_path = "configs/v430/sac_v430_test_10000.json"
    with open(test_config_path, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2, ensure_ascii=False)

    logger.info(f"Test configuration saved to {test_config_path}")
    return test_config_path


class TrainingCallback(BaseCallback):
    """Callback for monitoring training progress."""

    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.start_time = time.time()

    def _on_step(self) -> bool:
        if self.n_calls % 1000 == 0:  # Changed from 250 to 1000 for 10k steps
            elapsed = time.time() - self.start_time
            print(f"Step {self.n_calls}: {elapsed:.1f}s elapsed")
        return True


def run_test_training():
    """Run 10000 steps test training."""

    print("🧪 SAC v430 Test Training - 10000 Steps")
    print("=" * 60)

    # Create test config
    config_path = create_test_config()

    # Load config
    with open(config_path, "r", encoding="utf-8") as f:
        config = json.load(f)

    training_config = config["training"]
    env_config = training_config["environment"]
    sac_params = training_config["sac_hyperparameters"]

    print("📋 Test Configuration:")
    print(f"   Total timesteps: {training_config['total_timesteps']}")
    print(f"   Learning rate: {sac_params['learning_rate']:.6f}")
    print(f"   Batch size: {sac_params['batch_size']}")
    print(f"   Buffer size: {sac_params['buffer_size']}")
    print(f"   Reward scale: {env_config['reward_settings']['reward_scale']:.1f}")
    print()

    try:
        # Load data
        print("📊 Loading trading data...")
        data_path = "data/btc_jpy_real_dataset.csv"
        df = pd.read_csv(data_path)
        print(f"Loaded {len(df)} rows of data")

        # Create environment config
        print("⚙️  Creating environment configuration...")
        env_config_obj = EnvironmentConfig(
            reward_scaling=env_config["reward_settings"]["reward_scale"],
            transaction_cost=env_config["transaction_cost"],
            max_position_size=env_config["max_position_size"],
            reward_position_penalty_scale=env_config["reward_settings"]["trading_bonus"],
            use_continuous_actions=True,  # Enable continuous actions for SAC
            use_standardized_observations=env_config["use_standardized_observations"],
            initial_portfolio_value=env_config["initial_balance"],
        )

        # Create environment
        print("🚀 Creating trading environment...")
        env = HeavyTradingEnv(df=df, config=env_config_obj, random_start=True)

        # Wrap environment
        env = Monitor(env)
        env = DummyVecEnv([lambda: env])

        # Create SAC model
        print("🤖 Creating SAC model...")
        model = SAC(
            "MlpPolicy",
            env,
            learning_rate=sac_params["learning_rate"],
            buffer_size=sac_params["buffer_size"],
            learning_starts=sac_params["learning_starts"],
            batch_size=sac_params["batch_size"],
            tau=sac_params["tau"],
            gamma=sac_params["gamma"],
            ent_coef=sac_params["ent_coef"],
            target_entropy=sac_params["target_entropy"],
            verbose=1,
            device="auto",
        )

        # Create callback
        callback = TrainingCallback()

        # Run training
        print("🎯 Starting 10000 steps test training...")
        start_time = time.time()

        model.learn(
            total_timesteps=training_config["total_timesteps"], callback=callback
        )

        training_time = time.time() - start_time

        print()
        print("=" * 60)
        print("✅ Test training completed successfully!")
        print(f"⏱️  Training time: {training_time:.2f} seconds")

        # Save model
        model_path = "models/sac_v430_test/final_model_10k.zip"
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        model.save(model_path)
        print(f"💾 Model saved to: {model_path}")

        # Check if model file exists
        if os.path.exists(model_path):
            print("📊 Model file verified: EXISTS")
        else:
            print("⚠️  Model file not found")

        print("=" * 60)
        return True

    except Exception as e:
        print()
        print("=" * 60)
        print("❌ Test training failed!")
        print(f"Error: {e}")
        import traceback

        traceback.print_exc()
        print("=" * 60)
        return False


def main():
    """Main function."""
    try:
        success = run_test_training()
        return 0 if success else 1
    except Exception as e:
        logger.error(f"Test training failed with error: {e}")
        import traceback

        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())