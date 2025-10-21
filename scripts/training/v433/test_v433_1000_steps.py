#!/usr/bin/env python3
"""
SAC v433 Test Training - 1000 Steps Validation
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
    from ztb.trading.environment.utils.config import EnvironmentConfig, RewardSettings
    from ztb.utils.logging_utils import get_logger

    logger = get_logger(__name__)

except ImportError as e:
    print(f"Import error: {e}")
    print("Required packages not available. Please install dependencies.")
    sys.exit(1)


def create_v433_test_config():
    """Create v433 test configuration for 1000 steps."""

    config = {
        "version": "1.0",
        "training": {
            "model_name": "sac_v433_test_1000",
            "algorithm": "sac",
            "total_timesteps": 1000,
            "data_config": {
                "csv_path": "data/btc_jpy_real_dataset.csv",
                "use_real_data": True,
            },
            "sac_hyperparameters": {
                "learning_rate": 0.0003,
                "buffer_size": 100000,
                "batch_size": 256,
                "gamma": 0.99,
                "tau": 0.005,
                "ent_coef": "auto",
                "target_update_interval": 1,
                "gradient_steps": 1,
                "learning_starts": 1000,
                "use_sde": True,
                "use_sde_at_warmup": True,
                "sde_sample_freq": 4,
                "policy_kwargs": {
                    "net_arch": [400, 300]
                }
            }
        },
        "reward_function": {
            "sell_bonus": 0.4,
            "hold_bonus": -0.002,
            "buy_bonus": 0.4,
            "market_adaptive": {
                "sideways_multiplier": 2.5,
                "high_vol_multiplier": 1.1,
                "low_vol_multiplier": 1.0,
                "bull_multiplier": 1.6,
                "bear_multiplier": 1.6
            },
            "risk_penalty": 0.02,
            "time_penalty": 0.0003,
            "success_bonus": 0.4,
            "failure_penalty": 0.2
        },
        "action_thresholds": {
            "sell_threshold": -0.04,
            "buy_threshold": 0.04,
            "hold_range": [-0.04, 0.04],
            "adaptive_thresholds": True,
            "volatility_adjustment": True
        }
    }

    return config


class TrainingProgressCallback(BaseCallback):
    """Callback for monitoring training progress."""

    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.start_time = time.time()

    def _on_training_start(self):
        """Called at the beginning of training."""
        logger.info("🚀 Starting SAC v433 test training (1000 steps)")
        logger.info(f"Model: {self.model.__class__.__name__}")

    def _on_step(self) -> bool:
        """Called at each step."""
        if self.n_calls % 100 == 0:
            elapsed = time.time() - self.start_time
            logger.info(f"Step {self.n_calls}/1000 - Elapsed: {elapsed:.1f}s")

        return True

    def _on_training_end(self):
        """Called at the end of training."""
        elapsed = time.time() - self.start_time
        logger.info(f"✅ Training completed in {elapsed:.1f}s")


def load_data(csv_path):
    """Load and prepare data."""
    try:
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"Data file not found: {csv_path}")

        df = pd.read_csv(csv_path)
        logger.info(f"Loaded data: {len(df)} rows, {len(df.columns)} columns")
        return df

    except Exception as e:
        logger.error(f"Failed to load data: {e}")
        raise


def create_environment(config, data_df):
    """Create trading environment."""
    try:
        # Environment configuration
        env_config = EnvironmentConfig(
            reward_scaling=1.0,
            transaction_cost=0.0015,
            max_position_size=1.0,
            reward_position_penalty_scale=0.1,
            use_continuous_actions=True,
        )

        env = HeavyTradingEnv(df=data_df, config=env_config, random_start=True)
        env = Monitor(env)
        env = DummyVecEnv([lambda: env])

        logger.info("Environment created successfully")
        return env

    except Exception as e:
        logger.error(f"Failed to create environment: {e}")
        raise


def main():
    """Main training function."""
    try:
        logger.info("🤖 SAC v433 Test Training (1000 steps)")

        # Create configuration
        config = create_v433_test_config()
        logger.info("Configuration created")

        # Load data
        data_path = config["training"]["data_config"]["csv_path"]
        data_df = load_data(data_path)

        # Create environment
        env = create_environment(config, data_df)

        # Create model
        sac_params = config["training"]["sac_hyperparameters"]
        model = SAC(
            "MlpPolicy",
            env,
            verbose=1,
            **sac_params
        )

        # Create callback
        callback = TrainingProgressCallback()

        # Train model
        logger.info("Starting training...")
        start_time = time.time()

        model.learn(
            total_timesteps=config["training"]["total_timesteps"],
            callback=callback
        )

        training_time = time.time() - start_time
        logger.info(f"Training completed in {training_time:.1f} seconds")

        # Save model
        model_path = f"checkpoints/{config['training']['model_name']}"
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        model.save(model_path)
        logger.info(f"Model saved to {model_path}")

        # Verify model was saved
        if os.path.exists(f"{model_path}.zip"):
            logger.info(f"✅ Model verification: {model_path}.zip exists")
        else:
            logger.error(f"❌ Model verification failed: {model_path}.zip not found")

        logger.info("✅ SAC v433 test training completed successfully!")
        print("=" * 60)
        print("🎉 SUCCESS: SAC v433 1000-step test training completed!")
        print(f"   Model saved: {model_path}.zip")
        print(f"   Training time: {training_time:.1f} seconds")
        print("=" * 60)

    except Exception as e:
        logger.error(f"Training failed: {e}")
        raise


if __name__ == "__main__":
    main()