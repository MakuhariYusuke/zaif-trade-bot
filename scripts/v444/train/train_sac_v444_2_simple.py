#!/usr/bin/env python3
"""
SAC v444.2 Simple Training Script - 5000 Steps with Checkpoint Support

Simple training script for v444.2 configuration with checkpoint resumption.
Uses existing assets and provides continuation capability.
"""

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

# Add project root to path
project_root = get_project_root()
sys.path.insert(0, str(project_root))

# Add ztb package to path
ztb_path = project_root / "ztb"
sys.path.insert(0, str(ztb_path))

from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback
from ztb.trading.environment.heavy_env.core import HeavyTradingEnv
from ztb.utils.constants import DEFAULT_SEED
from ztb.utils.file_utils import get_project_root
from ztb.utils.logging_utils import setup_logging
from ztb.utils.training_utils import display_training_complete, save_model

setup_logging()
logger = logging.getLogger(__name__)


class ActionAverageCallback(BaseCallback):
    """
    Callback to calculate and log the average of continuous actions during training.
    """

    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.actions = []

    def _on_step(self) -> bool:
        # Collect actions from the current step
        actions = self.locals.get("actions")
        if actions is not None:
            # Assuming single environment, take the first action
            self.actions.append(float(actions[0]))
        return True

    def _on_training_end(self):
        if self.actions:
            avg_action = np.mean(self.actions)
            print(f"📊 Average continuous action value: {avg_action:.4f}")
        else:
            print("⚠️ No actions collected for averaging")


def load_config(config_path: str) -> dict:
    """Load configuration from JSON file."""
    with open(config_path, "r", encoding="utf-8") as f:
        return json.load(f)


def create_environment(config: dict):
    """Create trading environment from configuration."""
    # Create sample data for testing
    np.random.seed(DEFAULT_SEED)
    dates = pd.date_range("2023-01-01", periods=10000, freq="1h")

    # Generate realistic price data
    base_price = 5000000
    trend = np.sin(np.arange(10000) * 0.001) * 0.05  # Long-term trend
    noise = np.random.normal(0, 0.002, 10000)  # Short-term noise
    volatility = np.random.normal(0, 0.005, 10000)  # Volatility

    price_changes = trend + noise + volatility
    close = pd.Series(base_price * (1 + price_changes.cumsum()), index=dates)

    # Generate OHLCV
    high = close * (1 + np.abs(np.random.normal(0, 0.001, 10000)))
    low = close * (1 - np.abs(np.random.normal(0, 0.001, 10000)))
    open_price = close.shift(1).fillna(close.iloc[0])
    volume = pd.Series(np.random.uniform(1000, 10000, 10000), index=dates)

    df = pd.DataFrame(
        {
            "timestamp": dates,
            "open": open_price,
            "high": high,
            "low": low,
            "close": close,
            "volume": volume,
        }
    )

    # Create environment with minimal config (disable complex features for testing)
    env_config = {
        "initial_balance": config.get("environment", {}).get(
            "initial_balance", 200000.0
        ),
        "transaction_cost": config.get("environment", {}).get(
            "transaction_cost", 0.001
        ),
        "max_position_size": config.get("environment", {}).get(
            "max_position_size", 1.0
        ),
        "random_start": True,
        "feature_set": "v435_risk_managed_no_multi_timeframe",  # Use feature set without multi-timeframe to avoid dependencies
        "use_continuous_actions": True,  # Match checkpoint model's action space
        "include_regime_features": False,
        "include_correlation_features": False,
        "include_ensemble_features": False,
        "include_risk_features": False,
        "include_multi_timeframe_features": False,
        "curriculum_stage": config.get("training", {})
        .get("curriculum_learning", {})
        .get("curriculum_stage", "stability_optimized"),
    }

    env = HeavyTradingEnv(df=df, config=env_config, random_start=True)

    return env


def main():
    """Main training function."""
    parser = argparse.ArgumentParser(description="SAC v444.2 Simple Training")
    parser.add_argument(
        "--config",
        type=str,
        default="config/sac_v444_2_integrated_regime_adaptation_config.json",
        help="Path to configuration file",
    )
    parser.add_argument(
        "--resume-from",
        type=str,
        default=None,
        help="Path to checkpoint file to resume from",
    )
    parser.add_argument(
        "--total-timesteps", type=int, default=5000, help="Total timesteps to train"
    )
    parser.add_argument(
        "--checkpoint-freq", type=int, default=1000, help="Checkpoint save frequency"
    )

    args = parser.parse_args()

    try:
        print("🚀 SAC v444.2 Simple Training")
        print(f"Config: {args.config}")
        print(f"Total timesteps: {args.total_timesteps}")
        if args.resume_from:
            print(f"Resuming from: {args.resume_from}")

        # Load configuration
        config = load_config(args.config)
        print("✅ Configuration loaded")

        # Create environment
        env = create_environment(config)
        print("✅ Environment created")

        # SAC hyperparameters from config
        sac_config = config.get("training", {}).get("sac_hyperparameters", {})
        learning_rate = sac_config.get("learning_rate", 0.0003)
        buffer_size = sac_config.get("buffer_size", 1000000)
        learning_starts = sac_config.get("learning_starts", 1000)
        batch_size = sac_config.get("batch_size", 256)
        tau = sac_config.get("tau", 0.005)
        gamma = sac_config.get("gamma", 0.99)
        ent_coef = sac_config.get("ent_coef", "auto_1.0")
        target_entropy = sac_config.get("target_entropy", "auto")

        # Create SAC model
        if args.resume_from:
            print(f"Loading model from {args.resume_from}")
            model = SAC.load(args.resume_from, env=env)
            print("✅ Model loaded from checkpoint")
        else:
            model = SAC(
                "MlpPolicy",
                env,
                learning_rate=learning_rate,
                buffer_size=buffer_size,
                learning_starts=learning_starts,
                batch_size=batch_size,
                tau=tau,
                gamma=gamma,
                ent_coef=ent_coef,
                target_entropy=target_entropy,
                verbose=1,
                device="auto",
            )
            print("✅ New SAC model created")

        # Setup checkpoint callback
        checkpoint_callback = CheckpointCallback(
            save_freq=args.checkpoint_freq,
            save_path="models/sac_v444_2_checkpoints/",
            name_prefix="sac_v444_2",
            save_replay_buffer=True,
            save_vecnormalize=True,
        )

        # Setup action average callback
        action_callback = ActionAverageCallback()

        # Train the model
        print(f"🎯 Starting training for {args.total_timesteps} timesteps...")
        training_start_time = time.time()
        model.learn(
            total_timesteps=args.total_timesteps,
            callback=[checkpoint_callback, action_callback],
            reset_num_timesteps=False,  # Continue from checkpoint if resuming
        )
        training_time = time.time() - training_start_time

        # Save final model using centralized utility
        final_model_path = "models/sac_v444_2_final_model.zip"
        save_model(model, final_model_path)

        # Display completion using centralized utility
        final_metrics = {
            "total_timesteps": args.total_timesteps,
            "model_path": final_model_path,
            "action_average": np.mean(action_callback.actions)
            if action_callback.actions
            else 0.0,
        }
        display_training_complete(final_metrics, training_time)

        return True

    except Exception as e:
        logger.error(f"Training failed: {e}")
        import traceback

        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
