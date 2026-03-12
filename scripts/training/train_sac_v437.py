#!/usr/bin/env python3
"""
SAC v427 Training Script

Enhanced SAC training with v427 features for improved trading performance.
Addresses v436 over-trading issues with better feature engineering and frequency control.
"""

import argparse
import os
import sys
from pathlib import Path
from typing import Optional

import pandas as pd

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv

from ztb.features.sac_v427_feature_engineering import SACv427FeatureEngineer
from ztb.trading.environment.heavy_env.core import HeavyTradingEnv
from ztb.utils.logging_utils import get_logger

logger = get_logger(__name__)


def create_v427_environment(
    config_path: Optional[str] = None, feature_set: str = "full"
):
    """
    Create trading environment with v427 features.

    Args:
        config_path: Path to configuration file
        feature_set: Feature set to use ('full', 'minimal', 'high_quality')

    Returns:
        Trading environment with v427 features
    """
    # Load configuration
    config = load_config("config/sac_v427_default_config.json") if config_path is None else load_config(config_path)

    # Load data
    data_path = config.get("data_path", "data/btc_jpy_real_dataset.csv")
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Data file not found: {data_path}")

    logger.info(f"Loading data from {data_path}")
    df = pd.read_csv(data_path)

    # Ensure required columns exist
    required_cols = ["timestamp", "open", "high", "low", "close"]
    if not all(col in df.columns for col in required_cols):
        raise ValueError(f"Data must contain columns: {required_cols}")

    # Convert timestamp
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.set_index("timestamp")

    # Initialize v427 feature engineer
    logger.info(f"Generating v427 features with {feature_set} set")
    feature_engineer = SACv427FeatureEngineer()

    # Generate v427 features with quality filtering
    features_df = feature_engineer.generate_v427_quality_filtered_features(df, feature_set=feature_set)

    logger.info(
        f"Generated {len(features_df.columns)} features for {len(features_df)} samples after quality filtering"
    )

    # Detailed feature count logging
    total_features = len(features_df.columns)
    original_cols = len(df.columns)
    generated_features = total_features - original_cols
    padding_features = len([col for col in features_df.columns if col.startswith('padding')])
    real_features = generated_features - padding_features

    logger.info("Feature breakdown:")
    logger.info(f"  - Original OHLCV columns: {original_cols}")
    logger.info(f"  - Generated features: {generated_features}")
    logger.info(f"  - Real quality-filtered features: {real_features}")
    logger.info(f"  - Padding features: {padding_features}")
    logger.info(f"  - Total feature dimensions: {total_features}")

    # Verify target dimensions
    if total_features != 110:
        logger.warning(f"Expected 110 features, got {total_features}. Check feature generation.")

    # Verify data integrity
    if features_df["close"].isna().any() or (features_df["close"] == 0).any():
        raise ValueError("Invalid price data detected in features")

    env = HeavyTradingEnv(
        df=features_df,
        config=config["environment"],
        random_start=config["environment"]["random_start"],
    )

    return env


def train_v437_sac(
    total_timesteps: int = 100000,
    model_save_path: str = "models/v437",
    log_path: str = "tensorboard/v437",
    config_path: Optional[str] = None,
    feature_set: str = "full",
    eval_freq: int = 5000,
    save_freq: int = 10000,
):
    """
    Train SAC v427 model.

    Args:
        total_timesteps: Total training timesteps
        model_save_path: Path to save model checkpoints
        log_path: Path for tensorboard logs
        config_path: Path to configuration file
        feature_set: Feature set to use
        eval_freq: Evaluation frequency
        save_freq: Model save frequency
    """
    logger.info("Starting SAC v427 training")

    # Create environment
    env = create_v427_environment(config_path, feature_set)
    env = Monitor(env, log_path)
    env = DummyVecEnv([lambda: env])

    # Log feature dimensions from environment
    obs_space = env.observation_space
    logger.info(f"Environment observation space: {obs_space}")
    if hasattr(obs_space, 'shape'):
        logger.info(f"Feature dimensions: {obs_space.shape[0]}")
        if obs_space.shape[0] != 110:
            logger.warning(f"Expected 110 feature dimensions, got {obs_space.shape[0]}")

            # Get actual observation to verify
            try:
                sample_obs, _ = env.reset()
                actual_dim = len(sample_obs[0]) if isinstance(sample_obs, tuple) else len(sample_obs)
                logger.info(f"Actual observation dimension from reset(): {actual_dim}")
            except Exception as e:
                logger.error(f"Could not get sample observation: {e}")

    # Create evaluation environment
    eval_env = create_v427_environment(config_path, feature_set)
    eval_env = Monitor(eval_env, log_path + "_eval")

    # Load configuration
    config = load_config("config/sac_v427_default_config.json") if config_path is None else load_config(config_path)
    sac_config = config["sac_hyperparameters"]

    # Create SAC model
    model = SAC(
        "MlpPolicy",
        env,
        learning_rate=sac_config["learning_rate"],
        buffer_size=sac_config["buffer_size"],
        learning_starts=sac_config["learning_starts"],
        batch_size=sac_config["batch_size"],
        tau=sac_config["tau"],
        gamma=sac_config["gamma"],
        ent_coef=sac_config["ent_coef"],
        target_update_interval=sac_config["target_update_interval"],
        target_entropy=sac_config["target_entropy"],
        tensorboard_log=log_path,
        verbose=1,
    )

    # Create callbacks
    checkpoint_callback = CheckpointCallback(
        save_freq=save_freq, save_path=model_save_path, name_prefix="sac_v427"
    )

    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=model_save_path,
        log_path=log_path,
        eval_freq=eval_freq,
        deterministic=True,
        render=False,
    )

    # Train model
    logger.info(f"Training SAC v437 for {total_timesteps} timesteps")
    model.learn(
        total_timesteps=total_timesteps,
        callback=[checkpoint_callback, eval_callback],
        progress_bar=True,
    )

    # Save final model
    final_model_path = os.path.join(model_save_path, "sac_v427_final")
    model.save(final_model_path)
    logger.info(f"Training completed. Final model saved to {final_model_path}")

    return model


def load_config(config_path: str):
    """Load configuration from JSON file."""
    import json

    with open(config_path, "r") as f:
        return json.load(f)


def main():
    parser = argparse.ArgumentParser(description="Train SAC v427 model")
    parser.add_argument(
        "--timesteps", type=int, default=100000, help="Total training timesteps"
    )
    parser.add_argument(
        "--model-path", type=str, default="models/v427", help="Path to save model"
    )
    parser.add_argument(
        "--log-path",
        type=str,
        default="tensorboard/v427",
        help="Path for tensorboard logs",
    )
    parser.add_argument(
        "--config", type=str, default=None, help="Path to configuration file"
    )
    parser.add_argument(
        "--feature-set",
        type=str,
        default="full",
        choices=["full", "minimal", "high_quality"],
        help="Feature set to use",
    )
    parser.add_argument(
        "--eval-freq", type=int, default=5000, help="Evaluation frequency"
    )
    parser.add_argument(
        "--save-freq", type=int, default=10000, help="Model save frequency"
    )

    args = parser.parse_args()

    # Create directories
    os.makedirs(args.model_path, exist_ok=True)
    os.makedirs(args.log_path, exist_ok=True)

    # Train model
    train_v437_sac(
        total_timesteps=args.timesteps,
        model_save_path=args.model_path,
        log_path=args.log_path,
        config_path=args.config,
        feature_set=args.feature_set,
        eval_freq=args.eval_freq,
        save_freq=args.save_freq,
    )


if __name__ == "__main__":
    main()
