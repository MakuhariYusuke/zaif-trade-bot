#!/usr/bin/env python3
"""
SAC v439 Training Script - Aggressive Scalping Model

Enhanced SAC model with scalping-optimized settings for high-frequency trading.
Features action signal guidance and reduced trading barriers for active trading.
"""

import argparse
import json
import logging
import os
import sys
from pathlib import Path
from datetime import datetime

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

import pandas as pd
import numpy as np
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines3.common.vec_env import DummyVecEnv

from ztb.training.environments.heavy_trading_env import HeavyTradingEnv
from ztb.training.environments.environment_config import EnvironmentConfig
from ztb.utils.logging_utils import get_logger

logger = get_logger(__name__)


def load_config(config_path: str) -> dict:
    """Load configuration from JSON file."""
    with open(config_path, 'r') as f:
        return json.load(f)


def create_environment(config: dict, data_path: str):
    """Create optimized scalping environment."""
    # Load data
    df = pd.read_csv(data_path)
    logger.info(f"Loaded {len(df)} rows of market data")

    # Feature columns for scalping (exclude multi-timeframe)
    feature_columns = []
    if "features" in config:
        for category in ["technical_indicators", "price_features", "volatility_features"]:
            if category in config["features"]:
                feature_columns.extend(config["features"][category])

    # Remove duplicates
    feature_columns = list(dict.fromkeys(feature_columns))
    logger.info(f"Using {len(feature_columns)} features for scalping: {feature_columns[:5]}...")

    environment_settings = config["environment"]
    scalping_settings = environment_settings.get("scalping_optimization", {})
    signal_guidance_settings = environment_settings.get("signal_guidance", {})

    reward_config = config.get("reward_function", {})
    reward_settings = {
        "base_profit_bonus_atr_coeff": reward_config.get("base_profit_bonus_atr_coeff", 5.0),
        "base_profit_bonus_portfolio_coeff": reward_config.get("base_profit_bonus_portfolio_coeff", 10.0),
        "base_action_penalty": reward_config.get("base_action_penalty", 0.02),
        "loss_penalty_coeff": reward_config.get("loss_penalty_coeff", -1.0),
        "action_frequency_penalty": reward_config.get("action_frequency_penalty", 0.005),
        "long_short_asymmetry": reward_config.get("long_short_asymmetry", True),
        "risk_adjusted_bonus": reward_config.get("risk_adjusted_bonus", True),
        "market_regime_penalty": reward_config.get("market_regime_penalty", True),
        "scalping_mode": reward_config.get("scalping_mode", True),
        "signal_guidance_integration": reward_config.get("signal_guidance_integration", True),
        "use_simple_reward": reward_config.get("use_simple_reward", False),
        "target_action_rate": reward_config.get(
            "target_action_rate", scalping_settings.get("target_action_rate", 0.55)
        ),
        "low_activity_penalty_scale": reward_config.get("low_activity_penalty_scale", 0.05),
        "overtrade_threshold": reward_config.get("overtrade_threshold", 0.95),
        "overtrade_penalty_scale": reward_config.get("overtrade_penalty_scale", 0.01),
        "hold_penalty_multiplier": reward_config.get(
            "hold_penalty_multiplier", scalping_settings.get("hold_penalty_multiplier", 1.2)
        ),
    }

    # Environment configuration optimized for scalping
    env_config = EnvironmentConfig(
        initial_balance=environment_settings["initial_balance"],
        max_steps=environment_settings["max_steps"],
        commission=environment_settings["commission"],
        slippage=environment_settings["slippage"],
        max_position_size=environment_settings["max_position_size"],
        min_trade_size=environment_settings.get("min_trade_size", 1e-5),  # Reduced for scalping
        min_position_change=scalping_settings.get(
            "min_position_change", environment_settings.get("min_trade_size", 1e-5)
        ),
        reward_scaling=environment_settings["reward_scaling"],
        feature_names=feature_columns + ["balance_norm", "position", "unrealized_norm"],  # Align with observation layout
        curriculum_stage=environment_settings.get("curriculum_stage", "pnl_focused"),
        continuous_to_discrete_threshold=scalping_settings.get("action_threshold", 0.02),
        continuous_to_discrete_threshold_neg=scalping_settings.get("negative_action_threshold"),
        signal_guidance_enabled=signal_guidance_settings.get("enabled", True),
        signal_guidance=signal_guidance_settings,
        scalping_optimization=scalping_settings,
    )

    # Create environment with scalping optimizations
    env = HeavyTradingEnv(
        data=df,
        config=env_config,
        feature_columns=feature_columns,
        reward_settings=reward_settings,
    )

    neg_threshold = (
        env_config.continuous_to_discrete_threshold_neg
        if env_config.continuous_to_discrete_threshold_neg is not None
        else -env_config.continuous_to_discrete_threshold
    )
    logger.info(
        "Scalping thresholds: action=%.4f / neg=%.4f / min_position_change=%.6f / min_trade=%.6f",
        env_config.continuous_to_discrete_threshold,
        neg_threshold,
        env_config.min_position_change,
        env_config.min_trade_size,
    )
    if signal_guidance_settings.get("enabled", True):
        logger.info(
            "Signal guidance: level=%s threshold=%.2f bonus=%.2f penalty=%.2f",
            signal_guidance_settings.get("guidance_level", "moderate"),
            signal_guidance_settings.get("signal_strength_threshold", 0.2),
            signal_guidance_settings.get("reward_bonus_multiplier", 0.0),
            signal_guidance_settings.get("action_penalty_multiplier", 0.0),
        )

    return env


def create_model(env, config: dict):
    """Create SAC model with scalping-optimized hyperparameters."""
    training_config = config["training"]

    model = SAC(
        "MlpPolicy",
        env,
        learning_rate=training_config["learning_rate"],
        batch_size=training_config["batch_size"],
        buffer_size=training_config["buffer_size"],
        learning_starts=training_config["learning_starts"],
        tau=training_config["tau"],
        gamma=training_config["gamma"],
        ent_coef=training_config["ent_coef"],
        target_entropy=training_config["target_entropy"],
        verbose=1,
        tensorboard_log=config["output"]["tensorboard_log"]
    )

    return model


def train_model(model, config: dict, output_dir: Path):
    """Train the model with scalping optimizations."""
    total_timesteps = config["training"]["total_timesteps"]

    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)

    # Callbacks for monitoring
    checkpoint_callback = CheckpointCallback(
        save_freq=max(total_timesteps // 10, 1000),
        save_path=str(output_dir / "checkpoints"),
        name_prefix="sac_v439"
    )

    logger.info(f"Starting training for {total_timesteps:,} timesteps...")
    logger.info("Scalping optimizations active: reduced thresholds, signal guidance enabled")

    # Train the model
    model.learn(
        total_timesteps=total_timesteps,
        callback=checkpoint_callback,
        progress_bar=True
    )

    # Save final model
    model_path = output_dir / "sac_v439_scalping_final.zip"
    model.save(model_path)
    logger.info(f"Model saved to {model_path}")

    return model_path


def main():
    parser = argparse.ArgumentParser(description="Train SAC v439 Scalping Model")
    parser.add_argument(
        "--config",
        type=str,
        default="config/v439/sac_v439_scalping_config.json",
        help="Path to configuration file"
    )
    parser.add_argument(
        "--data",
        type=str,
        default="data/btc_jpy_featured_dataset.csv",
        help="Path to market data CSV"
    )
    parser.add_argument(
        "--timesteps",
        type=int,
        default=None,
        help="Override total timesteps"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="models/v439",
        help="Output directory for model"
    )

    args = parser.parse_args()

    # Load configuration
    config = load_config(args.config)
    logger.info(f"Loaded configuration: {config['model_name']}")

    # Override timesteps if specified
    if args.timesteps:
        config["training"]["total_timesteps"] = args.timesteps
        logger.info(f"Overriding timesteps to {args.timesteps:,}")

    # Create environment
    env = create_environment(config, args.data)

    # Create model
    model = create_model(env, config)

    # Train model
    output_dir = Path(args.output)
    model_path = train_model(model, config, output_dir)

    # Save training metadata
    metadata = {
        "model_name": config["model_name"],
        "version": config["version"],
        "training_completed": datetime.now().isoformat(),
        "total_timesteps": config["training"]["total_timesteps"],
        "scalping_optimizations": {
            "action_threshold": config["environment"]["scalping_optimization"]["action_threshold"],
            "negative_action_threshold": config["environment"]["scalping_optimization"].get("negative_action_threshold"),
            "min_trade_size": config["environment"]["min_trade_size"],
            "min_position_change": config["environment"]["scalping_optimization"]["min_position_change"],
            "signal_guidance_enabled": config["environment"]["signal_guidance"]["enabled"],
            "target_action_rate": config["reward_function"]["target_action_rate"],
            "max_trades_per_episode": config["environment"]["scalping_optimization"]["max_trades_per_episode"]
        },
        "model_path": str(model_path),
        "config_path": args.config
    }

    metadata_path = output_dir / "training_metadata.json"
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2, default=str)

    logger.info("Training completed successfully!")
    logger.info(f"Model saved: {model_path}")
    logger.info(f"Metadata saved: {metadata_path}")


if __name__ == "__main__":
    main()



