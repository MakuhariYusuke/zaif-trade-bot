#!/usr/bin/env python3
"""
SAC v440 Training Script - Pure PnL-Based Model

Simplified SAC model with pure PnL-based reward function.
Focuses on basic profit/loss learning without complex incentives.
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

from ztb.trading.environment.utils.config import EnvironmentConfig, RewardSettings
from ztb.training.environments.heavy_trading_env import HeavyTradingEnv
from ztb.utils.logging_utils import get_logger

logger = get_logger(__name__)


def load_config(config_path: str) -> dict:
    """Load configuration from JSON file."""
    with open(config_path, 'r') as f:
        return json.load(f)


def create_environment(config: dict, data_path: str):
    """Create simplified PnL-focused environment."""
    # Load data
    df = pd.read_csv(data_path)
    logger.info(f"Loaded {len(df)} rows of market data")

    # Basic feature columns for PnL-focused trading
    feature_columns = []
    if "features" in config:
        for category in ["technical_indicators", "price_features", "volatility_features", "volume_features", "momentum_features", "trend_features", "oscillator_features", "support_resistance"]:
            if category in config["features"]:
                feature_columns.extend(config["features"][category])

    # Remove duplicates
    feature_columns = list(set(feature_columns))

    # Add standard features
    feature_columns.extend(["balance_norm", "position", "unrealized_norm"])

    logger.info(f"Using {len(feature_columns)} features: {feature_columns}")

    # Create reward settings for pure PnL
    reward_settings: RewardSettings = {
        "position_soft_cap": 1.0,
        "position_penalty_scale": 0.0,  # No position penalty for pure PnL
        "position_penalty_exp": 1.0,
        "inventory_window": 10,
        "inventory_penalty_scale": 0.0,  # No inventory penalty
        "trade_frequency_penalty": 0.0,  # No frequency penalty
        "trade_frequency_halflife": 50,
        "trade_cooldown_steps": 0,
        "trade_cooldown_penalty": 0.0,
        "max_consecutive_trades": 100,
        "consecutive_trade_penalty": 0.0,
        "volatility_window": 20,
        "volatility_penalty_scale": 0.0,  # No volatility penalty
        "sharpe_bonus_scale": 0.0,  # No Sharpe bonus
        "sortino_bonus_scale": 0.0,
        "calmar_bonus_scale": 0.0,
        "reward_clip_value": 10.0,
        "profit_bonus_multipliers": [1.0],  # Simple profit bonus
        "enable_forced_diversity": False,
        "custom_reward_params": {},
        "balance_penalty": 0.0,  # No balance penalty
        "balance_penalty_tolerance": 0.1,
        "profit_weight": config["reward_function"]["base_profit_bonus"],
        "risk_weight": config["reward_function"]["loss_penalty_coeff"],
        "consistency_weight": 0.0,
        "ultra_profit_multiplier": 1.0,
        "ultra_risk_multiplier": 1.0
    }

    # Create environment config (simplified)
    env_config = EnvironmentConfig(
        initial_portfolio_value=config["environment"]["initial_balance"],
        transaction_cost=config["environment"]["commission"],
        max_position_size=config["environment"]["max_position_size"],
        reward_scaling=config["environment"]["reward_scaling"],
        feature_names=feature_columns,
        curriculum_stage="pnl_focused",
        correlation_reduction=False,  # Disable for simplicity
        stop_loss_threshold=0.5,  # High threshold to avoid interference
        max_consecutive_trades=100,
        min_holding_period=1,
        reward_position_soft_cap=1.0,
        reward_position_penalty_scale=0.0,  # No position penalty
        reward_position_penalty_exponent=1.0,
        reward_inventory_window=10,
        reward_inventory_penalty_scale=0.0,  # No inventory penalty
        reward_trade_frequency_penalty=0.0,  # No frequency penalty
        reward_trade_frequency_halflife=50,
        reward_trade_cooldown_steps=0,
        reward_trade_cooldown_penalty=0.0,
        reward_max_consecutive_trades=100,
        reward_consecutive_trade_penalty=0.0,
        reward_volatility_window=20,
        reward_volatility_penalty_scale=0.0,  # No volatility penalty
        reward_sharpe_bonus_scale=0.0,  # No Sharpe bonus
        reward_clip_value=10.0,
        reward_profit_bonus_multipliers=[1.0],  # Simple profit bonus
        enable_forced_diversity=False
    )
    # Add missing attributes for compatibility
    env_config.initial_balance = config["environment"]["initial_balance"]
    env_config.max_steps = config["environment"]["max_steps"]
    env_config.slippage = config["environment"]["slippage"]
    env_config.commission = config["environment"]["commission"]

    # Create environment
    env = HeavyTradingEnv(df, env_config, feature_columns=feature_columns, reward_settings=reward_settings)
    logger.info("Environment created successfully")

    return env


def create_model(env, config: dict):
    """Create SAC model with basic settings."""
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
    """Train the model with PnL focus."""
    total_timesteps = config["training"]["total_timesteps"]

    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)

    # Callbacks for monitoring
    checkpoint_callback = CheckpointCallback(
        save_freq=max(total_timesteps // 10, 1000),
        save_path=str(output_dir / "checkpoints"),
        name_prefix="sac_v440"
    )

    logger.info(f"Starting training for {total_timesteps:,} timesteps...")
    logger.info("Pure PnL-based reward function - no complex incentives")

    # Train the model
    model.learn(
        total_timesteps=total_timesteps,
        callback=checkpoint_callback,
        progress_bar=True
    )

    # Save final model
    model_path = output_dir / "sac_v440_pnl_final.zip"
    model.save(model_path)
    logger.info(f"Model saved to {model_path}")

    return model_path


def main():
    parser = argparse.ArgumentParser(description="Train SAC v440 PnL-Focused Model")
    parser.add_argument(
        "--config",
        type=str,
        default="config/v440/sac_v440_pnl_config.json",
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
        default="models/v440",
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
        "approach": "pure_pnl_based",
        "reward_function": "simplified",
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