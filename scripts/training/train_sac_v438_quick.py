#!/usr/bin/env python3
"""
Quick SAC v438 Training - Bear Market Enhanced

Fast training script for SAC v438 with improved bear market performance.
"""

import sys
from pathlib import Path
import json

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from ztb.training.trainers.sac_trainer import SACAlgorithmTrainer
from ztb.training.core.config_manager import ConfigManager
from ztb.utils.logging_utils import get_logger

logger = get_logger(__name__)


def create_v438_unified_config():
    """Create unified config for v438 bear market enhancements."""

    # Load base reward config
    reward_config_path = project_root / "config" / "sac_v438_reward_config.json"
    with open(reward_config_path, 'r') as f:
        reward_config = json.load(f)

    # Create unified config for quick training
    unified_config = {
        "version": "1.0",
        "training": {
            "model_name": "sac_v438_bear_quick",
            "algorithm": "sac",
            "total_timesteps": 100000,  # Quick training
            "data_path": str(project_root / "data" / "btc_jpy_real_dataset.csv"),
            "environment": {
                "initial_balance": 100000.0,
                "transaction_cost": 0.0015,
                "max_position_size": 0.1,
                "enable_action_masking": False,  # SAC doesn't support action masking
                "use_continuous_actions": True,
                "use_standardized_observations": True,
                "random_start": True,
                "feature_engineering": "sac_v427",  # Uses enhanced features with bear market additions
                "reward_settings": reward_config["reward_function"]
            },
            "sac_hyperparameters": {
                "learning_rate": 3e-4,
                "buffer_size": 50000,  # Smaller buffer for quick training
                "learning_starts": 1000,
                "batch_size": 256,
                "tau": 0.005,
                "gamma": 0.99,
                "ent_coef": 0.01,
                "target_update_interval": 1,
                "target_entropy": -2.0
            },
            "metrics_log_interval": 100,
            "checkpoint_interval": 10000
        },
        "evaluation": {
            "thresholds": {
                "re_evaluate": 0.05,
                "monitor": 0.01
            },
            "min_samples": 10000,
            "risk_metrics": ["sharpe", "sortino", "max_drawdown"],
            "performance_metrics": ["total_return", "win_rate", "profit_factor"]
        },
        "logging": {
            "level": "INFO",
            "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        }
    }

    return unified_config


def train_v438_quick():
    """Quick training for v438 bear market enhancement."""

    logger.info("🐻 Starting SAC v438 quick training - Bear Market Enhanced")

    # Create unified config
    unified_config = create_v438_unified_config()

    # Create config manager and trainer
    config_manager = ConfigManager(unified_config)
    trainer = SACAlgorithmTrainer(config_manager=config_manager)

    # Train model using unified config
    logger.info("🎯 Training SAC v438 with bear market enhancements...")
    training_result = trainer.train(unified_config)

    logger.info("✅ SAC v438 quick training completed!")
    logger.info(f"📈 Training result: {training_result}")

    return training_result


if __name__ == "__main__":
    train_v438_quick()
