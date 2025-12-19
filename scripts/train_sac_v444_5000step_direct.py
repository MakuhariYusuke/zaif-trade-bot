#!/usr/bin/env python3
"""
SAC v444.1 Training Script - Direct SAC Trainer Version
課題発見のための5000ステップ学習を実行
"""

import sys
import json
import logging
from pathlib import Path
import yaml

# Add project root to path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from ztb.training.unified_trainer.algorithms.sac_trainer import SACTrainer
from ztb.utils.logging_utils import setup_logging

setup_logging()
logger = logging.getLogger(__name__)


def main():
    """SAC v444.1トレーニング実行 - 5000ステップ"""
    config_path = "config/default.yaml"

    # Load configuration
    try:
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)
    except FileNotFoundError:
        logger.error(f"Config file not found: {config_path}")
        return

    # Override training parameters for 5000 steps analysis
    config["training"]["total_timesteps"] = 5000
    config["training"]["model_name"] = "sac_v444_5000step_analysis"

    logger.info("Starting SAC v444.1 training with 5000 steps for issue analysis...")

    # Initialize SAC Trainer
    trainer = SACTrainer(config=config)

    # Execute training
    try:
        success = trainer.train(total_timesteps=5000)
        logger.info(f"Training completed. Success: {success}")

        # Get training stats
        stats = trainer.training_stats if hasattr(trainer, 'training_stats') else {}

        # Save training stats for analysis
        stats_path = "analysis/training_stats_5000step.json"
        with open(stats_path, "w") as f:
            json.dump(stats, f, indent=2)
        logger.info(f"Training stats saved to {stats_path}")

    except Exception as e:
        logger.error(f"Training failed: {e}")
        raise


if __name__ == "__main__":
    main()