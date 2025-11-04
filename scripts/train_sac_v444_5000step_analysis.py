#!/usr/bin/env python3
"""
SAC v444.1 Training Script - 5000 Steps Analysis
課題発見のための5000ステップ学習を実行
"""

import os
import sys
import json
import logging
from pathlib import Path

# Add project root to path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from ztb.training.unified_trainer import UnifiedTrainer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    """SAC v444.1トレーニング実行 - 5000ステップ"""
    config_path = "config/default.yaml"

    # Load configuration
    try:
        with open(config_path, "r") as f:
            config = json.load(f)
    except FileNotFoundError:
        logger.error(f"Config file not found: {config_path}")
        return

    # Override training parameters for 5000 steps analysis
    config["training"]["total_timesteps"] = 5000
    config["training"]["model_name"] = "sac_v444_5000step_analysis"

    logger.info("Starting SAC v444.1 training with 5000 steps for issue analysis...")

    # Initialize UnifiedTrainer
    trainer = UnifiedTrainer(config=config)

    # Execute training
    try:
        stats = trainer.train()
        logger.info(f"Training completed. Stats: {stats}")

        # Save model
        model_path = "models/sac_v444_5000step_analysis.zip"
        trainer.save_model(model_path)
        logger.info(f"Model saved to {model_path}")

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