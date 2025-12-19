#!/usr/bin/env python3
"""
Improved 5000-step SAC training with exploration fixes
探索不足を修正した改良版5000ステップSAC学習
"""

import json
import logging
import sys
import time
from pathlib import Path

import gymnasium as gym
import numpy as np

from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import CheckpointCallback

from ztb.utils.constants import DEFAULT_PROGRESS_BAR, DEFAULT_SEED, DEFAULT_TOTAL_TIMESTEPS
from ztb.utils.file_utils import get_project_root
from ztb.utils.logging_utils import setup_logging

# Setup logging
setup_logging()
logger = logging.getLogger(__name__)

# Add project root to path
project_root = get_project_root()
sys.path.insert(0, str(project_root))


    # Track training statistics
    training_stats = {
        "total_timesteps": 5000,
        "environment": "improved_trading_env",
        "model_config": {
            "learning_rate": 3e-4,
            "buffer_size": 10000,
            "batch_size": 64,
            "net_arch": [64, 64],
            "ent_coef": 0.5,
            "learning_starts": 50,
        },
        "improvements": [
            "Lowered action thresholds (0.05 instead of 0.1)",
            "Added small penalty for inaction (-0.0001)",
            "Increased entropy coefficient (0.5) for more exploration",
            "Earlier learning start (50 steps)",
        ],
        "training_events": [],
    }

    # Train for 5000 steps
    logger.info("Starting improved training for 5000 steps...")
    training_start_time = time.time()
    try:
        model.learn(
            total_timesteps=DEFAULT_TOTAL_TIMESTEPS, callback=checkpoint_callback, progress_bar=DEFAULT_PROGRESS_BAR
        )
        training_time = time.time() - training_start_time
        logger.info("Training completed successfully")

        # Save final model using centralized utility
        model_path = "models/sac_improved_5000step_final.zip"
        from ztb.utils.training_utils import save_model
        save_model(model, model_path)
        logger.info(f"Model saved to {model_path}")

        # Display completion using centralized utility
        from ztb.utils.training_utils import display_training_complete
        final_metrics = {
            "total_timesteps": 5000,
            "model_path": model_path,
            "final_status": "success",
        }
        display_training_complete(final_metrics, training_time)

        # Update training stats
        training_stats.update(
            {
                "training_completed": True,
                "model_path": model_path,
                "final_status": "success",
            }
        )

    except Exception as e:
        logger.error(f"Training failed: {e}")
        training_stats.update(
            {"training_completed": False, "error": str(e), "final_status": "failed"}
        )

    # Save training stats
    stats_path = "analysis/training_stats_5000step_improved.json"
    with open(stats_path, "w") as f:
        json.dump(training_stats, f, indent=2)
    logger.info(f"Training stats saved to {stats_path}")

    # Print summary
    print("\n" + "=" * 50)
    print("IMPROVED 5000-STEP TRAINING SUMMARY")
    print("=" * 50)
    print(f"Status: {training_stats['final_status']}")
    print(f"Timesteps: {training_stats['total_timesteps']}")
    print(f"Model saved: {training_stats.get('model_path', 'N/A')}")
    print("\nImprovements applied:")
    for improvement in training_stats["improvements"]:
        print(f"  - {improvement}")
    print("=" * 50)


if __name__ == "__main__":
    main()
