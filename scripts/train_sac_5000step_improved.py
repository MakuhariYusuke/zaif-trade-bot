#!/usr/bin/env python3
"""
Improved 5000-step SAC training with exploration fixes
探索不足を修正した改良版5000ステップSAC学習
"""

import json
import logging
import sys
from pathlib import Path

import gymnasium as gym
import numpy as np
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import CheckpointCallback

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Add project root to path
project_root = Path(__file__).resolve().parent.parent
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
    try:
        model.learn(
            total_timesteps=5000, callback=checkpoint_callback, progress_bar=True
        )
        logger.info("Training completed successfully")

        # Save final model
        model_path = "models/sac_improved_5000step_final.zip"
        model.save(model_path)
        logger.info(f"Model saved to {model_path}")

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
