#!/usr/bin/env python3
"""
Aggressive exploration 5000-step SAC training
積極的な探索を促進した5000ステップSAC学習
"""

import json
import logging
import sys
from pathlib import Path

import gymnasium as gym
import numpy as np
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import CheckpointCallback

from ztb.utils.constants import DEFAULT_CHECKPOINT_FREQ, DEFAULT_CHECKPOINT_PATH, DEFAULT_PROGRESS_BAR, DEFAULT_SEED, DEFAULT_TOTAL_TIMESTEPS
from ztb.training.constants import DEFAULT_BUFFER_SIZE_AGGRESSIVE, DEFAULT_LEARNING_RATE_AGGRESSIVE
from ztb.utils.logging_utils import setup_logging

# Setup logging
setup_logging()
logger = logging.getLogger(__name__)

# Add project root to path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))




def main():
    """Execute aggressive exploration 5000-step training"""
    logger.info("Starting aggressive exploration 5000-step SAC training...")

    # Create aggressive environment
    env = create_aggressive_trading_env()
    logger.info("Created aggressive exploration trading environment")

    # Create SAC model with maximum exploration parameters
    model = SAC(
        "MlpPolicy",
        env,
        learning_rate=DEFAULT_LEARNING_RATE_AGGRESSIVE,  # Higher learning rate
        buffer_size=DEFAULT_BUFFER_SIZE_AGGRESSIVE,
    print("=" * 50)
    print(f"Status: {training_stats['final_status']}")
    print(f"Timesteps: {training_stats['total_timesteps']}")
    print(f"Model saved: {training_stats.get('model_path', 'N/A')}")
    print("\nAggressive improvements applied:")
    for improvement in training_stats["aggressive_improvements"]:
        print(f"  - {improvement}")
    print("=" * 50)


if __name__ == "__main__":
    main()
