#!/usr/bin/env python3
"""
Direct SAC v428 Training Script

This script directly trains the SAC v428 model with optimized parameters,
bypassing the complex unified trainer imports.
"""

import os
import sys
import json
import logging
from pathlib import Path

# Set environment variables early
os.environ["PYTORCH_DISABLE_TORCH_DYNAMO"] = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["TORCH_USE_CUDA_DSA"] = "0"
os.environ["CUDA_LAUNCH_BLOCKING"] = "0"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:512"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

# Add project root to path
project_root = Path(__file__).resolve().parent
sys.path.insert(0, str(project_root))

def main():
    """Train SAC v428 with optimized parameters."""

    # Load config
    config_path = project_root / "configs" / "sac_v428_extended_backtest.json"
    with open(config_path, 'r') as f:
        config = json.load(f)

    print("=== SAC v428 Optimized Training ===")
    print(f"Model: {config['model_name']}")
    print(f"Total Timesteps: {config['total_timesteps']}")
    print("\nOptimized Hyperparameters:")
    for key, value in config['sac_hyperparameters'].items():
        print(f"  {key}: {value}")
    print(f"Reward Scale: {config['reward_settings']['reward_scale']}")

    # Import minimal required modules
    try:
        from stable_baselines3 import SAC
        from stable_baselines3.common.callbacks import CheckpointCallback
        import gymnasium as gym
        import numpy as np
        import torch
    except ImportError as e:
        print(f"Import error: {e}")
        return

    # Create environment
    try:
        env = gym.make('TradingEnv-v0', config=config)
    except Exception as e:
        print(f"Environment creation error: {e}")
        return

    # Create SAC model with optimized parameters
    sac_params = config['sac_hyperparameters'].copy()
    sac_params.pop('target_entropy', None)  # Remove if present

    model = SAC(
        'MlpPolicy',
        env,
        verbose=1,
        **sac_params
    )

    # Create checkpoint callback
    checkpoint_callback = CheckpointCallback(
        save_freq=1000,
        save_path=str(project_root / "checkpoints" / config['model_name']),
        name_prefix=config['model_name']
    )

    print("\nStarting training...")
    try:
        model.learn(
            total_timesteps=config['total_timesteps'],
            callback=checkpoint_callback
        )

        # Save final model
        model_path = project_root / "models" / f"{config['model_name']}.zip"
        model.save(str(model_path))

        print(f"\n✅ Training completed successfully!")
        print(f"Model saved as: {model_path}")

    except Exception as e:
        print(f"Training error: {e}")
        return

if __name__ == "__main__":
    main()