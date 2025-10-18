#!/usr/bin/env python3
"""
Train SAC v428 with Optimized Hyperparameters

This script trains the SAC v428 model using the optimized hyperparameters
from the hyperparameter optimization pipeline.
"""

import sys
import os
from pathlib import Path

# Add project root to path
project_root = Path(__file__).resolve().parent
sys.path.insert(0, str(project_root))

import json
import subprocess

def main():
    """Train SAC v428 with optimized parameters."""

    # Load optimized config
    config_path = "configs/sac_v428_extended_backtest.json"
    with open(config_path, 'r') as f:
        config = json.load(f)

    print("=== SAC v428 Optimized Training ===")
    print(f"Model: {config['model_name']}")
    print(f"Total Timesteps: {config['total_timesteps']}")
    print("\nOptimized Hyperparameters:")
    for key, value in config['sac_hyperparameters'].items():
        print(f"  {key}: {value}")
    print(f"Reward Scale: {config['reward_settings']['reward_scale']}")

    # Run the unified_trainer.py script directly
    print("\nStarting training...")
    unified_trainer_path = project_root / "ztb" / "training" / "unified_trainer.py"
    cmd = [sys.executable, "-m", "ztb.training.unified_trainer", "--config", config_path]
    result = subprocess.run(cmd, cwd=str(project_root))

    if result.returncode == 0:
        print("\n✅ Training completed successfully!")
        print(f"Model saved as: {config['model_name']}")
    else:
        print(f"\n❌ Training failed with return code: {result.returncode}")
        sys.exit(result.returncode)

if __name__ == "__main__":
    main()