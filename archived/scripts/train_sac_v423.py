#!/usr/bin/env python3
"""
SAC v423 Training Script - Initial Test Version

This script runs SAC v423 with small timesteps for initial testing.
Uses unified_trainer with command-line configurable timesteps.
"""

import os
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from ztb.training.unified_trainer.main import main as unified_trainer_main


def run_v423_training(
    total_timesteps: int = 1000,
    config_file: str = "config/sac_v423_initial_test_config.json",
) -> None:
    """Run SAC v423 training with specified timesteps and config."""

    # Check if config exists
    if not os.path.exists(config_file):
        print(f"❌ Config file not found: {config_file}")
        sys.exit(1)

    print("🚀 Starting SAC v423 Training")
    print(f"   Config: {config_file}")
    print(f"   Timesteps: {total_timesteps}")
    print()

    # Prepare command line arguments for unified_trainer
    sys.argv = [
        "unified_trainer.py",
        "--config",
        config_file,
        "--total-timesteps",
        str(total_timesteps),
        "--force",  # Skip prompts
    ]

    # Run unified trainer
    try:
        unified_trainer_main()
        print("✅ SAC v423 training completed successfully!")
    except Exception as e:
        print(f"❌ Training failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="SAC v423 Training Script")
    parser.add_argument(
        "--timesteps",
        type=int,
        default=1000,
        help="Total training timesteps (default: 1000)",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config/sac_v423_initial_test_config.json",
        help="Path to configuration file",
    )

    args = parser.parse_args()
    run_v423_training(args.timesteps, args.config)
