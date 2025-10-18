#!/usr/bin/env python3
"""
SAC v424 Training Script - Cost Aware Version

This script runs SAC v424 with cost-aware configuration to address overtrading.
"""

import os
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from ztb.training.unified_trainer.main import main as unified_trainer_main


def run_v424_training(total_timesteps: int = 15000, config_file: str = "configs/sac_v424_cost_aware.json") -> None:
    """Run SAC v424 training with specified timesteps and config."""

    # Check if config exists
    if not os.path.exists(config_file):
        print(f"❌ Config file not found: {config_file}")
        sys.exit(1)

    print("🚀 Starting SAC v424 Cost-Aware Training")
    print(f"   Config: {config_file}")
    print(f"   Timesteps: {total_timesteps}")
    print()

    # Prepare command line arguments for unified_trainer
    sys.argv = [
        "unified_trainer.py",
        "--config", config_file,
        "--total-timesteps", str(total_timesteps),
        "--force"  # Skip prompts
    ]

    # Run unified trainer
    try:
        unified_trainer_main()
        print("✅ SAC v424 training completed successfully!")
    except Exception as e:
        print(f"❌ Training failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="SAC v424 Cost-Aware Training")
    parser.add_argument("--timesteps", type=int, default=15000, help="Total training timesteps")
    parser.add_argument("--config", type=str, default="configs/sac_v424_cost_aware.json", help="Config file path")

    args = parser.parse_args()
    run_v424_training(args.timesteps, args.config)