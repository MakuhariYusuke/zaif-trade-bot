#!/usr/bin/env python3
"""
Quick Train SAC v444 Multi Config - Multiple Config Training

Training script for SAC v444 that can run multiple configurations sequentially.
Supports comparison and analysis of different parameter settings.
"""

import argparse
import sys
from pathlib import Path
from typing import List

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from ztb.training.v4xx_unified_trainer import V4XXUnifiedTrainer
from ztb.utils.logging_utils import get_logger

logger = get_logger(__name__)


def train_single_config(config_path: str, verbose: bool = False) -> bool:
    """Train a single configuration."""
    try:
        print(f"\n🚀 Training with config: {config_path}")

        # Initialize unified trainer
        trainer = V4XXUnifiedTrainer(config_path=config_path, version="v444")

        # Validate configuration
        if not trainer.validate_config():
            logger.error(f"Configuration validation failed for {config_path}")
            return False

        # Start training
        success = trainer.train()
        if success:
            print(f"✅ Training completed successfully for {config_path}")
            return True
        else:
            print(f"❌ Training failed for {config_path}")
            return False

    except Exception as e:
        logger.error(f"Training failed for {config_path} with error: {e}")
        print(f"❌ Training failed for {config_path}: {e}")
        return False


def main() -> bool:
    """Main training function for multiple configs."""
    parser = argparse.ArgumentParser(
        description="Quick Train SAC v444 - Multiple Config Training"
    )
    parser.add_argument(
        "--configs",
        type=str,
        nargs="+",
        help="List of configuration files to train",
    )
    parser.add_argument(
        "--compare",
        action="store_true",
        help="Run comparison with predefined configs (sac_v444_3,4,5)",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose output",
    )

    args = parser.parse_args()

    # Define comparison configs
    comparison_configs = [
        "config/sac_v444_3_balanced_penalty_scale_200.json",
        "config/sac_v444_4_balanced_penalty_scale_300.json",
        "config/sac_v444_5_balanced_penalty_scale_500.json",
    ]

    if args.compare:
        configs_to_train = comparison_configs
        print("🔄 Running comparison training with predefined configs:")
        for config in configs_to_train:
            print(f"  - {config}")
    elif args.configs:
        configs_to_train = args.configs
    else:
        print("❌ Either --configs or --compare must be specified")
        return False

    try:
        print("🚀 Quick Train SAC v444 - Multiple Config Training")
        print(f"Training {len(configs_to_train)} configurations")

        results = []
        for config_path in configs_to_train:
            success = train_single_config(config_path, args.verbose)
            results.append((config_path, success))

        # Summary
        print("\n📊 Training Summary:")
        successful = sum(1 for _, success in results if success)
        total = len(results)
        print(f"✅ Successful: {successful}/{total}")

        if successful == total:
            print("🎉 All trainings completed successfully!")
            return True
        else:
            print("⚠️  Some trainings failed")
            return False

    except Exception as e:
        logger.error(f"Multi-config training failed with error: {e}")
        print(f"❌ Multi-config training failed: {e}")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)