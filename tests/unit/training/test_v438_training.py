#!/usr/bin/env python3
"""
SAC v438 Test Training Script - Using UnifiedTrainer
Test v438 reward configuration with 10k steps
"""

import argparse
import json
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from ztb.training.unified_trainer import UnifiedTrainer
from ztb.utils.logging_utils import get_logger

logger = get_logger(__name__)


def main():
    """Main training function using UnifiedTrainer."""

    parser = argparse.ArgumentParser(description="SAC v438 Test Training")
    parser.add_argument(
        "--config",
        default="config/sac_v438_test_config.json",
        help="Path to configuration file",
    )
    parser.add_argument(
        "--total-timesteps",
        type=int,
        default=10000,
        help="Total training timesteps (overrides config)",
    )
    parser.add_argument(
        "--force", action="store_true", help="Force execution without prompts"
    )

    args = parser.parse_args()

    try:
        print("🚀 SAC v438 Test Training - Unified Trainer Version")
        print("=" * 65)

        # Load configuration
        with open(args.config, "r") as f:
            config = json.load(f)

        print(f"📋 Configuration loaded: {args.config}")
        print(f"  - Model: {config['training']['model_name']}")
        print(f"  - Algorithm: {config['training']['algorithm']}")
        print(f"  - Config Timesteps: {config['training']['total_timesteps']:,}")
        print(f"  - Override Timesteps: {args.total_timesteps:,}")

        # Debug: Check config structure
        print("🔍 Original Config Keys:", list(config.keys()))
        if "training" in config:
            print("📊 Training section keys:", list(config["training"].keys()))
        print("-" * 65)

        # Initialize unified trainer with override timesteps
        trainer = UnifiedTrainer(
            config=config, total_timesteps=args.total_timesteps, force=args.force
        )

        print("🏃 Starting training...")
        print(f"   Target: {args.total_timesteps:,} timesteps")
        print("-" * 65)

        # Execute training
        result = trainer.train()

        if result:
            print("✅ Training completed successfully!")
            print(f"   Result: {result}")
        else:
            print("❌ Training failed or returned no result")

    except Exception as e:
        logger.error(f"Training failed: {e}")
        print(f"❌ Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
