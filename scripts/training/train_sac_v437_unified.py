#!/usr/bin/env python3
"""
SAC v437 Training Script - Unified Trainer Version

Enhanced SAC training with v427 features for improved trading performance.
Now uses unified trainer system for consistency and maintainability.
"""

import argparse
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from ztb.training.v4xx_unified_trainer import V4XXUnifiedTrainer
from ztb.utils.logging_utils import get_logger

logger = get_logger(__name__)


def main():
    """Main training function using unified trainer."""
    parser = argparse.ArgumentParser(
        description="SAC v437 Training with Unified Trainer"
    )
    parser.add_argument(
        "--config",
        default="config/sac_v427_default_config.json",
        help="Path to configuration file",
    )
    parser.add_argument(
        "--save-config", action="store_true", help="Save converted configuration"
    )

    args = parser.parse_args()

    try:
        print("🚀 SAC v437 Training - Unified Trainer Version")
        print("=" * 65)

        # Initialize unified trainer
        trainer = V4XXUnifiedTrainer(args.config, version="v427")

        print("📋 Configuration loaded and converted:")
        print(f"  - Original Version: {trainer.version}")
        print(f"  - Model: {trainer.config['model_name']}")
        print(f"  - Algorithm: {trainer.config['algorithm']}")
        print(f"  - Timesteps: {trainer.config['training']['total_timesteps']:,}")

        # Save converted config if requested
        if args.save_config:
            trainer.save_config()

        # Execute training
        trainer.train()

        print("\n✅ Training completed successfully!")

    except Exception as e:
        logger.error(f"Training failed: {e}")
        print(f"\n❌ Training failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
