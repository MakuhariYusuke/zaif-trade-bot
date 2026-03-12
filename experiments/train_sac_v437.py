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
project_root = Path(__file__).parent
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
        type=str,
        default="config/v437/sac_v437_enhanced_config.json",
        help="Path to configuration file",
    )
    parser.add_argument(
        "--version", type=str, default=None, help="Override version detection"
    )

    args = parser.parse_args()

    try:
        print("🚀 SAC v437 Training - Unified Trainer Version")
        print(f"Configuration: {args.config}")

        # Initialize unified trainer
        trainer = V4XXUnifiedTrainer(config_path=args.config, version=args.version)

        # Validate configuration
        if not trainer.validate_config():
            logger.error("Configuration validation failed")
            return False

        # Initialize trainer
        trainer.initialize_trainer()

        # Execute training
        trainer.train()

        print("✅ Training completed successfully!")
        return True

    except Exception as e:
        logger.error(f"Training failed: {e}")
        raise


if __name__ == "__main__":
    main()
