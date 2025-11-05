#!/usr/bin/env python3
"""
Quick Train SAC v444 Configurable - Single Config Training

Fast training script for SAC v444 with configurable parameters.
Supports verbose output and quick testing of different configurations.
"""

import argparse
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from ztb.training.v4xx_unified_trainer import V4XXUnifiedTrainer
from ztb.utils.logging_utils import get_logger

logger = get_logger(__name__)


def main() -> bool:
    """Main training function using unified trainer."""
    parser = argparse.ArgumentParser(
        description="Quick Train SAC v444 - Single Config Training"
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to configuration file",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose output",
    )
    parser.add_argument(
        "--version", type=str, default="v444", help="Override version detection"
    )

    args = parser.parse_args()

    try:
        print("🚀 Quick Train SAC v444 - Single Config Training")
        print(f"Configuration: {args.config}")
        if args.verbose:
            print("Verbose mode enabled")

        # Initialize unified trainer
        trainer = V4XXUnifiedTrainer(config_path=args.config, version=args.version)

        # Validate configuration
        if not trainer.validate_config():
            logger.error("Configuration validation failed")
            return False

        # Start training
        success = trainer.train()
        if success:
            print("✅ SAC v444 training completed successfully!")
            return True
        else:
            print("❌ SAC v444 training failed!")
            return False

    except Exception as e:
        logger.error(f"Training failed with error: {e}")
        print(f"❌ Training failed: {e}")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)