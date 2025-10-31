#!/usr/bin/env python3
"""
SAC v444 Training Script - Advanced Regime Adaptation

Enhanced SAC training with v444 advanced regime adaptation features.
Uses 12-regime classification for improved trading performance.
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
        description="SAC v444 Training with Advanced Regime Adaptation"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config/sac_v444_advanced_regime_adaptation_config.json",
        help="Path to configuration file",
    )
    parser.add_argument(
        "--version", type=str, default="v444", help="Override version detection"
    )

    args = parser.parse_args()

    try:
        print("🚀 SAC v444 Training - Advanced Regime Adaptation")
        print(f"Configuration: {args.config}")

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