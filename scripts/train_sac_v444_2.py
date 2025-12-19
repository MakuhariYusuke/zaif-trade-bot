#!/usr/bin/env python3
"""
SAC v444.2 Training Script - Integrated Regime Adaptation

Enhanced SAC training with v444.2 integrated regime adaptation features.
Uses comprehensive 12-regime classification with risk management and bearish signal features.
Supports checkpoint resumption for continued training.
"""

import argparse
import sys
import time
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from ztb.training.v4xx_unified_trainer import V4XXUnifiedTrainer
from ztb.utils.logging_utils import get_logger
from ztb.utils.training_utils import display_training_complete

logger = get_logger(__name__)


def main() -> bool:
    """Main training function using unified trainer with checkpoint support."""
    parser = argparse.ArgumentParser(
        description="SAC v444.2 Training with Integrated Regime Adaptation"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config/sac_v444_2_integrated_regime_adaptation_config.json",
        help="Path to configuration file",
    )
    parser.add_argument(
        "--version", type=str, default="v444.2", help="Override version detection"
    )
    parser.add_argument(
        "--resume-from",
        type=str,
        default=None,
        help="Path to checkpoint file to resume training from (e.g., checkpoints/sac_v438_production_50000_steps.zip)",
    )
    parser.add_argument(
        "--total-timesteps",
        type=int,
        default=5000,
        help="Total timesteps to train (default: 5000 for initial testing)",
    )
    parser.add_argument(
        "--checkpoint-freq",
        type=int,
        default=1000,
        help="Frequency to save checkpoints (default: 1000)",
    )

    args = parser.parse_args()

    try:
        print("🚀 SAC v444.2 Training - Integrated Regime Adaptation")
        print(f"Configuration: {args.config}")
        print(f"Total timesteps: {args.total_timesteps}")
        if args.resume_from:
            print(f"Resuming from checkpoint: {args.resume_from}")

        # Initialize unified trainer
        trainer = V4XXUnifiedTrainer(config_path=args.config, version=args.version)

        # Validate configuration
        if not trainer.validate_config():
            logger.error("Configuration validation failed")
            return False

        # Override total timesteps for testing
        trainer.config["training"]["total_timesteps"] = args.total_timesteps

        # Set checkpoint frequency
        if "checkpoint_freq" not in trainer.config:
            trainer.config["checkpoint_freq"] = args.checkpoint_freq

        # Resume from checkpoint if specified
        if args.resume_from:
            print(f"Loading checkpoint from: {args.resume_from}")
            if not trainer.load_checkpoint(args.resume_from):
                logger.error(f"Failed to load checkpoint: {args.resume_from}")
                return False
            print("✅ Checkpoint loaded successfully")

        # Start training
        print("🎯 Starting training...")
        start_time = time.time()
        success = trainer.train()
        training_time = time.time() - start_time

        if success:
            display_training_complete(trainer.config, training_time)
            print(f"Checkpoints saved with frequency: {args.checkpoint_freq} steps")
            return True
        else:
            print("❌ Training failed!")
            return False

    except Exception as e:
        logger.error(f"Training failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)