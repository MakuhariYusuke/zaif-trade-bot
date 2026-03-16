#!/usr/bin/env python3
"""
SAC v444.1 Training Script - Unified Trainer Version

Enhanced SAC training with v444.1 advanced regime adaptation.
Uses unified trainer system for consistency and maintainability.
"""

import argparse
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from ztb.training.unified_trainer.trainer import UnifiedTrainer
from ztb.utils.logging_utils import get_logger

logger = get_logger(__name__)


def main():
    """Main training function using unified trainer."""
    parser = argparse.ArgumentParser(
        description="SAC v444.1 Training with Unified Trainer"
    )
    parser.add_argument(
        "--config",
        default="configs/sac_v444.1_config.json",
        help="Path to configuration file",
    )
    parser.add_argument(
        "--total-timesteps",
        type=int,
        default=None,
        help="Override total timesteps for quick testing",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force execution without prompts",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate configuration without training",
    )

    args = parser.parse_args()

    try:
        print("🚀 SAC v444.1 Training - Unified Trainer Version")
        print("=" * 65)

        # Load configuration
        import json

        with open(args.config, "r") as f:
            config = json.load(f)

        # Override timesteps if specified
        if args.total_timesteps:
            config["total_timesteps"] = args.total_timesteps

        print("📋 Configuration loaded:")
        print(f"  - Model: {config.get('model_name', 'Unknown')}")
        print(f"  - Version: {config.get('version', 'Unknown')}")
        print(f"  - Algorithm: {config.get('algorithm', 'Unknown')}")
        print(f"  - Timesteps: {config.get('total_timesteps', 'Default'):,}")

        # Count features
        features = config.get("features", {})
        total_features = sum(
            len(feature_list) if isinstance(feature_list, list) else 1
            for feature_list in features.values()
        )
        print(f"  - Features: {total_features} categories")

        if args.dry_run:
            print("🔍 Dry run mode - validating configuration only")
            print("✅ Configuration validation successful")
            return

        # Initialize unified trainer
        trainer = UnifiedTrainer(config=config, force=args.force, dry_run=args.dry_run)

        print("🎯 Starting training...")
        print("-" * 65)

        # Run training
        trainer.run()

        print("✅ Training completed successfully!")

    except Exception as e:
        logger.error(f"Training failed: {e}")
        print(f"❌ Training failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
