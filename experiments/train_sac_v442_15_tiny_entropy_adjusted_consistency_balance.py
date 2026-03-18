#!/usr/bin/env python3
"""
SAC v442.15 Training Script - Tiny Entropy Regularization Test
Tests entropy_regularization=0.0001 with consistency_penalty=0.02
"""

import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ztb.training.v4xx_unified_trainer import V4XXUnifiedTrainer


def main():
    """Main training function for SAC v442.15"""

    config_path = (
        "config/sac_v442_14_zero_entropy_adjusted_consistency_balance_config.json"
    )

    # Parameter descriptions
    print("=" * 60)
    print("SAC v442.15 Training - Tiny Entropy Regularization Test")
    print("=" * 60)
    print(f"Config: {config_path}")
    print("Parameters:")
    print("  - entropy_regularization: 0.0001 (very small positive value)")
    print("  - consistency_penalty: 0.02 (increased from v442.12)")
    print("  - action_balance_target: 0.5")
    print("  - action_smoothing: 0.15")
    print("Goal: Test if tiny entropy_regularization prevents extreme BUY bias")
    print("=" * 60)

    # Initialize trainer
    try:
        trainer = V4XXUnifiedTrainer(config_path=config_path)
        print("✅ Trainer initialized")
    except Exception as e:
        print(f"❌ Failed to initialize trainer: {e}")
        return

    # Run training
    try:
        print("🚀 Starting training...")
        trainer.train()
        print("✅ Training completed successfully")
    except Exception as e:
        print(f"❌ Training failed: {e}")
        return


if __name__ == "__main__":
    main()
