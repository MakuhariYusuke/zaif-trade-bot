#!/usr/bin/env python3
"""
SAC V442.14 Zero Entropy Adjusted Consistency Balance Training
Tests entropy_regularization=0.0 with adjusted consistency_penalty=0.02 and action_balance_target=0.5
"""

import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


from ztb.training.v4xx_unified_trainer import V4XXUnifiedTrainer


def main():
    """Main training function for v442.14"""

    config_path = (
        "config/sac_v442_14_zero_entropy_adjusted_consistency_balance_config.json"
    )
    model_name = "sac_v442_14_zero_entropy_adjusted_consistency_balance_trading"

    print(
        "🚀 Starting SAC v442.14 training with zero entropy and adjusted consistency..."
    )
    print("Parameters:")
    print("- entropy_regularization: 0.0")
    print("- consistency_penalty: 0.02")
    print("- action_balance_target: 0.5")
    print("- action_smoothing: 0.15")

    # Initialize trainer
    trainer = V4XXUnifiedTrainer(config_path=config_path)

    # Start training
    try:
        results = trainer.train()
        print("✅ SAC v442.14 training completed!")
        return results
    except Exception as e:
        print(f"❌ Training failed: {e}")
        raise


if __name__ == "__main__":
    main()
