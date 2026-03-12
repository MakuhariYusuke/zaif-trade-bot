#!/usr/bin/env python3
"""
SAC v442.13 Training Script - Tiny Entropy Regularization Balance Test
Tests entropy_regularization = 0.001 to find optimal balance between BUY/SELL bias
"""

import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ztb.training.v4xx_unified_trainer import V4XXUnifiedTrainer


def main():
    """Main training function for v442.13 with tiny entropy regularization"""

    config_path = "config/sac_v442_13_tiny_entropy_balance_config.json"
    model_name = "sac_v442_13_tiny_entropy_balance_trading"

    print("🚀 Starting SAC v442.13 training with tiny entropy regularization...")
    print("Configuration: entropy_regularization = 0.001")
    print("Goal: Test if very small entropy_regularization improves action balance")

    trainer = V4XXUnifiedTrainer(config_path=config_path)

    try:
        results = trainer.train()
        print("✅ SAC v442.13 training completed!")
        return results
    except Exception as e:
        print(f"❌ Training failed: {e}")
        raise


if __name__ == "__main__":
    main()
