#!/usr/bin/env python3
"""
SAC v442.12 Training Script - Zero Entropy Regularization Balance Refinement
Focus: Eliminate entropy_regularization completely to test if it's causing SELL bias
"""

import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ztb.training.v4xx_unified_trainer import V4XXUnifiedTrainer


def main():
    """Execute SAC v442.12 training with zero entropy regularization"""

    config_path = "config/sac_v442_12_zero_entropy_balance_config.json"

    print(
        "🚀 Starting SAC v442.12 Training - Zero Entropy Regularization Balance Refinement"
    )
    print("=" * 80)
    print("Configuration: Zero entropy_regularization (0.0)")
    print("Parameters:")
    print("  - entropy_regularization: 0.0 (disabled)")
    print("  - consistency_penalty: 0.01")
    print("  - action_balance_target: 0.45")
    print("  - action_smoothing: 0.15")
    print("=" * 80)

    trainer = V4XXUnifiedTrainer(config_path=config_path)
    trainer.train()

    print("✅ SAC v442.12 training completed!")


if __name__ == "__main__":
    main()
