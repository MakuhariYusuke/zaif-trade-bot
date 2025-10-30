#!/usr/bin/env python3
"""
SAC v442.11 Balance Refinement Final Training Script

This script trains SAC v442.11 with final balance parameter refinements:
- entropy_regularization: 0.005 (reduced from 0.02 to minimize SELL bias)
- consistency_penalty: 0.01 (reduced from 0.02 for less aggressive consistency enforcement)
- action_balance_target: 0.45 (reduced from 0.55 to encourage more BUY actions)
- action_smoothing: 0.15 (maintained)

Goal: Achieve balanced action distribution after persistent SELL bias in v442.10 (93.5% SELL)
"""

import os
import sys

# Add the project root to the Python path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)
sys.path.insert(0, os.path.join(project_root, "src"))

from ztb.training.v4xx_unified_trainer import V4XXUnifiedTrainer


def main():
    config_path = "config/sac_v442_11_balance_refinement_final_config.json"

    print("🚀 Starting SAC v442.11 Balance Refinement Final Training")
    print("Configuration: Final parameter tuning to eliminate SELL bias")
    print("- entropy_regularization: 0.005 (reduced)")
    print("- consistency_penalty: 0.01 (reduced)")
    print("- action_balance_target: 0.45 (reduced)")
    print("- action_smoothing: 0.15 (maintained)")
    print()

    trainer = V4XXUnifiedTrainer(config_path=config_path)
    trainer.train()

    print("\n✅ SAC v442.11 training completed successfully!")


if __name__ == "__main__":
    main()
