#!/usr/bin/env python3
"""
SAC v442.16 Training Script - Balance Mechanism Fixed

This script trains SAC with corrected balance mechanism implementation.
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from ztb.training.v4xx_unified_trainer import V4XXUnifiedTrainer


def main():
    """Main training function."""
    config_path = "config/sac_v442_16_balance_mechanism_fixed_config.json"

    print("=== SAC v442.16 Training: Balance Mechanism Fixed ===")
    print(f"Config: {config_path}")
    print("Parameters:")
    print("- entropy_regularization: 0.0 (disabled)")
    print("- action_balance_target: 0.4 (BUY/SELL target)")
    print("- consistency_penalty: 0.01 (reduced)")
    print("- Fixed target_ratios calculation")
    print("- Safe entropy regularization (no penalties)")
    print()

    # Initialize trainer
    trainer = V4XXUnifiedTrainer(config_path=config_path)

    # Run training
    trainer.train()

    print("\n=== Training Completed ===")


if __name__ == "__main__":
    main()
