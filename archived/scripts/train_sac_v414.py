#!/usr/bin/env python3
"""
Train SAC v414 Balanced Trading Model
"""

import os
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

os.environ["MPLBACKEND"] = "Agg"


def main():
    """Train the balanced trading model"""
    from ztb.training.core.config_manager import ConfigManager
    from ztb.training.trainers.sac_trainer import SACAlgorithmTrainer
    from ztb.utils.config_loader import ConfigLoader

    config_path = project_root / "config" / "sac_v414_balanced_trading_config.json"

    if not config_path.exists():
        print(f"Config file not found: {config_path}")
        sys.exit(1)

    print("=" * 80)
    print("SAC v414 Balanced Trading Training")
    print("=" * 80)
    print(f"Config: {config_path}")
    print("\nKey improvements:")
    print("  - Equal BUY/SELL profit/loss treatment")
    print("  - Moderate HOLD penalty with position-based adjustment")
    print("  - Trading constraints (can't buy with sell position)")
    print("  - Balanced target ratios: HOLD 10%, BUY 45%, SELL 45%")
    print("=" * 80)

    try:
        # Load config using ConfigLoader
        config = ConfigLoader.load(config_path)

        # Create ConfigManager and SACAlgorithmTrainer
        config_manager = ConfigManager(config)
        trainer = SACAlgorithmTrainer(config_manager, progress_bar_enabled=True)

        # Train the model
        result = trainer.train(config)

        if result and result.get("success"):
            print("Training completed successfully!")
            print(f"Model saved to: {result.get('model_path')}")
            print(f"Logs saved to: {result.get('log_path')}")
        else:
            print("Training failed!")
            sys.exit(1)

    except Exception as e:
        print(f"Training failed: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
