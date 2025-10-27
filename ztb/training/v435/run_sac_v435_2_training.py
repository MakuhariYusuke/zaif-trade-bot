#!/usr/bin/env python3
"""
Script to run SAC v435.2 training with curriculum learning (Aggressive variant)
"""

import json
import sys
from pathlib import Path

# Add the project root to Python path
sys.path.insert(0, str(Path(__file__).parent))

from ztb.training.v435.train_sac_v435 import SACv435Trainer


def main():
    """Run SAC v435.2 training with curriculum learning"""

    # Load configuration
    config_path = Path("config/v435/sac_v435_2_config.json")
    if not config_path.exists():
        print(f"Configuration file not found: {config_path}")
        return

    with open(config_path, "r") as f:
        config = json.load(f)

    print("Loaded configuration for SAC v435.2 (Aggressive):")
    print(f"- Model: {config['model_name']} v{config['version']}")
    print(f"- Total timesteps: {config['training']['total_timesteps']}")
    print(
        f"- Curriculum learning: {config['training'].get('curriculum_learning', False)}"
    )
    print(f"- Max position size: {config['environment']['max_position_size']}")
    print("- Risk management: Aggressive settings")

    # Create trainer
    try:
        trainer = SACv435Trainer(config)
        print("Trainer initialized successfully")
    except Exception as e:
        print(f"Failed to initialize trainer: {e}")
        return

    # Train with curriculum learning
    try:
        print("Starting training...")
        result = trainer.train()
        print("Training completed successfully!")
        print(f"Result: {result}")
    except Exception as e:
        print(f"Training failed: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
