#!/usr/bin/env python3
"""
Test training with improved reward function settings.
"""

import sys
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from ztb.training.unified_trainer.config import load_config
from ztb.training.unified_trainer.trainer import UnifiedTrainer

def main():
    # Load the improved config
    config_path = "config/sac_v396_optimized.json"
    config = load_config(config_path)

    if config is None:
        print(f"Failed to load config from {config_path}")
        return

    print("=== Training with Improved Reward Function ===")
    print(f"Model Name: {config.get('model_name')}")
    print(f"Reward Settings: {config.get('environment', {}).get('reward_settings', {})}")
    print()

    # Create trainer
    trainer = UnifiedTrainer(config)

    # Run training with limited timesteps for testing
    test_config = config.copy()
    test_config["total_timesteps"] = 1000  # Short test run

    print("Starting training test with improved reward function...")
    success = trainer.run()
    print(f"Training completed: {'Success' if success else 'Failed'}")

if __name__ == "__main__":
    main()