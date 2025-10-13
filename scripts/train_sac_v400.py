#!/usr/bin/env python3
"""
SAC v400 Training Script with High Reward Scale and Continuous Action Fix
- Fixed continuous action to discrete conversion bug
- Increased reward scale to 2000x for higher win rate
- Improved action balance with proper discrete action handling
"""

import sys
import os
import json
from pathlib import Path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from ztb.training.core.algorithm_trainer import AlgorithmTrainer
from ztb.training.core.config_manager import ConfigManager

def main():
    # Load configuration
    config_path = "config/sac_v396_optimized.json"

    try:
        # Load config file
        with open(config_path, 'r', encoding='utf-8') as f:
            config = json.load(f)

        print("=== SAC v400 Training with High Reward & Continuous Fix ===")
        print(f"Model: {config.get('model_name')}")
        reward_settings = config.get("environment", {}).get("reward_settings", {})
        print(f"Reward Scale: {reward_settings.get('reward_scale')}")
        print(f"Reward Clip: [{reward_settings.get('reward_clip_min')}, {reward_settings.get('reward_clip_max')}]")
        print("Fixes: Continuous action → discrete conversion, Higher reward scale")

        # Create config manager
        config_manager = ConfigManager(config)

        # Initialize trainer
        trainer = AlgorithmTrainer(config_manager, progress_bar_enabled=True)

        # Build unified config (simplified for SAC)
        unified_config = {
            "algorithm": config.get("algorithm", "sac"),
            "total_timesteps": config.get("total_timesteps", 5000),
            "model_name": config.get("model_name", "sac_v400_high_reward_continuous_fix"),
            "environment": config.get("environment", {}),
            "sac_hyperparameters": config.get("sac_hyperparameters", {}),
            "data_source": config.get("data_source", "csv"),
            "data_path": config.get("data_path", "btc_jpy_real_dataset.csv"),
            "checkpoint_interval": config.get("checkpoint_interval", 1000),
        }

        # Train the model
        print("\nStarting training...")
        result = trainer.train(unified_config["algorithm"], unified_config)

        if result and result.get("success"):
            print("✅ Training completed successfully!")
            print(f"   Model saved to: {result.get('model_path', 'N/A')}")
            print(f"   Final metrics: {result.get('final_metrics', {})}")
        else:
            print("❌ Training failed")
            if result:
                print(f"   Error: {result.get('error', 'Unknown error')}")

    except Exception as e:
        print(f"❌ Error during training: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()