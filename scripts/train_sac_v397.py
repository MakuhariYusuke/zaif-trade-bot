#!/usr/bin/env python3
"""
Train SAC v397 with improved reward function.
"""

import sys
import json
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from ztb.training.core.algorithm_trainer import AlgorithmTrainer
from ztb.training.core.config_manager import ConfigManager

def main():
    print("🚀 SAC v397 Training with Improved Reward Function 🚀")
    print("=" * 80)

    # Load the improved config
    config_path = "config/sac_v396_optimized.json"

    try:
        # Load config file
        with open(config_path, 'r', encoding='utf-8') as f:
            config = json.load(f)

        print(f"✅ Config loaded: {config_path}")
        print(f"   Model name: {config.get('model_name')}")
        print(f"   Algorithm: {config.get('algorithm')}")
        print(f"   Total timesteps: {config.get('total_timesteps')}")
        print()

        # Create config manager
        config_manager = ConfigManager(config)

        # Create trainer
        trainer = AlgorithmTrainer(config_manager, progress_bar_enabled=True)

        # Build unified config (simplified for SAC)
        unified_config = {
            "algorithm": config.get("algorithm", "sac"),
            "total_timesteps": config.get("total_timesteps", 5000),
            "model_name": config.get("model_name", "sac_v397_reward_improved"),
            "environment": config.get("environment", {}),
            "sac_hyperparameters": config.get("sac_hyperparameters", {}),
            "data_source": config.get("data_source", "csv"),
            "data_path": config.get("data_path", "btc_jpy_real_dataset.csv"),
            "checkpoint_interval": config.get("checkpoint_interval", 1000),
        }

        # Run training
        print("🎯 Starting SAC training with improved reward function...")
        print("   Reward settings:")
        reward_settings = config.get("environment", {}).get("reward_settings", {})
        for key, value in reward_settings.items():
            print(f"     {key}: {value}")
        print()

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