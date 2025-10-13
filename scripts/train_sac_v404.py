#!/usr/bin/env python3
"""
SAC v404 Training Script with Extreme Rewards for 80%+ Win Rate
- Extreme reward scale (8000) and clip range (-80, 80)
- Enhanced win rate bonus for 90%+ target
- Narrower action thresholds for more trading
- Extended training (15k steps) for maximum convergence
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
    config_path = "config/sac_v404_config.json"

    try:
        # Load config file
        with open(config_path, 'r', encoding='utf-8') as f:
            config = json.load(f)

        print("=== SAC v404 Training with Extreme Rewards for 80%+ Win Rate ===")
        print(f"Model: {config.get('model_name')}")
        reward_settings = config.get("environment", {}).get("reward_settings", {})
        print(f"Reward Scale: {reward_settings.get('reward_scale')}")
        print(f"Reward Clip: [{reward_settings.get('reward_clip_min')}, {reward_settings.get('reward_clip_max')}]")
        print(f"Training Steps: {config.get('total_timesteps')}")
        print("Features: Extreme win rate bonus (50pts for 90%+, 30pts for 80%+)")
        print("         Narrower thresholds (0.15) for more actions, Increased bonuses")
        print()

        # Create config manager
        config_manager = ConfigManager(config)

        # Initialize trainer
        trainer = AlgorithmTrainer(config_manager, progress_bar_enabled=True)

        # Build unified config (simplified for SAC)
        unified_config = {
            "algorithm": config.get("algorithm", "sac"),
            "total_timesteps": config.get("total_timesteps", 15000),
            "model_name": config.get("model_name", "sac_v404_extreme_win_rate"),
            "environment": config.get("environment", {}),
            "sac_hyperparameters": config.get("sac_hyperparameters", {}),
            "data_source": config.get("data_source", "csv"),
            "data_path": config.get("data_path", "btc_jpy_real_dataset.csv"),
            "checkpoint_interval": config.get("checkpoint_interval", 1000),
        }

        # Start training
        print("Starting training...")
        result = trainer.train(unified_config["algorithm"], unified_config)

        if result and result.get("success"):
            print("✅ Training completed successfully!")
            print(f"   Model saved to: {result.get('model_path', 'N/A')}")
            print(f"   Final metrics: {result.get('final_metrics', {})}")
        else:
            print("❌ Training failed!")
            print(f"   Error: {result.get('error', 'Unknown error') if result else 'No result returned'}")

    except Exception as e:
        print(f"❌ Training failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()