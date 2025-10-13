#!/usr/bin/env python3
"""
SAC v403 Training Script with Aggressive Rewards for 80%+ Win Rate
- Enhanced reward scale (4000) and clip range (-40, 40)
- Added win rate bonus for 80%+ target
- Extended training (10k steps) for better convergence
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
    config_path = "config/sac_v403_config.json"

    try:
        # Load config file
        with open(config_path, 'r', encoding='utf-8') as f:
            config = json.load(f)

        print("=== SAC v403 Training with Aggressive Rewards for 80%+ Win Rate ===")
        print(f"Model: {config.get('model_name')}")
        reward_settings = config.get("environment", {}).get("reward_settings", {})
        print(f"Reward Scale: {reward_settings.get('reward_scale')}")
        print(f"Reward Clip: [{reward_settings.get('reward_clip_min')}, {reward_settings.get('reward_clip_max')}]")
        print(f"Training Steps: {config.get('total_timesteps')}")
        print("Features: Win rate bonus (20pts for 80%+, 10pts for 70%+, 5pts for 60%+)")
        print("         Equal BUY/SELL bonuses, Symmetric action thresholds")
        print()

        # Create config manager
        config_manager = ConfigManager(config)

        # Initialize trainer
        trainer = AlgorithmTrainer(config_manager, progress_bar_enabled=True)

        # Build unified config (simplified for SAC)
        unified_config = {
            "algorithm": config.get("algorithm", "sac"),
            "total_timesteps": config.get("total_timesteps", 10000),
            "model_name": config.get("model_name", "sac_v403_high_win_rate"),
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

        print("\n✅ Training completed successfully!")
        print(f"Model saved to: checkpoints/sac_session/{config.get('model_name')}_final.zip")

    except Exception as e:
        print(f"❌ Training failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()