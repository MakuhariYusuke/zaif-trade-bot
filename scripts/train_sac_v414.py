#!/usr/bin/env python3
"""
SAC v414 Training Script with Balanced Trading Rewards
- Balanced BUY/SELL rewards for 45%/45%/10% target distribution
- Removed BUY bias penalties and constraint penalties
- Linear win-rate bonus for stable learning
- Moderate reward scale (500) with balanced penalties
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
    config_path = "config/sac_v414_balanced_trading_config.json"

    try:
        # Load config file
        with open(config_path, 'r', encoding='utf-8') as f:
            config = json.load(f)

        print("=== SAC v414 Training with Balanced Trading Rewards ===")
        print(f"Model: {config.get('model_name')}")
        reward_settings = config.get("reward_settings", {})
        print(f"Reward Scale: {reward_settings.get('reward_scale')}")
        print(f"Reward Clip: [{reward_settings.get('reward_clip_min')}, {reward_settings.get('reward_clip_max')}]")
        print(f"Training Steps: {config.get('total_timesteps')}")
        print("Features: Balanced BUY/SELL rewards, removed BUY bias, linear win-rate bonus")
        print("         Target: BUY 45%, SELL 45%, HOLD 10%")
        print()

        # Create config manager
        config_manager = ConfigManager(config)

        # Initialize trainer
        trainer = AlgorithmTrainer(config_manager, progress_bar_enabled=True)

        # Build unified config
        unified_config = {
            "algorithm": config.get("algorithm", "sac"),
            "total_timesteps": config.get("total_timesteps", 50000),
            "model_name": config.get("model_name", "sac_v414_balanced_trading"),
            "environment": config.get("environment", {}),
            "reward_settings": config.get("reward_settings", {}),
            "sac_hyperparameters": config.get("sac_hyperparameters", {}),
            "data_source": config.get("data_source", "csv"),
            "data_path": config.get("data_path", "btc_jpy_real_dataset.csv"),
            "checkpoint_interval": config.get("checkpoint_interval", 1000)
        }

        # Start training
        print("Starting training...")
        result = trainer.train(unified_config["algorithm"], unified_config)

        print("\n=== Training Complete ===")
        print(f"Model saved as: checkpoints/{config.get('model_name')}.zip")

    except Exception as e:
        print(f"Training failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()