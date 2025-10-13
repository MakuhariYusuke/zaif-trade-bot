#!/usr/bin/env python3
"""
SAC v399 Training Script with Balanced Reward Function
- Uses Sharpe-based reward with action balance bonus
- Improved reward scaling and clipping
- Focus on balanced BUY/SELL actions
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

        print("=== SAC v399 Training with Balanced Reward ===")
        print(f"Model: {config.get('model_name')}")
        reward_settings = config.get("environment", {}).get("reward_settings", {})
        print(f"Reward Scale: {reward_settings.get('reward_scale')}")
        print(f"Reward Clip: {reward_settings.get('reward_clip')}")
        print("Features: Sharpe-based reward + Action Balance Bonus")

        # Create config manager
        config_manager = ConfigManager(config)

        # Initialize trainer
        trainer = AlgorithmTrainer(config_manager, progress_bar_enabled=True)

        # Build unified config (simplified for SAC)
        unified_config = {
            "algorithm": config.get("algorithm", "sac"),
            "total_timesteps": config.get("total_timesteps", 5000),
            "model_name": config.get("model_name", "sac_v399_balanced_reward"),
            "environment": config.get("environment", {}),
            "sac_hyperparameters": config.get("sac_hyperparameters", {}),
            "data_source": config.get("data_source", "csv"),
            "data_path": config.get("data_path", "btc_jpy_real_dataset.csv"),
            "checkpoint_interval": config.get("checkpoint_interval", 1000),
        }

        # Train the model
        print("\nStarting training...")
        print(f"Algorithm: {unified_config['algorithm']}")
        print(f"Total timesteps: {unified_config['total_timesteps']}")
        print(f"Model name: {unified_config['model_name']}")
        print("Debug: Checking environment action space...")
        
        # Check environment action space
        env_config = unified_config.get("environment", {})
        use_continuous = env_config.get("use_continuous_actions", False)
        print(f"Use continuous actions: {use_continuous}")
        
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
        print(f"Error type: {type(e)}")
        print(f"Error args: {e.args}")