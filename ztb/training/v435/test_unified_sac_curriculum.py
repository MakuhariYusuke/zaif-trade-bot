#!/usr/bin/env python3
"""
Test script for unified trainer with SAC curriculum learning integration
"""

import sys
from pathlib import Path

# Add the project root to Python path
sys.path.insert(0, str(Path(__file__).parent))

from ztb.training.unified_trainer.algorithms import SACTrainer


def main():
    """Test unified trainer with SAC curriculum learning"""

    # Create unified config with curriculum learning enabled
    unified_config = {
        "training": {
            "algorithm": "sac",
            "total_timesteps": 1000000,
            "curriculum_learning": True,
            "sac_hyperparameters": {
                "learning_rate": 3e-4,
                "buffer_size": 1000000,
                "learning_starts": 1000,
                "batch_size": 256,
                "tau": 0.005,
                "gamma": 0.99,
                "ent_coef": "auto_1.0",
                "target_entropy": "auto",
            },
            "environment": {
                "initial_balance": 100000,
                "transaction_cost": 0.0015,
                "max_position_size": 0.1,
                "reward_settings": {"reward_scale": 500.0, "reward_clip_max": 200.0},
            },
            "data_config": {"data_path": "data/btc_jpy_featured_dataset.csv"},
        },
        "output": {
            "model_dir": "models/unified_sac_curriculum",
            "tensorboard_log": "tensorboard/unified_sac_curriculum",
        },
    }

    print("Testing unified trainer with SAC curriculum learning...")
    print(f"- Algorithm: {unified_config['training']['algorithm']}")
    print(f"- Curriculum learning: {unified_config['training']['curriculum_learning']}")
    print(f"- Total timesteps: {unified_config['training']['total_timesteps']}")

    # Create SAC trainer
    try:
        trainer = SACTrainer(unified_config)
        print("SAC trainer initialized successfully")
    except Exception as e:
        print(f"Failed to initialize trainer: {e}")
        return

    # Test curriculum learning training
    try:
        print("Starting curriculum learning training...")
        success = trainer.train()
        print(f"Training completed: {'SUCCESS' if success else 'FAILED'}")

        # Get training stats
        stats = trainer.get_training_stats()
        print(f"Training stats: {stats}")

    except Exception as e:
        print(f"Training failed: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
