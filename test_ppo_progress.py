#!/usr/bin/env python3
"""
Test PPO Trainer with progress bar.
"""

import json
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from ztb.training.ppo_trainer import PPOTrainer

def main() -> None:
    # Simple config for testing
    config = {
        "data_path": "ml-dataset-enhanced.csv",
        "total_timesteps": 5000,  # Very short test
        "learning_rate": 0.0001,
        "ent_coef": 0.01,
        "checkpoint_dir": "checkpoints",
        "verbose": 1,
        "seed": 42,
    }

    print("Testing PPO Trainer with progress bar...")

    trainer = PPOTrainer(
        data_path=config["data_path"],
        config=config,
        checkpoint_dir=config["checkpoint_dir"],
    )

    model = trainer.train(session_id="progress_bar_test")
    
    # Save the model
    if model is not None:
        model_path = "models/progress_bar_test.zip"
        model.save(model_path)
        print(f"Model saved to {model_path}")

    print("Training completed successfully!")

if __name__ == "__main__":
    main()