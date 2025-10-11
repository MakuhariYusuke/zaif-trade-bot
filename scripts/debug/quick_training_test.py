#!/usr/bin/env python3
"""
Quick training test with balanced reward function.
"""

import json
import os
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from ztb.training.unified_trainer import UnifiedTrainer

def main() -> None:
    # Load config
    config_path = project_root / "ppo_100k_config.json"
    with open(config_path, "r") as f:
        config = json.load(f)

    # Modify config for full test
    config["total_timesteps"] = 15000  # Very quick test
    config["seed"] = 42  # Set seed for reproducibility
    config["session_id"] = "scalping_15s_sell_boost_test"

    print(f"Starting full training test with {config['total_timesteps']} timesteps")

    # Create trainer and run
    trainer = UnifiedTrainer(config)
    result = trainer.train()

    # Try to save the model from the result or trainer
    model_path = project_root / "models" / f"{config['session_id']}_final.zip"
    model_path.parent.mkdir(exist_ok=True)

    # Check if result contains the model
    if hasattr(result, "model"):
        result.model.save(str(model_path))
        print(f"Model saved to {model_path}")
    elif hasattr(result, "save"):
        result.save(str(model_path))
        print(f"Model saved to {model_path}")
    else:
        print(f"Warning: Could not save model. Result type: {type(result)}")
        # Try to find any .zip files in current directory that might be the model
        import glob
        zip_files = glob.glob("*.zip")
        if zip_files:
            print(f"Found zip files: {zip_files}")
            # Move the latest one
            latest_zip = max(zip_files, key=os.path.getctime)
            os.rename(latest_zip, str(model_path))
            print(f"Moved {latest_zip} to {model_path}")

    print("Training completed!")

if __name__ == "__main__":
    main()