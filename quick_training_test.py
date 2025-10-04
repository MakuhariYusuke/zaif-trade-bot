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
    config_path = project_root / "unified_training_config_balance_test1.json"
    with open(config_path, "r") as f:
        config = json.load(f)

    # Modify config for full test
    config["total_timesteps"] = 100000  # Full 100k test
    config["session_id"] = "scalping_15s_balance_quick_test"

    print(f"Starting quick training test with {config['total_timesteps']} timesteps")

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
            # Move the latest one to models directory
            import shutil

            latest_zip = max(zip_files, key=lambda f: os.path.getmtime(f))
            shutil.move(latest_zip, str(model_path))
            print(f"Moved {latest_zip} to {model_path}")

    print("Quick training test completed")
    return result


if __name__ == "__main__":
    main()
