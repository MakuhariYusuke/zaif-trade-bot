#!/usr/bin/env python3
"""
Quick training test with balanced reward function.
"""

import glob
import shutil
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root.parent.parent))

from ztb.training.core.unified_trainer import UnifiedTrainer
from ztb.utils.file_utils import safe_json_load


def main() -> None:
    # Load config
    config_path = (
        project_root.parent.parent
        / "config"
        / "training"
        / "balance_tests"
        / "unified_training_config_balance_test1.json"
    )
    config = safe_json_load(config_path)

    # Run multiple tests with different seeds
    for run_num in range(1, 4):  # Run 3 times
        # Modify config for this run
        config["total_timesteps"] = 50000  # Longer test for better learning
        config["seed"] = 42 + run_num  # Different seed for each run
        config["session_id"] = f"scalping_15s_balance_quick_test_run{run_num}"

        print(f"\n=== Starting run {run_num} with seed {config['seed']} ===")

        # Create trainer and run
        trainer = UnifiedTrainer(config)
        result = trainer.train()

        # Try to save the model from the result or trainer
        model_path = (
            project_root.parent.parent / "models" / f"{config['session_id']}_final.zip"
        )
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
            zip_files = glob.glob("*.zip")
            if zip_files:
                print(f"Found zip files: {zip_files}")
                # Move the first one
                shutil.move(zip_files[0], str(model_path))
                print(f"Moved {zip_files[0]} to {model_path}")

        print(f"Run {run_num} completed")

    print("Quick training test completed")


if __name__ == "__main__":
    main()
