#!/usr/bin/env python3
"""
Training script for SAC v436 signal guidance variants
"""

import argparse
import json
import os
import sys
from pathlib import Path

import pandas as pd

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from ztb.trading.environment.schema_env_factory import create_env_from_schema
from ztb.training.sac_trainer import SACTrainer


def load_config_from_file(config_path: str) -> dict:
    """Load configuration from JSON file."""
    with open(config_path, "r") as f:
        return json.load(f)


def main():
    parser = argparse.ArgumentParser(description="Train SAC v436 models")
    parser.add_argument("--config", type=str, required=True, help="Path to config file")

    args = parser.parse_args()

    # Load configuration
    try:
        config_data = load_config_from_file(args.config)
        print("✅ Configuration loaded successfully")
    except Exception as e:
        print(f"❌ Failed to load config: {e}")
        return

    # Extract configurations
    training_config = config_data.get("training", {})
    env_config = training_config.get("environment", {})
    data_config = training_config.get("data_config", {})

    # Load data
    data_path = data_config.get("data_path", "data/btc_jpy_real_dataset.csv")
    if not os.path.exists(data_path):
        print(f"❌ Data not found: {data_path}")
        return

    try:
        df = pd.read_csv(data_path)
        print("✅ Data loaded successfully")
    except Exception as e:
        print(f"❌ Failed to load data: {e}")
        return

    # Create environment
    try:
        env = create_env_from_schema(
            model_name=training_config.get("model_name", "sac_v436"),
            df=df,
            config=env_config,
        )
        print("✅ Environment created successfully")
    except Exception as e:
        print(f"❌ Failed to create environment: {e}")
        return

    # Create trainer
    try:
        trainer = SACTrainer(config_path=args.config)
        print("✅ Trainer created successfully")
    except Exception as e:
        print(f"❌ Failed to create trainer: {e}")
        return

    # Train model
    try:
        print("🚀 Starting training...")
        trainer.train(data=df)

        # Save model
        model_path = f"models/{training_config.get('model_name', 'sac_model')}.zip"
        trainer.save_model(model_path)
        print(f"✅ Training completed. Model saved to {model_path}")

    except Exception as e:
        print(f"❌ Training failed: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
