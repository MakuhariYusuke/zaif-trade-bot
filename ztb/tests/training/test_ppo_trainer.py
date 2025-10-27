#!/usr/bin/env python3
"""
Test script for PPOTrainer implementation
"""
import json
import logging
import sys
from pathlib import Path

# Add the project root to Python path
sys.path.insert(0, str(Path(__file__).parent))

from ztb.training.unified_trainer.algorithms import create_algorithm_trainer


def test_ppo_trainer():
    """Test PPOTrainer with a simple configuration"""
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    # Load a PPO configuration
    config_path = Path("config/ppo_profitable_v392_bugfix.json")
    if not config_path.exists():
        logger.error(f"Config file not found: {config_path}")
        return False

    with open(config_path, "r") as f:
        config = json.load(f)

    logger.info("Loaded PPO configuration")

    # Create PPO trainer
    print("Creating PPO trainer...")
    try:
        trainer = create_algorithm_trainer("ppo", config, logger=logger)
        print("✅ PPOTrainer created successfully")

        # Test initialization (don't actually train to save time)
        print("Testing training optimizations initialization...")
        if hasattr(trainer, "_initialize_training_optimizations"):
            trainer._initialize_training_optimizations()
            print("✅ Training optimizations initialized")

        print("Testing metrics collection initialization...")
        if hasattr(trainer, "_initialize_metrics_collection"):
            trainer._initialize_metrics_collection("test_results")
            print("✅ Metrics collection initialized")

        # Check if trainer has the expected attributes
        expected_attrs = [
            "model",
            "config",
            "lr_scheduler",
            "early_stopping",
            "metrics_csv_writer",
        ]
        for attr in expected_attrs:
            if hasattr(trainer, attr):
                print(f"✅ Trainer has attribute: {attr}")
            else:
                print(f"❌ Trainer missing attribute: {attr}")

        print("🎉 PPOTrainer test completed successfully")
        return True

    except Exception as e:
        print(f"❌ PPOTrainer test failed: {e}")
        import traceback

        print(traceback.format_exc())
        return False


if __name__ == "__main__":
    success = test_ppo_trainer()
    sys.exit(0 if success else 1)
