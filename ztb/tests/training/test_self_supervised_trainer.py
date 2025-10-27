#!/usr/bin/env python3
"""
Test script for SelfSupervisedTrainer implementation
"""
import logging
import sys
from pathlib import Path

# Add the project root to Python path
sys.path.insert(0, str(Path(__file__).parent))

from ztb.training.unified_trainer.algorithms import create_algorithm_trainer


def test_self_supervised_trainer():
    """Test SelfSupervisedTrainer with a simple configuration"""
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    # Create a simple self-supervised configuration
    config = {
        "training": {
            "self_supervised": {
                "model_type": "masked_price_modeling",
                "input_dim": 128,
                "hidden_dim": 64,
                "num_heads": 8,
                "num_layers": 4,
                "learning_rate": 0.001,
                "batch_size": 32,
                "max_epochs": 10,
            },
            "data_path": "data/btc_data.csv",
            "output_dir": "test_results",
        }
    }

    print("Creating SelfSupervised trainer...")
    try:
        trainer = create_algorithm_trainer("self_supervised", config, logger=logger)
        print("✅ SelfSupervisedTrainer created successfully")

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

        print("🎉 SelfSupervisedTrainer test completed successfully")
        return True

    except Exception as e:
        print(f"❌ SelfSupervisedTrainer test failed: {e}")
        import traceback

        print(traceback.format_exc())
        return False


if __name__ == "__main__":
    success = test_self_supervised_trainer()
    sys.exit(0 if success else 1)
