#!/usr/bin/env python3
"""
Training script for v443.2 Phase 2: Market Regime Adaptation
"""

import logging
import sys
import time
from pathlib import Path

# Add project root to path
project_root = next(
    (p for p in Path(__file__).resolve().parents if (p / "pyproject.toml").exists()),
    Path(__file__).resolve().parent,
)
sys.path.insert(0, str(project_root))

from ztb.training.v4xx_unified_trainer import V4XXUnifiedTrainer
from ztb.utils.logging_utils import setup_logging
from ztb.utils.training_utils import display_training_complete


def train_v443_2_phase2():
    """Run full training of v443.2 Phase 2 with market regime adaptation"""

    print("=== Training v443.2 Phase 2: Market Regime Adaptation ===")

    start_time = time.time()

    # Set up logging to file and console
    setup_logging(log_file="training_log_v443_2_phase2.txt")

    # Ensure all loggers inherit the root logger configuration
    logging.getLogger().setLevel(logging.INFO)

    config_path = "config/v443_2_phase2_config.json"

    try:
        # Create trainer with config file
        trainer = V4XXUnifiedTrainer(config_path=config_path)

        # Re-add file handler after trainer initialization (which may clear handlers)
        file_handler = logging.FileHandler("training_log_v443_2_phase2.txt", mode="a")
        file_handler.setLevel(logging.INFO)
        file_handler.setFormatter(
            logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
        )
        logging.getLogger().addHandler(file_handler)

        print(f"Starting training with config: {config_path}")
        print("Market regime adaptation enabled: True")
        print("Total timesteps: 50,000")

        # Run training
        trainer.train()

        training_time = time.time() - start_time
        final_metrics = {
            "training_success": True,
        }
        display_training_complete(final_metrics, training_time)

    except Exception as e:
        print(f"❌ Training failed: {e}")
        import traceback

        traceback.print_exc()
        return False

    return True


if __name__ == "__main__":
    success = train_v443_2_phase2()
    sys.exit(0 if success else 1)
