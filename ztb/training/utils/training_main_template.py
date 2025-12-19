"""
Template for main functions in training scripts.
"""

import argparse
from pathlib import Path
from typing import Any, Dict, Optional, Type

from ztb.training.trainers.base_trainer import BaseTrainer
from ztb.training.utils.common_utils import load_config_file
from ztb.training.utils.logging_utils import get_logger

logger = get_logger(__name__)


def create_simple_main_template(
    trainer_class: Type[BaseTrainer],
    config_path: str,
    description: str = "Training script",
    extra_info: str = ""
) -> callable:
    """
    Create a simple main function template for training scripts with fixed config.

    Args:
        trainer_class: The trainer class to instantiate
        config_path: Path to config file
        description: Description for logging
        extra_info: Extra information to print

    Returns:
        Main function
    """

    def main() -> None:
        print(f"🚀 {description}")
        print("=" * 60)
        if extra_info:
            print(extra_info)
            print()

        # Load configuration
        config = load_config_file(Path(config_path))
        logger.info(f"Loaded config from {config_path}")

        # Create and run trainer
        try:
            trainer = trainer_class(config)
            trainer.run_training()
            logger.info("Training completed successfully")
            print("✅ Training completed!")
        except Exception as e:
            logger.error(f"Training failed: {e}")
            print("❌ Training failed")
            raise

    return main