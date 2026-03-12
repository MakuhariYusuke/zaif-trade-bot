"""
Template for main functions in training scripts.
"""

import argparse
from pathlib import Path
from typing import Any, Callable

from ztb.training.trainers.base_trainer import BaseTrainer
from ztb.utils.file_utils import safe_json_load
from ztb.utils.logging_utils import get_logger

logger = get_logger(__name__)

def create_simple_main_template(
    trainer_class: type[BaseTrainer],
    config_path: str,
    description: str = "Training script",
    extra_info: str = "",
    use_argparse: bool = False,
) -> Callable[[], None]:
    """
    Create a simple main function template for training scripts with fixed config.

    Args:
        trainer_class: The trainer class to instantiate
        config_path: Path to config file
        description: Description for logging
        extra_info: Extra information to print
        use_argparse: Whether to use argparse for config override

    Returns:
        Main function
    """

    def main() -> None:
        if use_argparse:
            parser = argparse.ArgumentParser(description=description)
            parser.add_argument(
                "--config",
                type=str,
                default=config_path,
                help=f"Path to configuration file (default: {config_path})",
            )
            args = parser.parse_args()
            actual_config_path = args.config
        else:
            actual_config_path = config_path

        print(f"🚀 {description}")
        print("=" * 60)
        if extra_info:
            print(extra_info)
            print()

        # Load configuration
        config = safe_json_load(Path(actual_config_path))
        logger.info(f"Loaded config from {actual_config_path}")

        # Create and run trainer
        try:
            trainer = trainer_class(config)
            trainer.run_training()
            # Try to extract final metrics/report if available
            final_metrics: dict[str, Any] = {}
            training_time: float = 0.0
            try:
                if hasattr(trainer, "training_report") and trainer.training_report:
                    final_metrics = (
                        trainer.training_report.get("training_stats", {}) or {}
                    )
                    training_time = final_metrics.get("training_time", 0.0)
                elif hasattr(trainer, "reporter") and hasattr(
                    trainer.reporter, "generate_report"
                ):
                    report = trainer.reporter.generate_report(trainer.config, {}, True)
                    final_metrics = report.get("training_stats", {}) or {}
                    training_time = final_metrics.get("training_time", 0.0)
            except Exception:
                final_metrics = {}
                training_time = 0.0

            from ztb.utils.training_utils import display_training_complete

            display_training_complete(final_metrics, training_time)
        except Exception as e:
            logger.error(f"Training failed: {e}")
            # Attempt to display failure summary
            from ztb.utils.training_utils import display_training_complete

            training_time = (
                getattr(trainer, "training_time", 0.0) if "trainer" in locals() else 0.0
            )
            display_training_complete({}, training_time)
            raise

    return main
