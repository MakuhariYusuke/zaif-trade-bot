#!/usr/bin/env python3
# ruff: noqa: E402
"""
SAC v432.1 Training Script with Advanced Position Management
Enhanced position management with negative HOLD penalty
"""

import json
import sys
from pathlib import Path

# Add project root to path using path_utils
project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(project_root))

from ztb.training.unified_trainer.config import load_config
from ztb.training.unified_trainer.trainer import UnifiedTrainer
from ztb.utils.logging_utils import get_logger
from ztb.utils.path_utils import get_project_root

logger = get_logger(__name__)


def train_sac_v432_1():
    """Train SAC v432.1 with Advanced Position Management"""
    logger.info("Starting SAC v432.1 training with advanced position management")

    # Load configuration
    config_path = (
        get_project_root()
        / "ztb"
        / "configs"
        / "v432"
        / "sac_v432_1_advanced_position_management.json"
    )
    config = load_config(str(config_path))

    # Initialize Unified Trainer
    trainer = UnifiedTrainer(config=config)

    # Set up results directory
    results_dir = get_project_root() / "ztb" / "reports" / "v432"
    results_dir.mkdir(parents=True, exist_ok=True)

    try:
        # Run training
        logger.info("Starting training with advanced position management...")
        results = trainer.train()

        # Save results manually
        results_file = results_dir / "training_results_v432_1.json"
        with open(results_file, "w") as f:
            json.dump(results, f, indent=2, default=str)

        # Try to extract final metrics and training time from trainer
        final_metrics = {}
        training_time = getattr(trainer, "training_time", 0.0)
        try:
            if hasattr(trainer, "training_report") and trainer.training_report:
                final_metrics = trainer.training_report.get("training_stats", {}) or {}
                training_time = final_metrics.get("training_time", training_time)
        except Exception:
            final_metrics = {}

        from ztb.utils.training_utils import display_training_complete

        display_training_complete(final_metrics, training_time)

        print(f"Results saved to: {results_file}")

        return results

    except Exception as e:
        logger.error(f"Training failed: {e}")
        raise


def main():
    """Main function"""
    try:
        train_sac_v432_1()
        print("🎉 SAC v432.1 training completed!")
        return 0
    except Exception as e:
        print(f"\n❌ Training failed: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
