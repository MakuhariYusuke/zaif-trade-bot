#!/usr/bin/env python3
"""
SAC v427 Training Script

Execute complete SAC v427 training with advanced ML techniques.
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from ztb.training.sac_v427_advanced_trainer import SACv427AdvancedTrainer
from ztb.utils.logging_utils import get_logger

logger = get_logger(__name__)


def main():
    """Main training execution."""
    try:
        # Configuration path
        config_path = "configs/sac_v427_market_adaptive_ensemble.json"

        if not Path(config_path).exists():
            logger.error(f"Configuration file not found: {config_path}")
            return 1

        logger.info("Initializing SAC v427 Advanced Trainer...")
        trainer = SACv427AdvancedTrainer(config_path)

        logger.info("Starting SAC v427 training pipeline...")
        results = trainer.train_v427_system()

        logger.info("Training completed successfully!")
        logger.info(f"Final model: {results.get('final_model', 'N/A')}")
        logger.info(
            f"Advanced techniques used: {results.get('advanced_techniques_used', [])}"
        )

        # Print summary
        print("\n" + "=" * 60)
        print("SAC v427 TRAINING SUMMARY")
        print("=" * 60)
        print(f"Training completed: {results.get('training_end', 'N/A')}")
        print(f"Final model: {results.get('final_model', 'N/A')}")
        print(
            f"Advanced techniques: {', '.join(results.get('advanced_techniques_used', []))}"
        )

        if "performance_metrics" in results:
            metrics = results["performance_metrics"]
            print(f"Final reward: {metrics.get('final_reward', 'N/A')}")
            print(f"Max drawdown: {metrics.get('max_drawdown', 'N/A')}")
            print(f"Annual return: {metrics.get('annual_return', 'N/A')}")

        print("=" * 60)

        return 0

    except Exception as e:
        logger.error(f"Training failed: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
