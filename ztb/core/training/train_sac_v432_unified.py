#!/usr/bin/env python3
"""
SAC v432 Training Script using Unified Trainer
Enhanced ensemble learning with optimized reward structure
"""

import json
import sys
from pathlib import Path

# Add project root to path using path_utils
project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(project_root))

from ztb.training.unified_trainer.config import get_training_config, load_config
from ztb.training.unified_trainer.ensemble_system import EnsembleConfig
from ztb.training.unified_trainer.trainer import UnifiedTrainer
from ztb.utils.logging_utils import get_logger
from ztb.utils.path_utils import get_project_root

logger = get_logger(__name__)


def create_v432_ensemble_config() -> EnsembleConfig:
    """Create optimized ensemble configuration for v432"""
    return EnsembleConfig(
        enabled=True,
        members=5,
        specializations=["bull", "bear", "sideways", "high_vol", "low_vol"],
        voting_mechanism="weighted_confidence",
        diversity_weight=0.4,
        consensus_requirement={
            "enabled": True,
            "min_agreement": 0.7,
            "max_confidence_gap": 0.3,
        },
    )


def train_sac_v432():
    """Train SAC v432 using Unified Trainer"""
    print("=" * 80)
    print("SAC v432 Training with Unified Trainer")
    print("=" * 80)

    # Get project root using path_utils
    project_root = get_project_root()
    config_path = (
        project_root / "ztb" / "configs" / "v432" / "sac_v432_0_ensemble_optimized.json"
    )

    # Load configuration
    logger.info(f"Loading configuration from {config_path}")
    config = load_config(str(config_path))
    training_config = get_training_config(config)

    # Create ensemble configuration
    ensemble_config = create_v432_ensemble_config()

    # Initialize Unified Trainer
    trainer = UnifiedTrainer(config=config)

    # Start training
    logger.info("Starting SAC v432 training with ensemble learning")
    try:
        results = trainer.train()
        logger.info("Training completed successfully")

        # Save results manually
        results_dir = project_root / "ztb" / "reports" / "v432"
        results_dir.mkdir(parents=True, exist_ok=True)

        results_file = results_dir / "training_results.json"
        with open(results_file, "w") as f:
            json.dump(results, f, indent=2, default=str)

        print("✅ Training completed successfully!")
        print(f"Results saved to: {results_file}")

        return results

    except Exception as e:
        logger.error(f"Training failed: {e}")
        raise


def main():
    """Main function"""
    try:
        results = train_sac_v432()
        print("🎉 SAC v432 training completed!")
        return 0
    except Exception as e:
        print(f"\n❌ Training failed: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
