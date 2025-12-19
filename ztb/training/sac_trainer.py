#!/usr/bin/env python3
# ruff: noqa: E402
"""
SAC Training Suite - Unified training tools for SAC models

This script provides unified training capabilities for SAC trading models including:
- Standard SAC training
- Curriculum learning
- Multi-stage training
- Hyperparameter optimization integration
- Model validation
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

# Add project root to path
project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(project_root))

from ztb.training.components.regime_adaptive_trainer import RegimeAdaptiveTrainerMixin
from ztb.training.core.base_trainer import BaseTrainer
from ztb.training.unified_trainer import UnifiedTrainer
from ztb.types.common import ConfigDict
from ztb.utils.logging_utils import get_logger
from ztb.utils.path_utils import get_project_root

# Get project root using utility
project_root = get_project_root()

logger = get_logger(__name__)


class SACTrainer(BaseTrainer, RegimeAdaptiveTrainerMixin):
    """Unified SAC training interface."""

    def __init__(self, config_path: str, config: Optional[Dict[str, Any]] = None):
        super().__init__(name="SACTrainer", config=config)
        self.config_path = Path(config_path)
        self.config_data = self._load_config()
        self.trainer = None

        # Initialize regime adaptation
        regime_config = self.config_data.get("regime_adaptation", {})
        RegimeAdaptiveTrainerMixin.__init__(self, regime_config)

    def _load_config(self) -> Dict[str, Any]:
        """Load configuration file."""
        from ztb.training.utils.common_utils import load_config_file
        from ztb.training.utils.logging_utils import get_logger

        logger = get_logger(__name__)
        config = load_config_file(self.config_path)
        logger.info(f"Config loaded from {self.config_path}")
        return config

    def setup_trainer(self):
        """Setup the unified trainer."""
        self.trainer = UnifiedTrainer(self.config)
        logger.info("Trainer initialized")

    def run_training(
        self,
        total_timesteps: Optional[int] = None,
        output_dir: Optional[str] = None,
        resume_path: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Run SAC training.

        Args:
            total_timesteps: Override total timesteps
            output_dir: Override output directory
            resume_path: Path to training state file for resuming training

        Returns:
            Training results
        """
        if not self.trainer:
            self.setup_trainer()

        # Override config if specified
        if total_timesteps:
            self.config["training"]["total_timesteps"] = total_timesteps
            logger.info(f"Total timesteps set to: {total_timesteps}")

        if output_dir:
            self.config["output_dir"] = output_dir
            logger.info(f"Output directory set to: {output_dir}")

        if resume_path:
            self.config["training"]["resume_from"] = resume_path
            logger.info(f"Resume path set to: {resume_path}")

        logger.info("Starting SAC training...")
        logger.info(f"Model: {self.config.get('model_name', 'SAC')}")
        logger.info(f"Total Timesteps: {self.config['training']['total_timesteps']}")

        # Initialize regime adaptation if enabled
        if self.regime_adaptation_enabled:
            logger.info("Regime adaptation enabled - monitoring market conditions")

        try:
            results = self.trainer.run()
            logger.info("Training completed successfully")
            return results

        except Exception as e:
            logger.error(f"Training failed: {e}")
            return {"success": False, "error": str(e)}

    def run_curriculum_training(self, stages: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Run curriculum learning with multiple stages.

        Args:
            stages: List of training stages with their configurations

        Returns:
            Curriculum training results
        """
        logger.info(f"Starting curriculum training with {len(stages)} stages")

        curriculum_results = []
        current_config = self.config.copy()

        for i, stage in enumerate(stages):
            logger.info(
                f"Stage {i+1}/{len(stages)}: {stage.get('name', f'Stage {i+1}')}"
            )

            # Update config for this stage
            stage_config = current_config.copy()
            stage_config.update(stage)

            # Create trainer for this stage
            stage_trainer = UnifiedTrainer(stage_config)

            try:
                stage_results = stage_trainer.run()
                curriculum_results.append(
                    {
                        "stage": i + 1,
                        "name": stage.get("name", f"Stage {i+1}"),
                        "config": stage,
                        "results": stage_results,
                        "success": True,
                    }
                )

                # Use final model from this stage as starting point for next
                if "model_path" in stage_results:
                    current_config["load_path"] = stage_results["model_path"]

            except Exception as e:
                logger.error(f"Stage {i+1} failed: {e}")
                curriculum_results.append(
                    {
                        "stage": i + 1,
                        "name": stage.get("name", f"Stage {i+1}"),
                        "error": str(e),
                        "success": False,
                    }
                )
                break

        return {
            "curriculum_training": True,
            "total_stages": len(stages),
            "completed_stages": len([r for r in curriculum_results if r["success"]]),
            "stage_results": curriculum_results,
        }

    def validate_training(self, model_path: Optional[str] = None) -> Dict[str, Any]:
        """
        Validate trained model.

        Args:
            model_path: Path to model to validate

        Returns:
            Validation results
        """
        if not model_path and self.trainer:
            # Try to get model path from trainer results
            model_path = getattr(self.trainer, "model_path", None)

        if not model_path:
            return {"validation_error": "No model path available"}

        logger.info(f"Validating model: {model_path}")

        # Basic validation - check if model loads
        try:
            from stable_baselines3 import SAC

            model = SAC.load(model_path)
            validation_results = {
                "model_loaded": True,
                "model_path": model_path,
                "observation_space": str(model.observation_space),
                "action_space": str(model.action_space),
            }
            logger.info("Model validation successful")
            return validation_results

        except Exception as e:
            logger.error(f"Model validation failed: {e}")
            return {"model_loaded": False, "validation_error": str(e)}

    def train(self, data: ConfigDict) -> Dict[str, Any]:
        """
        Train the SAC model. Required by BaseTrainer.

        Args:
            data: Training configuration/data

        Returns:
            Training results
        """
        # Extract training parameters from data if provided
        total_timesteps = (
            data.get("total_timesteps") if isinstance(data, dict) else None
        )
        output_dir = data.get("output_dir") if isinstance(data, dict) else None

        return self.run_training(total_timesteps=total_timesteps, output_dir=output_dir)

    def evaluate(self, data: ConfigDict) -> Dict[str, Any]:
        """
        Evaluate the SAC model. Required by BaseTrainer.

        Args:
            data: Evaluation configuration/data

        Returns:
            Evaluation results
        """
        # Extract model path from data if provided
        model_path = data.get("model_path") if isinstance(data, dict) else None

        return self.validate_training(model_path=model_path)

    def _load_model(self, path: str) -> Any:
        """
        Load SAC model. Required by BaseTrainer.

        Args:
            path: Path to model file

        Returns:
            Loaded model
        """
        from stable_baselines3 import SAC

        return SAC.load(path)

    # RegimeAdaptiveTrainerMixin abstract method implementations
    def apply_hyperparameter_adaptation(self, adapted_params: Dict[str, Any]):
        """
        Apply adapted hyperparameters to the training process

        Args:
            adapted_params: Dictionary of parameters to apply
        """
        if not self.trainer:
            logger.warning(
                "No trainer initialized, cannot apply hyperparameter adaptation"
            )
            return

        try:
            # Apply parameters to the underlying trainer if it supports adaptation
            if hasattr(self.trainer, "update_hyperparameters"):
                self.trainer.update_hyperparameters(adapted_params)
                logger.info(f"Applied hyperparameter adaptation: {adapted_params}")
            else:
                logger.warning("Trainer does not support hyperparameter adaptation")
        except Exception as e:
            logger.error(f"Failed to apply hyperparameter adaptation: {e}")

    def get_current_market_data(self) -> Optional[pd.DataFrame]:
        """
        Get current market data for regime detection

        Returns:
            DataFrame with market data or None
        """
        # Try to get data from trainer or config
        if hasattr(self.trainer, "get_current_data"):
            return self.trainer.get_current_data()

        # Fallback to config data path
        data_config = self.config_data.get("training", {}).get("data_config", {})
        data_path = data_config.get("data_path") or data_config.get("csv_path")

        if data_path and Path(data_path).exists():
            try:
                return pd.read_csv(data_path)
            except Exception as e:
                logger.warning(f"Failed to load market data from {data_path}: {e}")

        return None

    def get_current_step_count(self) -> int:
        """
        Get current training step count

        Returns:
            Current step count
        """
        if hasattr(self.trainer, "get_step_count"):
            return self.trainer.get_step_count()

        # Fallback to config
        return self.config_data.get("training", {}).get("total_timesteps", 0)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="SAC Training Suite")
    parser.add_argument(
        "--config", type=str, required=True, help="Path to configuration file"
    )
    parser.add_argument("--timesteps", type=int, help="Override total timesteps")
    parser.add_argument("--output-dir", type=str, help="Override output directory")
    parser.add_argument(
        "--curriculum", type=str, help="Path to curriculum config (JSON)"
    )
    parser.add_argument(
        "--validate", action="store_true", help="Validate trained model"
    )
    parser.add_argument("--model-path", type=str, help="Path to model for validation")
    parser.add_argument(
        "--resume", type=str, help="Path to training state file for resuming training"
    )

    args = parser.parse_args()

    # Initialize trainer
    trainer = SACTrainer(args.config)

    if args.curriculum:
        # Run curriculum training
        try:
            with open(args.curriculum, "r", encoding="utf-8") as f:
                curriculum_config = json.load(f)

            stages = curriculum_config.get("stages", [])
            results = trainer.run_curriculum_training(stages)

        except Exception as e:
            logger.error(f"Curriculum training failed: {e}")
            results = {"error": str(e)}

    else:
        # Run standard training
        results = trainer.run_training(args.timesteps, args.output_dir, args.resume)

    # Print results summary
    print("\n" + "=" * 60)
    print("SAC TRAINING RESULTS")
    print("=" * 60)

    if "success" in results and results["success"]:
        print("✅ Training completed successfully")
        if "model_path" in results:
            print(f"📁 Model saved to: {results['model_path']}")
    else:
        print("❌ Training failed")
        if "error" in results:
            print(f"Error: {results['error']}")

    if args.validate or args.model_path:
        validation_results = trainer.validate_training(args.model_path)
        print("\n🔍 MODEL VALIDATION:")
        if validation_results.get("model_loaded", False):
            print("✅ Model validation successful")
            print(
                f"  Observation space: {validation_results.get('observation_space', 'N/A')}"
            )
            print(f"  Action space: {validation_results.get('action_space', 'N/A')}")
        else:
            print("❌ Model validation failed")
            print(f"  Error: {validation_results.get('validation_error', 'Unknown')}")

    print("=" * 60)


if __name__ == "__main__":
    main()
