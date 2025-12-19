#!/usr/bin/env python3
# ruff: noqa: E402
"""
SAC v430 Advanced Training Suite - Efficient Learning with Curriculum & Optimization

This script implements advanced training techniques for SAC v430:
- Curriculum learning with progressive difficulty
- Multi-stage training with different objectives
- Memory-efficient training with gradient accumulation
- Parallel validation and early stopping
- Dynamic hyperparameter adjustment
- Comprehensive logging and monitoring

Usage:
    python train_v430_advanced.py --config configs/v430/sac_v430_optimized.json --mode curriculum
"""

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

import torch
import copy

from ztb.utils.file_utils import get_project_root

# Add project root to path
project_root = get_project_root()
sys.path.insert(0, str(project_root))

from ztb.evaluation.evaluator import Evaluator
from ztb.training.unified_trainer import UnifiedTrainer
from ztb.training.unified_trainer.parallel_trainer import ParallelTrainer
from ztb.utils.logging_utils import get_logger
from ztb.utils.memory_utils import OperationMemoryTracker
from ztb.utils.performance_profiler import PerformanceProfiler

logger = get_logger(__name__)


class CurriculumStage:
    """Represents a single stage in curriculum learning."""

    def __init__(
        self,
        name: str,
        timesteps: int,
        config_mods: Dict[str, Any],
        success_criteria: Dict[str, float],
    ):
        self.name = name
        self.timesteps = timesteps
        self.config_mods = config_mods
        self.success_criteria = success_criteria


class SACv430AdvancedTrainer:
    """Advanced SAC v430 trainer with curriculum learning and optimization."""

    def __init__(self, config_path: str, mode: str = "standard") -> None:
        """
        Initialize advanced trainer.

        Args:
            config_path: Path to v430 configuration
            mode: Training mode ('standard', 'curriculum', 'multi_stage', 'ensemble')
        """
        self.config_path = Path(config_path)
        self.mode = mode
        self.base_config = self._load_config()
        self.memory_tracker = OperationMemoryTracker()
        self.performance_profiler = PerformanceProfiler()

        # Setup output directory
        self.output_dir = Path("runs") / f"sac_v430_{mode}_{int(time.time())}"
        self.output_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"Initialized SAC v430 Advanced Trainer (mode: {mode})")
        logger.info(f"Output directory: {self.output_dir}")

    def _load_config(self) -> Dict[str, Any]:
        """Load and validate v430 configuration."""
        with open(self.config_path, "r", encoding="utf-8") as f:
            config = json.load(f)

        # Validate required fields
        required_fields = ["training", "reward_function", "version"]
        for field in required_fields:
            if field not in config:
                raise ValueError(f"Missing required field: {field}")

        if config.get("version") != "v430":
            logger.warning(f"Config version is {config.get('version')}, expected v430")

        return config

    def _create_curriculum_stages(self) -> List[CurriculumStage]:
        """Create curriculum learning stages for progressive difficulty."""

        stages = [
            CurriculumStage(
                name="warmup",
                timesteps=10000,
                config_mods={
                    "training": {
                        "learning_rate": self.base_config["training"]["learning_rate"]
                        * 2,
                        "ent_coef": "auto_0.1",  # Higher exploration
                        "batch_size": min(
                            128, self.base_config["training"]["batch_size"]
                        ),
                    },
                    "reward_function": {
                        "reward_scale": self.base_config["reward_function"][
                            "reward_scale"
                        ]
                        * 0.5,
                        "risk_penalty": self.base_config["reward_function"][
                            "risk_penalty"
                        ]
                        * 0.5,
                    },
                },
                success_criteria={
                    "avg_reward": -1.0,
                    "ep_length": 50,
                },  # Basic stability
            ),
            CurriculumStage(
                name="foundation",
                timesteps=25000,
                config_mods={
                    "training": {
                        "learning_rate": self.base_config["training"]["learning_rate"]
                        * 1.5,
                        "ent_coef": "auto_0.05",
                        "batch_size": min(
                            256, self.base_config["training"]["batch_size"]
                        ),
                    },
                    "reward_function": {
                        "reward_scale": self.base_config["reward_function"][
                            "reward_scale"
                        ]
                        * 0.8,
                        "action_balance_weight": self.base_config["reward_function"][
                            "action_balance_weight"
                        ]
                        * 1.2,
                    },
                },
                success_criteria={
                    "avg_reward": -0.5,
                    "win_rate": 0.45,
                    "ep_length": 100,
                },
            ),
            CurriculumStage(
                name="optimization",
                timesteps=50000,
                config_mods={
                    "training": {
                        "learning_rate": self.base_config["training"]["learning_rate"],
                        "ent_coef": "auto_0.01",
                        "batch_size": self.base_config["training"]["batch_size"],
                    },
                    "reward_function": self.base_config[
                        "reward_function"
                    ],  # Use optimized parameters
                },
                success_criteria={
                    "avg_reward": 0.0,
                    "win_rate": 0.52,
                    "sharpe_ratio": 0.8,
                    "max_drawdown": 0.25,
                },
            ),
            CurriculumStage(
                name="refinement",
                timesteps=25000,
                config_mods={
                    "training": {
                        "learning_rate": self.base_config["training"]["learning_rate"]
                        * 0.5,
                        "ent_coef": "auto_0.005",
                        "tau": self.base_config["training"]["tau"]
                        * 0.8,  # Slower target updates
                    },
                    "reward_function": {
                        "risk_penalty": self.base_config["reward_function"][
                            "risk_penalty"
                        ]
                        * 1.2,
                        "hold_penalty": self.base_config["reward_function"][
                            "hold_penalty"
                        ]
                        * 0.8,
                    },
                },
                success_criteria={
                    "avg_reward": 0.2,
                    "win_rate": 0.55,
                    "sharpe_ratio": 1.0,
                    "max_drawdown": 0.2,
                },
            ),
        ]

        return stages

    def _apply_config_modifications(
        self, base_config: Dict[str, Any], modifications: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Apply configuration modifications recursively."""
        config = copy.deepcopy(base_config)

        def update_dict_recursive(target: Dict[str, Any], updates: Dict[str, Any]):
            for key, value in updates.items():
                if (
                    isinstance(value, dict)
                    and key in target
                    and isinstance(target[key], dict)
                ):
                    update_dict_recursive(target[key], value)
                else:
                    target[key] = value

        update_dict_recursive(config, modifications)
        return config

    def _evaluate_stage_success(
        self, stage: CurriculumStage, metrics: Dict[str, Any]
    ) -> bool:
        """Evaluate if a curriculum stage was successful."""
        success = True

        for metric, threshold in stage.success_criteria.items():
            if metric in metrics:
                value = metrics[metric]
                if metric in ["max_drawdown"]:  # Lower is better
                    if value > threshold:
                        logger.warning(
                            f"Stage {stage.name}: {metric} {value:.4f} > {threshold:.4f}"
                        )
                        success = False
                else:  # Higher is better
                    if value < threshold:
                        logger.warning(
                            f"Stage {stage.name}: {metric} {value:.4f} < {threshold:.4f}"
                        )
                        success = False
            else:
                logger.warning(f"Stage {stage.name}: Missing metric {metric}")
                success = False

        return success

    def train_curriculum(self) -> bool:
        """Run curriculum learning with progressive difficulty."""
        logger.info("🚀 Starting Curriculum Learning for SAC v430")
        print("\n🎯 Curriculum Learning Stages:")
        print("=" * 60)

        stages = self._create_curriculum_stages()
        cumulative_timesteps = 0

        for i, stage in enumerate(stages):
            print(f"\n📚 Stage {i+1}/{len(stages)}: {stage.name.upper()}")
            print(f"   Timesteps: {stage.timesteps:,}")
            print(f"   Cumulative: {cumulative_timesteps + stage.timesteps:,}")
            print("-" * 40)

            # Apply stage modifications
            stage_config = self._apply_config_modifications(
                self.base_config, stage.config_mods
            )
            stage_config["training"]["total_timesteps"] = stage.timesteps

            # Add stage-specific logging
            stage_config["logging"] = stage_config.get("logging", {})
            stage_config["logging"]["eval_interval"] = max(1000, stage.timesteps // 20)
            stage_config["logging"]["save_interval"] = max(5000, stage.timesteps // 10)

            # Train stage
            trainer = UnifiedTrainer(stage_config)
            stage_output_dir = self.output_dir / f"stage_{stage.name}"
            stage_output_dir.mkdir(exist_ok=True)

            # Redirect output to stage directory
            stage_config.get("logging", {}).get(
                "model_dir", "models"
            )
            stage_config["logging"]["model_dir"] = str(stage_output_dir / "models")

            try:
                success = trainer.train()
                if not success:
                    logger.error(f"Stage {stage.name} training failed")
                    return False

                # Evaluate stage performance
                evaluator = Evaluator()
                metrics = evaluator.evaluate_model(
                    model_path=stage_output_dir / "models" / "final_model.zip",
                    n_episodes=10,
                )

                if self._evaluate_stage_success(stage, metrics):
                    logger.info(f"✅ Stage {stage.name} completed successfully")
                    cumulative_timesteps += stage.timesteps
                else:
                    logger.warning(
                        f"⚠️ Stage {stage.name} did not meet success criteria, continuing..."
                    )

            except Exception as e:
                logger.error(f"Stage {stage.name} failed with error: {e}")
                return False

        # Final consolidation training
        logger.info("🔄 Running final consolidation training...")
        final_config = copy.deepcopy(self.base_config)
        final_config["training"]["total_timesteps"] = 25000
        final_config["logging"] = final_config.get("logging", {})
        final_config["logging"]["model_dir"] = str(self.output_dir / "final_model")

        trainer = UnifiedTrainer(final_config)
        success = trainer.train()

        if success:
            logger.info("🎉 Curriculum learning completed successfully!")
            self._save_training_summary()
        else:
            logger.error("❌ Final consolidation training failed")

        return success

    def train_multi_stage(self) -> bool:
        """Run multi-stage training with different objectives."""
        logger.info("🚀 Starting Multi-Stage Training for SAC v430")

        # Stage 1: Exploration-focused
        exploration_config = self._apply_config_modifications(
            self.base_config,
            {
                "training": {
                    "total_timesteps": 20000,
                    "ent_coef": "auto_0.1",
                    "learning_rate": self.base_config["training"]["learning_rate"] * 2,
                },
                "reward_function": {
                    "action_balance_weight": 0.8,
                    "risk_penalty": 0.0,
                },  # Focus on exploration
            },
        )

        # Stage 2: Exploitation-focused
        exploitation_config = self._apply_config_modifications(
            self.base_config,
            {
                "training": {
                    "total_timesteps": 30000,
                    "ent_coef": "auto_0.01",
                    "learning_rate": self.base_config["training"]["learning_rate"]
                    * 0.5,
                },
                "reward_function": {
                    "action_balance_weight": 0.2,
                    "risk_penalty": self.base_config["reward_function"]["risk_penalty"]
                    * 1.5,
                },
            },
        )

        # Stage 3: Fine-tuning
        finetune_config = self._apply_config_modifications(
            self.base_config,
            {
                "training": {
                    "total_timesteps": 20000,
                    "ent_coef": "auto_0.005",
                    "learning_rate": self.base_config["training"]["learning_rate"]
                    * 0.2,
                }
            },
        )

        stages = [
            ("exploration", exploration_config),
            ("exploitation", exploitation_config),
            ("finetune", finetune_config),
        ]

        for stage_name, config in stages:
            logger.info(f"📋 Starting {stage_name} stage...")
            trainer = UnifiedTrainer(config)
            success = trainer.train()
            if not success:
                logger.error(f"{stage_name} stage failed")
                return False

        logger.info("🎉 Multi-stage training completed successfully!")
        return True

    def train_ensemble(self) -> bool:
        """Train ensemble of SAC models with different seeds."""
        logger.info("🚀 Starting Ensemble Training for SAC v430")

        n_models = 5
        trainers = []

        for i in range(n_models):
            config = copy.deepcopy(self.base_config)
            config["training"]["seed"] = 42 + i  # Different seeds
            config["training"]["total_timesteps"] = 30000  # Shorter for ensemble
            config["logging"] = config.get("logging", {})
            config["logging"]["model_dir"] = str(
                self.output_dir / f"ensemble_model_{i}"
            )

            trainer = UnifiedTrainer(config)
            trainers.append(trainer)

        # Train in parallel if possible
        try:
            parallel_trainer = ParallelTrainer(trainers)
            success = parallel_trainer.train_all()
        except Exception:
                # Fallback to sequential training
            logger.warning("Parallel training failed, falling back to sequential...")
            success = True
            for i, trainer in enumerate(trainers):
                logger.info(f"Training ensemble model {i+1}/{n_models}")
                if not trainer.train():
                    success = False
                    break

        if success:
            logger.info("🎉 Ensemble training completed successfully!")
            self._create_ensemble_config(n_models)

        return success

    def _create_ensemble_config(self, n_models: int) -> None:
        """Create ensemble configuration file."""
        ensemble_config = {
            "version": "v430_ensemble",
            "description": "SAC v430 Ensemble Configuration",
            "ensemble": {
                "n_models": n_models,
                "model_paths": [
                    str(
                        self.output_dir
                        / f"ensemble_model_{i}"
                        / "models"
                        / "final_model.zip"
                    )
                    for i in range(n_models)
                ],
                "voting_method": "weighted_average",
                "weights": [1.0] * n_models,
            },
            "base_config": str(self.config_path),
        }

        ensemble_config_path = self.output_dir / "ensemble_config.json"
        with open(ensemble_config_path, "w", encoding="utf-8") as f:
            json.dump(ensemble_config, f, indent=2, ensure_ascii=False)

        logger.info(f"Ensemble configuration saved to {ensemble_config_path}")

    def _save_training_summary(self) -> None:
        """Save comprehensive training summary."""
        summary = {
            "version": "v430",
            "training_mode": self.mode,
            "start_time": time.time(),
            "config_path": str(self.config_path),
            "output_directory": str(self.output_dir),
            "system_info": {
                "torch_version": torch.__version__,
                "cuda_available": torch.cuda.is_available(),
                "gpu_count": torch.cuda.device_count()
                if torch.cuda.is_available()
                else 0,
            },
            "memory_usage": self.memory_tracker.get_summary(),
            "performance_profile": self.performance_profiler.get_summary(),
        }

        summary_path = self.output_dir / "training_summary.json"
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False, default=str)

    def train(self) -> bool:
        """Main training method based on selected mode."""
        start_time = time.time()

        try:
            self.memory_tracker.start_monitoring()
            self.performance_profiler.start_profiling()

            if self.mode == "curriculum":
                success = self.train_curriculum()
            elif self.mode == "multi_stage":
                success = self.train_multi_stage()
            elif self.mode == "ensemble":
                success = self.train_ensemble()
            else:  # standard
                trainer = UnifiedTrainer(self.base_config)
                success = trainer.train()

            training_time = time.time() - start_time
            logger.info(f"Training completed in {training_time:.2f} seconds")
            return success

        except Exception as e:
            logger.error(f"Training failed with error: {e}")
            import traceback

            traceback.print_exc()
            return False

        finally:
            self.memory_tracker.stop_monitoring()
            self.performance_profiler.stop_profiling()


def main() -> None:
    """Main function."""
    parser = argparse.ArgumentParser(description="SAC v430 Advanced Training Suite")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/v430/sac_v430_optimized.json",
        help="Path to v430 configuration file",
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["standard", "curriculum", "multi_stage", "ensemble"],
        default="curriculum",
        help="Training mode",
    )
    parser.add_argument("--output", type=str, help="Output directory override")

    args = parser.parse_args()

    print("🎯 SAC v430 Advanced Training Suite")
    print("=" * 60)
    print(f"Config: {args.config}")
    print(f"Mode: {args.mode}")
    print("=" * 60)

    trainer = SACv430AdvancedTrainer(args.config, args.mode)

    if args.output:
        trainer.output_dir = Path(args.output)

    success = trainer.train()

    if success:
        print("\n🎉 Training completed successfully!")
        print(f"📁 Results saved to: {trainer.output_dir}")
    else:
        print("\n❌ Training failed!")
        sys.exit(1)


if __name__ == "__main__":
    main()
