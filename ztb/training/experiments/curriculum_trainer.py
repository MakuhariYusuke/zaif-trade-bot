"""
Curriculum Learning Trainer with Forced Diversity and Progressive Rewards
カリキュラム学習トレーナー - 強制多様性と累進報酬付き

Three-phase training:
  Phase 1: HOLD禁止 (Action masking: HOLD=impossible)
  Phase 2: HOLD制限 (HOLD allowed max 20% of time)
  Phase 3: 通常 (No restrictions, full learning)

Features:
  - Action diversity tracking and enforcement
  - Progressive reward shaping
  - Consecutive action penalties
  - Milestone bonuses
"""

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, cast

from ztb.training.core.unified_trainer import UnifiedTrainer
from ztb.utils.config import ZTBConfig

logger = logging.getLogger(__name__)


class CurriculumPhase:
    """Configuration for a single curriculum phase"""

    def __init__(
        self,
        phase_id: int,
        name: str,
        timesteps: int,
        hold_restriction: str = "none",  # "forbidden", "limited_20", "none"
        consecutive_action_penalty: float = 0.0,
        diversity_bonus_weight: float = 0.0,
        min_diversity_threshold: float = 0.0,
    ):
        self.phase_id = phase_id
        self.name = name
        self.timesteps = timesteps
        self.hold_restriction = hold_restriction
        self.consecutive_action_penalty = consecutive_action_penalty
        self.diversity_bonus_weight = diversity_bonus_weight
        self.min_diversity_threshold = min_diversity_threshold

    def __repr__(self) -> str:
        return (
            f"Phase {self.phase_id}: {self.name}\n"
            f"  Timesteps: {self.timesteps}\n"
            f"  HOLD restriction: {self.hold_restriction}\n"
            f"  Consecutive penalty: {self.consecutive_action_penalty}\n"
            f"  Diversity bonus: {self.diversity_bonus_weight}\n"
            f"  Min diversity: {self.min_diversity_threshold}"
        )


class CurriculumTrainer:
    """Curriculum learning trainer with progressive difficulty"""

    def __init__(
        self,
        base_config_path: str,
        output_dir: Optional[str] = None,
        experiment_name: str = "curriculum_v1",
    ):
        if output_dir is None:
            output_dir = ZTBConfig().get_model_path("curriculum")
        self.base_config_path = Path(base_config_path)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.experiment_name = experiment_name

        # Load base configuration
        with open(self.base_config_path) as f:
            self.base_config = json.load(f)

        # Define curriculum phases
        self.phases = self._define_curriculum_phases()

        # Training history
        self.phase_results: List[Dict[str, Any]] = []

        logger.info(f"Curriculum Trainer initialized: {experiment_name}")
        logger.info(f"Base config: {self.base_config_path}")
        logger.info(f"Output dir: {self.output_dir}")

    def _define_curriculum_phases(self) -> list[CurriculumPhase]:
        """Define the three-phase curriculum"""
        phases = [
            CurriculumPhase(
                phase_id=1,
                name="HOLD禁止 (Forced Active Trading)",
                timesteps=5000,
                hold_restriction="forbidden",
                consecutive_action_penalty=0.01,  # Strong penalty for same action
                diversity_bonus_weight=0.1,  # Strong bonus for diversity
                min_diversity_threshold=0.4,  # At least 40% each BUY/SELL
            ),
            CurriculumPhase(
                phase_id=2,
                name="HOLD制限20% (Limited HOLD)",
                timesteps=5000,
                hold_restriction="limited_20",
                consecutive_action_penalty=0.005,  # Moderate penalty
                diversity_bonus_weight=0.05,  # Moderate bonus
                min_diversity_threshold=0.3,  # At least 30% each action
            ),
            CurriculumPhase(
                phase_id=3,
                name="通常 (Normal Training)",
                timesteps=10000,
                hold_restriction="none",
                consecutive_action_penalty=0.002,  # Light penalty
                diversity_bonus_weight=0.01,  # Light bonus
                min_diversity_threshold=0.2,  # At least 20% active trading
            ),
        ]

        logger.info("\n" + "=" * 80)
        logger.info("CURRICULUM PHASES DEFINED")
        logger.info("=" * 80)
        for phase in phases:
            logger.info(f"\n{phase}")
        logger.info("=" * 80)

        return phases

    def _create_phase_config(
        self, phase: CurriculumPhase, prev_model_path: Optional[str] = None
    ) -> Dict[str, Any]:
        """Create configuration for a specific phase"""
        # Copy base config
        phase_config = self.base_config.copy()

        # Update session ID
        phase_config["session_id"] = f"{self.experiment_name}_phase{phase.phase_id}"

        # Update training timesteps
        phase_config["training"]["total_timesteps"] = phase.timesteps

        # Update environment settings for curriculum
        if "environment" not in phase_config:
            phase_config["environment"] = {}

        if "reward_settings" not in phase_config["environment"]:
            phase_config["environment"]["reward_settings"] = {}

        # Add curriculum-specific reward settings
        phase_config["environment"]["reward_settings"][
            "consecutive_action_penalty"
        ] = phase.consecutive_action_penalty
        phase_config["environment"]["reward_settings"][
            "diversity_bonus_weight"
        ] = phase.diversity_bonus_weight
        phase_config["environment"]["reward_settings"][
            "min_diversity_threshold"
        ] = phase.min_diversity_threshold

        # Add HOLD restriction settings
        phase_config["environment"]["hold_restriction"] = phase.hold_restriction
        phase_config["environment"]["enable_action_masking"] = True

        # If we have a previous model, use it for warm start
        if prev_model_path:
            phase_config["warm_start_model"] = prev_model_path

        return cast(Dict[str, Any], phase_config)

    def run_phase(
        self, phase: CurriculumPhase, prev_model_path: Optional[str] = None
    ) -> Tuple[str, Dict[str, Any]]:
        """Run a single curriculum phase"""
        logger.info("\n" + "=" * 80)
        logger.info(f"STARTING PHASE {phase.phase_id}: {phase.name}")
        logger.info("=" * 80)

        # Create phase configuration
        phase_config = self._create_phase_config(phase, prev_model_path)

        # Save phase config
        config_path = self.output_dir / f"phase{phase.phase_id}_config.json"
        with open(config_path, "w") as f:
            json.dump(phase_config, f, indent=2)
        logger.info(f"Phase config saved: {config_path}")

        # Create trainer
        trainer = UnifiedTrainer(config=phase_config, total_timesteps=phase.timesteps)

        # Train
        logger.info(f"\nTraining for {phase.timesteps:,} timesteps...")
        model = trainer.train()

        # Save model
        model_path = self.output_dir / f"phase{phase.phase_id}_model.zip"
        model.save(str(model_path))
        logger.info(f"Phase {phase.phase_id} model saved: {model_path}")

        # Collect phase results
        results = {
            "phase_id": phase.phase_id,
            "phase_name": phase.name,
            "timesteps": phase.timesteps,
            "model_path": str(model_path),
            "config_path": str(config_path),
            "hold_restriction": phase.hold_restriction,
        }

        logger.info(f"\n✅ PHASE {phase.phase_id} COMPLETED")
        logger.info("=" * 80)

        return str(model_path), results

    def run_full_curriculum(self) -> Dict[str, Any]:
        """Run all curriculum phases sequentially"""
        logger.info("\n" + "#" * 80)
        logger.info(f"# STARTING CURRICULUM LEARNING: {self.experiment_name}")
        logger.info("#" * 80)

        prev_model_path = None

        for phase in self.phases:
            model_path, results = self.run_phase(phase, prev_model_path)
            self.phase_results.append(results)
            prev_model_path = model_path  # Use this model for warm start in next phase

        # Save curriculum results
        results_path = self.output_dir / "curriculum_results.json"
        with open(results_path, "w") as f:
            json.dump(
                {
                    "experiment_name": self.experiment_name,
                    "base_config": str(self.base_config_path),
                    "total_phases": len(self.phases),
                    "phases": self.phase_results,
                },
                f,
                indent=2,
            )

        logger.info("\n" + "#" * 80)
        logger.info("# CURRICULUM LEARNING COMPLETED")
        logger.info("#" * 80)
        logger.info(f"Final model: {prev_model_path}")
        logger.info(f"Results saved: {results_path}")
        logger.info("#" * 80)

        return {
            "final_model_path": prev_model_path,
            "results_path": str(results_path),
            "phase_results": self.phase_results,
        }


def main() -> None:
    """Main curriculum learning execution"""
    import argparse

    parser = argparse.ArgumentParser(description="Curriculum Learning Trainer")
    parser.add_argument("--config", required=True, help="Base configuration file")
    parser.add_argument(
        "--output-dir", default="models/curriculum", help="Output directory"
    )
    parser.add_argument("--name", default="curriculum_v1", help="Experiment name")

    args = parser.parse_args()

    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    # Create and run curriculum trainer
    trainer = CurriculumTrainer(
        base_config_path=args.config,
        output_dir=args.output_dir,
        experiment_name=args.name,
    )

    results = trainer.run_full_curriculum()

    print("\n" + "=" * 80)
    print("✅ CURRICULUM LEARNING COMPLETED SUCCESSFULLY")
    print("=" * 80)
    print(f"Final model: {results['final_model_path']}")
    print(f"Results: {results['results_path']}")
    print("=" * 80)


if __name__ == "__main__":
    main()
