#!/usr/bin/env python3
"""
Algorithm Trainer Factory for Unified Training.

Provides a unified interface for all training algorithms.

REFACTORED (2025-10-11):
- PPO now uses AlgorithmFactory for pluggable algorithm architecture
- Other algorithms (base_ml, iterative, etc.) use legacy trainers (to be migrated)
"""

from typing import Any

# Legacy PPO trainer (deprecated, use AlgorithmFactory instead)
# from ztb.training.trainers.ppo_trainer import PPOAlgorithmTrainer
from ztb.training.algorithms import AlgorithmFactory  # 🆕 New architecture
from ztb.training.core.config_manager import ConfigManager
from ztb.training.trainers.base_ml_trainer import BaseMLAlgorithmTrainer
from ztb.training.trainers.curriculum_trainer import CurriculumAlgorithmTrainer
from ztb.training.trainers.ensemble_trainer import EnsembleAlgorithmTrainer
from ztb.training.trainers.iterative_trainer import IterativeAlgorithmTrainer
from ztb.utils.logging_utils import get_logger

logger = get_logger(__name__)

class AlgorithmTrainer:
    """
    Factory and interface for algorithm-specific trainers.

    ARCHITECTURE (2025-10-11):
    - PPO: Uses new AlgorithmFactory (pluggable architecture)
    - Others: Legacy trainers (to be migrated to AlgorithmFactory)
    """

    def __init__(
        self, config_manager: ConfigManager, progress_bar_enabled: bool = False
    ):
        """
        Initialize algorithm trainer.

        Args:
            config_manager: ConfigManager instance
            progress_bar_enabled: Whether progress bar is enabled
        """
        self.config_manager = config_manager
        self.progress_bar_enabled = progress_bar_enabled
        self.logger = get_logger(__name__)

        # 🆕 New architecture: Use AlgorithmFactory for PPO
        # No need to pre-initialize PPO trainer, will be created on-demand

        # Legacy trainers (to be migrated)
        self.base_ml_trainer = BaseMLAlgorithmTrainer(config_manager)
        self.iterative_trainer = IterativeAlgorithmTrainer(config_manager)
        self.ensemble_trainer = EnsembleAlgorithmTrainer(config_manager)
        self.curriculum_trainer = CurriculumAlgorithmTrainer(config_manager)

    def train(self, algorithm: str, unified_config: dict[str, Any]) -> Any:
        """
        Execute training for specified algorithm.

        Args:
            algorithm: Algorithm name
            unified_config: Unified configuration

        Returns:
            Training result
        """
        algorithm = algorithm.lower()

        if algorithm in ["ppo", "sac"]:
            # 🆕 Use AlgorithmFactory for PPO and SAC
            self.logger.info(f"Using AlgorithmFactory for {algorithm.upper()} training")
            return self._train_with_algorithm_factory(algorithm, unified_config)
        elif algorithm == "base_ml":
            return self.base_ml_trainer.train(unified_config)
        elif algorithm == "iterative":
            return self.iterative_trainer.train(unified_config)
        elif algorithm == "ensemble":
            return self.ensemble_trainer.train(unified_config)
        elif algorithm == "curriculum":
            return self.curriculum_trainer.train(unified_config)
        else:
            raise ValueError(f"Unknown algorithm: {algorithm}")

    def _train_with_algorithm_factory(
        self, algorithm: str, unified_config: dict[str, Any]
    ) -> Any:
        """
        Train using AlgorithmFactory (new architecture).

        Args:
            algorithm: Algorithm name ("ppo", "sac", etc.)
            unified_config: Unified configuration

        Returns:
            Training result
        """
        # Create algorithm instance
        algo = AlgorithmFactory.create(algorithm)
        self.logger.info(f"✅ Created algorithm: {algo}")

        # Select appropriate trainer based on algorithm
        if algorithm == "ppo":
            # Use legacy PPOAlgorithmTrainer
            from ztb.training.trainers.ppo_trainer import PPOAlgorithmTrainer

            legacy_trainer = PPOAlgorithmTrainer(
                self.config_manager, self.progress_bar_enabled
            )
            result = legacy_trainer.train(unified_config)
        elif algorithm == "sac":
            # 🆕 Use new SACAlgorithmTrainer
            from ztb.training.trainers.sac_trainer import SACAlgorithmTrainer

            sac_trainer = SACAlgorithmTrainer(
                self.config_manager, self.progress_bar_enabled
            )
            result = sac_trainer.train(unified_config)
        else:
            raise ValueError(
                f"AlgorithmFactory supports {algorithm}, but no trainer implemented yet"
            )

        self.logger.info(
            f"🎉 {algorithm.upper()} training completed via AlgorithmFactory"
        )
        return result
