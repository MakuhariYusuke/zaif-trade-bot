#!/usr/bin/env python3
"""
Parallel Trainer for Unified Trainer - Horizontal scaling support

This module provides parallel training capabilities for multiple algorithms
and configurations simultaneously.
"""

import logging
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from ztb.training.unified_trainer.config import UnifiedTrainerConfig
from ztb.training.unified_trainer.trainer import UnifiedTrainer


@dataclass
class ParallelTrainingResult:
    """Result of parallel training session."""

    algorithm: str
    success: bool
    training_time: float
    final_reward: Optional[float] = None
    model_path: Optional[str] = None
    error_message: Optional[str] = None


class ParallelTrainer:
    """Parallel trainer for running multiple training sessions simultaneously."""

    def __init__(self, configs: List[UnifiedTrainerConfig], max_workers: int = 4):
        """
        Initialize parallel trainer.

        Args:
            configs: List of training configurations
            max_workers: Maximum number of parallel workers
        """
        self.configs = configs
        self.max_workers = max_workers
        self.logger = logging.getLogger(__name__)
        self.results: List[ParallelTrainingResult] = []

    def train_all(self) -> bool:
        """
        Train all configurations in parallel.

        Returns:
            True if all trainings succeeded, False otherwise
        """
        self.logger.info(
            f"Starting parallel training with {len(self.configs)} configurations"
        )

        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit all training tasks
            future_to_config = {
                executor.submit(self._train_single, config): config
                for config in self.configs
            }

            # Collect results as they complete
            for future in as_completed(future_to_config):
                config = future_to_config[future]
                try:
                    result = future.result()
                    self.results.append(result)
                    self.logger.info(
                        f"Training completed for {result.algorithm}: {result.success}"
                    )
                except Exception as e:
                    self.logger.error(
                        f"Training failed for {config.algorithm.value}: {e}"
                    )
                    # Add failed result
                    failed_result = ParallelTrainingResult(
                        algorithm=config.algorithm.value,
                        success=False,
                        training_time=0.0,
                        error_message=str(e),
                    )
                    self.results.append(failed_result)

        # Check if all trainings succeeded
        all_success = all(result.success for result in self.results)

        # Print summary
        self._print_summary()

        return all_success

    def _train_single(self, config: UnifiedTrainerConfig) -> ParallelTrainingResult:
        """Train a single configuration."""
        start_time = time.time()

        try:
            self.logger.info(f"Starting training for {config.algorithm.value}")

            trainer = UnifiedTrainer(config)
            success = trainer.train()

            training_time = time.time() - start_time

            return ParallelTrainingResult(
                algorithm=config.algorithm.value,
                success=success,
                training_time=training_time,
                # Note: Would need to extract these from trainer if available
                final_reward=None,
                model_path=None,
            )

        except Exception as e:
            training_time = time.time() - start_time
            self.logger.error(f"Training failed for {config.algorithm.value}: {e}")

            return ParallelTrainingResult(
                algorithm=config.algorithm.value,
                success=False,
                training_time=training_time,
                error_message=str(e),
            )

    def _print_summary(self) -> None:
        """Print training summary."""
        self.logger.info("=== Parallel Training Summary ===")

        total_time = sum(result.training_time for result in self.results)
        successful_count = sum(1 for result in self.results if result.success)

        self.logger.info(f"Total configurations: {len(self.results)}")
        self.logger.info(f"Successful trainings: {successful_count}")
        self.logger.info(f"Failed trainings: {len(self.results) - successful_count}")
        self.logger.info(f"Total training time: {total_time:.2f} seconds")

        for result in self.results:
            status = "✓" if result.success else "✗"
            self.logger.info(
                f"  {status} {result.algorithm}: {result.training_time:.2f}s"
            )
            if result.error_message:
                self.logger.info(f"    Error: {result.error_message}")


class MultiModelTrainer:
    """Trainer for multiple models of the same algorithm with different configurations."""

    def __init__(
        self, base_config: UnifiedTrainerConfig, variations: List[Dict[str, Any]]
    ):
        """
        Initialize multi-model trainer.

        Args:
            base_config: Base configuration
            variations: List of parameter variations
        """
        self.base_config = base_config
        self.variations = variations
        self.logger = logging.getLogger(__name__)

    def train_variations(self) -> List[ParallelTrainingResult]:
        """
        Train multiple variations of the same algorithm.

        Returns:
            List of training results
        """
        configs = []

        for i, variation in enumerate(self.variations):
            # Create config copy with variations
            config = UnifiedTrainerConfig(
                algorithm=self.base_config.algorithm,
                total_timesteps=self.base_config.total_timesteps,
            )

            # Apply variations
            for key, value in variation.items():
                if hasattr(config, key):
                    setattr(config, key, value)

            configs.append(config)

        # Use parallel trainer
        parallel_trainer = ParallelTrainer(configs)
        parallel_trainer.train_all()

        return parallel_trainer.results
