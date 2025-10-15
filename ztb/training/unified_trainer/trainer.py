#!/usr/bin/env python3
"""
Refactored Unified Trainer implementation with enhanced UI and modularity.
"""

import logging
from typing import Any, Dict, Optional

from ztb.training.unified_trainer.algorithms import create_algorithm_trainer
from ztb.training.unified_trainer.config_manager import ConfigManager
from ztb.training.unified_trainer.reporting import TrainingReporter
from ztb.training.unified_trainer.ui import TrainingUI
from ztb.utils.logging_utils import get_logger


class UnifiedTrainer:
    """
    Refactored Unified training interface with enhanced UI and modularity.

    WORK ASSIGNMENT:
    ---------------
    - PPO Algorithm: @trading-team - Standard RL training, evaluation, logging
    - Base ML Algorithm: @ml-research-team - Custom experiments, prototyping
    - Iterative Algorithm: @production-team - Long-running training, monitoring
    """

    def __init__(
        self,
        config: Dict[str, Any],
        force: bool = False,
        dry_run: bool = False,
        enable_streaming: bool = False,
        stream_batch_size: int = 256,
        max_features: Optional[int] = None,
        total_timesteps: Optional[int] = None,
    ):
        """
        Initialize UnifiedTrainer.

        Args:
            config: Training configuration dictionary
            force: Force execution without prompts
            dry_run: Validate without executing
            enable_streaming: Enable streaming data pipeline
            stream_batch_size: Batch size for streaming
            max_features: Maximum number of features
            total_timesteps: Override total_timesteps from config (for quick validation runs)
        """
        # Store configuration
        self.config = config
        self.force = force
        self.dry_run = dry_run
        self.enable_streaming = enable_streaming
        self.stream_batch_size = stream_batch_size
        self.max_features = max_features
        self.total_timesteps = total_timesteps

        # Initialize components
        self.logger = get_logger(__name__)
        self.ui = TrainingUI(self.logger)
        self.config_manager = ConfigManager(self.logger)
        self.reporter = TrainingReporter(self.logger)

        # Algorithm trainer (created during run)
        self.algorithm_trainer = None

        # Training results
        self.training_success = False
        self.training_stats = {}
        self.training_report = {}

    def run(self) -> bool:
        """
        Execute training based on configured algorithm.

        Returns:
            bool: True if training completed successfully
        """
        try:
            # Display header
            algorithm = self.config.get('algorithm', 'unknown')
            config_name = self.config.get('model_name', 'unnamed')
            self.ui.print_header(algorithm, config_name)

            # Display configuration summary
            self.ui.print_config_summary(self.config)

            # Validate configuration
            if not self._validate_configuration():
                return False

            # Handle dry run
            if self.dry_run:
                self.ui.print_info("Dry run mode: validation completed successfully")
                return True

            # Execute training
            return self._execute_training()

        except Exception as e:
            self.ui.print_error(f"Training execution failed: {e}")
            self.logger.error(f"Training execution failed: {e}", exc_info=True)
            return False

    def _validate_configuration(self) -> bool:
        """Validate configuration using enhanced validator."""
        self.logger.info("Validating configuration...")

        # Use the algorithm trainer's validation if available
        algorithm = self.config.get('algorithm', '').lower()

        try:
            # Create algorithm trainer for validation
            trainer = create_algorithm_trainer(algorithm, self.config, self.logger)

            # Validate using trainer
            is_valid = trainer.validate_config()

            if is_valid:
                self.ui.print_success("Configuration validation passed")
                return True
            else:
                self.ui.print_error("Configuration validation failed")
                return False

        except ValueError as e:
            self.ui.print_error(f"Invalid algorithm: {e}")
            return False
        except Exception as e:
            self.ui.print_error(f"Configuration validation error: {e}")
            return False

    def _execute_training(self) -> bool:
        """Execute the actual training."""
        algorithm = self.config.get('algorithm', '').lower()

        try:
            # Override total_timesteps from command line if provided
            if self.total_timesteps is not None:
                self.config['total_timesteps'] = self.total_timesteps
                self.logger.info(f"Overriding total_timesteps from command line: {self.total_timesteps:,}")

            # Create algorithm trainer
            self.logger.info(f"Creating {algorithm.upper()} trainer...")
            self.algorithm_trainer = create_algorithm_trainer(algorithm, self.config, self.logger)

            # Start training UI
            self.ui.start_training()

            # Execute training
            self.logger.info(f"Starting {algorithm.upper()} training...")
            success = self.algorithm_trainer.train()

            # Get training statistics
            if success and hasattr(self.algorithm_trainer, 'get_training_stats'):
                self.training_stats = self.algorithm_trainer.get_training_stats()

            # Display completion
            self.ui.print_training_complete(success, self.training_stats if success else None)

            # Generate and save training report
            if success:
                self.training_report = self.reporter.generate_report(
                    self.config, self.training_stats, success
                )
                report_path = self.reporter.save_report(self.training_report)
                self.reporter.print_summary(self.training_report)

                if report_path:
                    self.ui.print_success(f"Training report saved to: {report_path}")

            self.training_success = success
            return success

        except Exception as e:
            self.ui.print_error(f"Training execution failed: {e}")
            self.logger.error(f"Training execution failed: {e}", exc_info=True)
            return False

    def get_training_stats(self) -> Dict[str, Any]:
        """Get training statistics."""
        return self.training_stats.copy()

    def get_training_report(self) -> Dict[str, Any]:
        """Get complete training report."""
        return self.training_report.copy()

    def is_training_complete(self) -> bool:
        """Check if training completed successfully."""
        return self.training_success