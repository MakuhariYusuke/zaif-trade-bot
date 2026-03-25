#!/usr/bin/env python3
"""
Integration tests for Week 9-10: ログ・チェックポイント管理.

Tests the integration of checkpoint management and logging optimization.
"""

import logging
import os
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch
import numpy as np
from stable_baselines3 import SAC

from ztb.trading.environment.components.calculators.reward_calculator import RewardCalculator
from ztb.trading.environment.utils.config import EnvironmentConfig, RewardSettings
from ztb.training.checkpoint.checkpoint_manager import TrainingCheckpointManager, TrainingCheckpointConfig
from ztb.training.unified_trainer.base.callbacks import TrainingProgressCallback
from ztb.utils.logging_utils import StructuredLogger


# Module-level picklable stand-in for SAC model.
# Mock(spec=SAC) cannot be pickled reliably across test ordering.
class _StubOptimizer:
    param_groups = [{"lr": 0.0003}]
    def state_dict(self) -> dict:
        return {"param_groups": self.param_groups}


class _StubPolicy:
    def __init__(self) -> None:
        self.optimizer = _StubOptimizer()
    def state_dict(self) -> dict:
        return {"weights": []}


class _StubModel:
    """Picklable SAC model stub for checkpoint tests."""
    def __init__(self) -> None:
        self.policy = _StubPolicy()


class TestCheckpointLoggingIntegration(unittest.TestCase):
    """Integration tests for checkpoint and logging functionality."""

    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.temp_dir, ignore_errors=True)

        # Setup checkpoint manager
        self.checkpoint_config = TrainingCheckpointConfig(
            interval_steps=100,  # More frequent for testing
            keep_last=3,
            async_save=False,
        )
        self.checkpoint_manager = TrainingCheckpointManager(
            save_dir=str(self.temp_dir),
            config=self.checkpoint_config
        )

        self.mock_model = _StubModel()

        # Setup structured logger
        self.structured_logger = StructuredLogger("test.integration", json_format=True)

    def tearDown(self):
        """Clean up after tests."""
        self.checkpoint_manager.shutdown()

    def test_checkpoint_with_structured_logging(self):
        """Test checkpoint saving with structured logging."""
        step = 100
        metrics = {
            "actor_loss": 0.5,
            "critic_loss": 0.3,
            "reward": 1.2,
            "step": step
        }

        # Log checkpoint operation
        self.structured_logger.info(
            "Starting checkpoint save",
            extra={"step": step, "operation": "save"}
        )

        # Save checkpoint
        self.checkpoint_manager.save(
            step=step,
            model=self.mock_model,
            metrics=metrics,
            extra={"integration_test": True}
        )

        # Log successful save
        self.structured_logger.info(
            "Checkpoint saved successfully",
            extra={"step": step, "checkpoint_path": str(self.temp_dir)}
        )

        # Verify checkpoint was saved
        snapshot = self.checkpoint_manager.load_latest()
        self.assertIsNotNone(snapshot)
        self.assertEqual(snapshot.step, step)
        self.assertEqual(snapshot.payload["metrics"], metrics)

    def test_checkpoint_validation_integration(self):
        """Test checkpoint validation integrated with logging."""
        # Create and save a checkpoint
        step = 200
        self.checkpoint_manager.save(
            step=step,
            model=self.mock_model,
            metrics={"test_metric": 42.0}
        )

        # Load and validate checkpoint
        snapshot = self.checkpoint_manager.load_latest()
        self.assertIsNotNone(snapshot)

        # Log validation start
        self.structured_logger.info(
            "Starting checkpoint validation",
            extra={"step": snapshot.step}
        )

        # Validate checkpoint
        validation = self.checkpoint_manager.validate_checkpoint_integrity(snapshot)

        # Log validation results
        self.structured_logger.info(
            "Checkpoint validation completed",
            extra={
                "valid": validation["valid"],
                "errors_count": len(validation["errors"]),
                "warnings_count": len(validation["warnings"])
            }
        )

        self.assertTrue(validation["valid"])
        self.assertEqual(len(validation["errors"]), 0)

    def test_training_callback_checkpoint_integration(self):
        """Test TrainingProgressCallback checkpoint integration."""
        callback = TrainingProgressCallback(
            check_freq=50,
            checkpoint_manager=self.checkpoint_manager
        )

        # Mock training locals
        callback.locals = {
            "actions": np.array([0.5]),
            "rewards": np.array([1.0]),
            "dones": np.array([False]),
            "infos": [{"portfolio_value": 100000.0, "position": 0.0}],
        }
        callback.model = self.mock_model
        callback.n_calls = 100  # Should trigger checkpoint

        # Execute callback step
        continue_training = callback._on_step()

        self.assertTrue(continue_training)

        # Verify checkpoint was created
        snapshot = self.checkpoint_manager.load_latest()
        self.assertIsNotNone(snapshot)
        self.assertEqual(snapshot.step, 100)

    def test_reward_calculator_dynamic_logging_integration(self):
        """Test RewardCalculator dynamic logging with structured logging."""
        config = EnvironmentConfig.from_dict({
            "curriculum_stage": "forced_balance",
            "logging": {
                "reward_calculator_level": "INFO",
                "dynamic_level_control": True,
                "level_change_threshold": 10,  # More frequent for testing
            }
        })
        reward_settings = RewardSettings.from_dict({
            "behavior_optimization": {
                "action_balance_target": 0.8,
            }
        })

        calculator = RewardCalculator(
            config=config,
            reward_settings=reward_settings,
            initial_portfolio_value=100000.0
        )

        # Test multiple reward calculations to trigger dynamic logging
        observation = np.array([1.0, 2.0, 3.0])

        for step in range(50, 150, 10):  # Test across multiple steps
            reward = calculator.calculate_reward(
                action=1,
                current_price=100.0,
                position=0.0,
                portfolio_value=100000.0,
                atr=1.0,
                transaction_cost=0.0,
                reward_scaling=1.0,
                pnl=0.0,
                old_position=0.0,
                step=step,
                observation=observation,
                reward_history=[0.0] * 10,
                portfolio_value_history=[100000.0] * 10
            )

            self.assertIsInstance(reward, float)

        # Verify dynamic logging was triggered
        self.assertGreater(calculator._log_evaluation_counter, 0)

    def test_end_to_end_checkpoint_workflow(self):
        """Test complete checkpoint save/load workflow with logging."""
        # Step 1: Save multiple checkpoints
        steps = [100, 200, 300, 400]
        for step in steps:
            self.structured_logger.info(
                "Saving checkpoint",
                extra={"step": step, "phase": "save"}
            )

            self.checkpoint_manager.save(
                step=step,
                model=self.mock_model,
                metrics={"step": step, "loss": 1.0 / step},
                extra={"phase": "integration_test"}
            )

        # Step 2: Load latest checkpoint
        latest_snapshot = self.checkpoint_manager.load_latest()
        self.assertIsNotNone(latest_snapshot)
        self.assertEqual(latest_snapshot.step, 400)

        self.structured_logger.info(
            "Loaded latest checkpoint",
            extra={"step": latest_snapshot.step, "phase": "load"}
        )

        # Step 3: Validate checkpoint
        validation = self.checkpoint_manager.validate_checkpoint_integrity(latest_snapshot)

        self.structured_logger.info(
            "Checkpoint validation result",
            extra={
                "valid": validation["valid"],
                "errors": validation["errors"],
                "warnings": validation["warnings"]
            }
        )

        self.assertTrue(validation["valid"])

    @patch("logging.getLogger")
    def test_logging_level_integration(self, mock_get_logger):
        """Test logging level integration across components."""
        mock_logger = Mock()
        mock_get_logger.return_value = mock_logger

        # Test RewardCalculator log level setting
        calculator = RewardCalculator(
            config=EnvironmentConfig.from_dict({"curriculum_stage": "forced_balance"}),
            reward_settings=RewardSettings.from_dict({}),
            initial_portfolio_value=100000.0
        )

        calculator.set_log_level("DEBUG")
        mock_logger.setLevel.assert_called_with(logging.DEBUG)

        calculator.set_log_level("ERROR")
        mock_logger.setLevel.assert_called_with(logging.ERROR)


class TestPerformanceUnderLoad(unittest.TestCase):
    """Performance tests for checkpoint and logging under load."""

    def setUp(self):
        """Set up performance test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.temp_dir, ignore_errors=True)
        self.checkpoint_manager = TrainingCheckpointManager(
            save_dir=self.temp_dir,
            config=TrainingCheckpointConfig(async_save=False)
        )

        self.mock_model = _StubModel()

    def tearDown(self):
        """Clean up after tests."""
        self.checkpoint_manager.shutdown()

    def test_checkpoint_performance_under_load(self):
        """Test checkpoint performance with multiple rapid saves."""
        import time

        start_time = time.time()

        # Save many checkpoints rapidly
        for step in range(100, 600, 100):
            self.checkpoint_manager.save(
                step=step,
                model=self.mock_model,
                metrics={"performance_test": True, "step": step}
            )

        end_time = time.time()
        total_time = end_time - start_time

        # Should complete within reasonable time (adjust threshold as needed)
        self.assertLess(total_time, 10.0, "Checkpoint operations took too long")

        # Verify all checkpoints were saved
        # Note: Due to cleanup, we may not have all checkpoints
        latest = self.checkpoint_manager.load_latest()
        self.assertIsNotNone(latest)

    def test_memory_usage_during_checkpoint_operations(self):
        """Test memory usage during intensive checkpoint operations."""
        import psutil

        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB

        # Perform many checkpoint operations
        for i in range(50):
            self.checkpoint_manager.save(
                step=1000 + i * 10,
                model=self.mock_model,
                metrics={"memory_test": i}
            )

        final_memory = process.memory_info().rss / 1024 / 1024  # MB
        memory_increase = final_memory - initial_memory

        # Memory increase should be reasonable (less than 50MB)
        self.assertLess(memory_increase, 50.0,
                       f"Memory usage increased by {memory_increase:.1f}MB during checkpoint operations")


if __name__ == "__main__":
    unittest.main()
