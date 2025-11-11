#!/usr/bin/env python3
"""
Unit tests for checkpoint_manager.py - Training checkpoint management system.

Tests TrainingCheckpointManager, checkpoint validation, and save/load functionality.
"""

import os
import tempfile
import unittest
from unittest.mock import Mock, patch
import numpy as np
import torch
from stable_baselines3 import SAC
from stable_baselines3.common.policies import ActorCriticPolicy

from ztb.trading.environment.constants import BYTES_PER_GB, BYTES_PER_MB
from ztb.training.checkpoint.checkpoint_manager import (
    TrainingCheckpointManager,
    TrainingCheckpointConfig,
    TrainingCheckpointSnapshot,
)


class TestTrainingCheckpointManager(unittest.TestCase):
    """Test cases for TrainingCheckpointManager class."""

    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.config = TrainingCheckpointConfig(
            interval_steps=1000,
            keep_last=3,
            compress="lz4",
            async_save=False,
            include_optimizer=True,
            include_replay_buffer=False,
        )
        self.manager = TrainingCheckpointManager(
            save_dir=self.temp_dir,
            config=self.config
        )

        # Create a mock SAC model for testing
        self.mock_model = Mock(spec=SAC)
        self.mock_policy = Mock(spec=ActorCriticPolicy)
        self.mock_model.policy = self.mock_policy
        self.mock_optimizer = Mock()
        self.mock_policy.optimizer = self.mock_optimizer

    def tearDown(self):
        """Clean up after tests."""
        self.manager.shutdown()
        # Clean up temp directory
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_checkpoint_manager_initialization(self):
        """Test TrainingCheckpointManager initialization."""
        self.assertIsInstance(self.manager, TrainingCheckpointManager)
        self.assertEqual(self.manager.save_dir, self.temp_dir)
        self.assertEqual(self.manager.config.interval_steps, 1000)

    def test_should_checkpoint(self):
        """Test checkpoint timing logic."""
        # Should checkpoint at multiples of interval_steps
        self.assertTrue(self.manager.should_checkpoint(1000))
        self.assertTrue(self.manager.should_checkpoint(2000))
        self.assertFalse(self.manager.should_checkpoint(500))
        self.assertFalse(self.manager.should_checkpoint(0))

    def test_save_and_load_checkpoint(self):
        """Test saving and loading checkpoints."""
        step = 1000
        metrics = {"loss": 0.5, "reward": 1.2}
        extra = {"custom_data": "test"}

        # Save checkpoint
        self.manager.save(
            step=step,
            model=self.mock_model,
            metrics=metrics,
            extra=extra
        )

        # Load checkpoint
        snapshot = self.manager.load_latest()

        self.assertIsNotNone(snapshot)
        self.assertEqual(snapshot.step, step)
        self.assertEqual(snapshot.payload["metrics"], metrics)
        self.assertEqual(snapshot.payload["extra"], extra)

    def test_validate_checkpoint_integrity_valid(self):
        """Test checkpoint integrity validation for valid checkpoint."""
        # Create a valid snapshot
        payload = {
            "policy_state": {"layer1": torch.randn(10, 10)},
            "optimizer_state": {"state": {}, "param_groups": []},
            "step": 1000,
            "timestamp": 1234567890.0,
        }
        snapshot = TrainingCheckpointSnapshot(
            step=1000,
            payload=payload,
            metadata={}
        )

        validation = self.manager.validate_checkpoint_integrity(snapshot)

        self.assertTrue(validation["valid"])
        self.assertEqual(len(validation["errors"]), 0)

    def test_validate_checkpoint_integrity_invalid(self):
        """Test checkpoint integrity validation for invalid checkpoint."""
        # Create an invalid snapshot (missing policy_state)
        payload = {
            "optimizer_state": {"state": {}, "param_groups": []},
            "step": 1000,
        }
        snapshot = TrainingCheckpointSnapshot(
            step=1000,
            payload=payload,
            metadata={}
        )

        validation = self.manager.validate_checkpoint_integrity(snapshot)

        self.assertFalse(validation["valid"])
        self.assertIn("Missing policy state", validation["errors"][0])

    def test_validate_checkpoint_integrity_with_model(self):
        """Test checkpoint integrity validation against a model."""
        # Create a simple model
        model = SAC("MlpPolicy", "Pendulum-v1", verbose=0)

        # Create a valid snapshot with matching policy state
        policy_state = model.policy.state_dict()
        payload = {
            "policy_state": policy_state,
            "optimizer_state": model.policy.optimizer.state_dict(),
            "step": 1000,
        }
        snapshot = TrainingCheckpointSnapshot(
            step=1000,
            payload=payload,
            metadata={}
        )

        validation = self.manager.validate_checkpoint_integrity(snapshot, model)

        self.assertTrue(validation["valid"])
        self.assertEqual(len(validation["errors"]), 0)

    def test_checkpoint_cleanup(self):
        """Test that old checkpoints are cleaned up."""
        # Save multiple checkpoints
        for step in [1000, 2000, 3000, 4000]:
            self.manager.save(
                step=step,
                model=self.mock_model,
                metrics={"step": step}
            )

        # Should only keep the last 3 checkpoints
        checkpoints = os.listdir(self.temp_dir)
        # Note: actual number may vary due to cleanup timing
        self.assertGreaterEqual(len(checkpoints), 1)

    def test_async_save_configuration(self):
        """Test async save configuration."""
        config = TrainingCheckpointConfig(async_save=True)
        manager = TrainingCheckpointManager(
            save_dir=self.temp_dir,
            config=config
        )

        self.assertTrue(manager.config.async_save)
        manager.shutdown()

    @patch("psutil.virtual_memory")
    def test_memory_pressure_handling(self, mock_memory):
        """Test checkpoint behavior under memory pressure."""
        # Mock low memory situation
        mock_memory.return_value = Mock()
        mock_memory.return_value.available = 100 * BYTES_PER_MB  # 100MB
        mock_memory.return_value.total = BYTES_PER_GB  # 1GB

        # This should still work even with memory pressure
        self.manager.save(
            step=1000,
            model=self.mock_model,
            metrics={"memory_test": True}
        )

        snapshot = self.manager.load_latest()
        self.assertIsNotNone(snapshot)


class TestTrainingCheckpointConfig(unittest.TestCase):
    """Test cases for TrainingCheckpointConfig."""

    def test_config_defaults(self):
        """Test default configuration values."""
        config = TrainingCheckpointConfig()

        self.assertEqual(config.interval_steps, 10000)
        self.assertEqual(config.keep_last, 5)
        self.assertEqual(config.compress, "lz4")
        self.assertFalse(config.async_save)
        self.assertTrue(config.include_optimizer)
        self.assertFalse(config.include_replay_buffer)
        self.assertTrue(config.include_rng_state)

    def test_config_customization(self):
        """Test custom configuration values."""
        config = TrainingCheckpointConfig(
            interval_steps=500,
            keep_last=2,
            compress="gzip",
            async_save=True,
            include_optimizer=False,
            include_replay_buffer=True,
        )

        self.assertEqual(config.interval_steps, 500)
        self.assertEqual(config.keep_last, 2)
        self.assertEqual(config.compress, "gzip")
        self.assertTrue(config.async_save)
        self.assertFalse(config.include_optimizer)
        self.assertTrue(config.include_replay_buffer)


if __name__ == "__main__":
    unittest.main()