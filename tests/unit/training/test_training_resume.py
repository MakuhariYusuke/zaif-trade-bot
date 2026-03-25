"""
Tests for training resume functionality

Tests the ability to save, load, and resume training state across interruptions.
"""

import importlib
import os
import shutil
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np
import torch
from stable_baselines3 import SAC

from ztb.training.unified_trainer.algorithms.sac_trainer import SACTrainer
from ztb.utils.checkpoint import TrainingStateCheckpointData, TrainingStateManager


class TestTrainingResume(unittest.TestCase):
    """Test cases for training resume functionality"""

    def setUp(self):
        """Set up test fixtures"""
        self.temp_dir = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.temp_dir, ignore_errors=True)
        self.training_state_manager = TrainingStateManager(str(self.temp_dir))

        self.mock_torch = Mock()
        self.mock_torch.cuda.is_available.return_value = False
        self.mock_torch.random.get_rng_state.return_value = b"cpu_rng_state"
        self.mock_torch.random.set_rng_state = Mock()
        real_import_module = importlib.import_module
        self.import_module_patcher = patch(
            "importlib.import_module",
            side_effect=lambda name, package=None: self.mock_torch
            if name == "torch"
            else real_import_module(name, package),
        )
        self.import_module_patcher.start()

        # Create a mock SAC model
        self.mock_env = Mock()
        self.mock_model = Mock(spec=SAC)
        self.mock_model.policy = Mock()
        self.mock_model.policy.state_dict.return_value = {
            "weight": np.zeros((10, 10), dtype=np.float32)
        }
        self.mock_model.policy.optimizer = Mock()
        self.mock_model.policy.optimizer.state_dict.return_value = {
            "state": {},
            "param_groups": [],
        }
        self.mock_model.policy_kwargs = {}
        self.mock_model.replay_buffer = SimpleNamespace(size=1000, pos=500)

    def tearDown(self):
        """Clean up test fixtures"""
        self.import_module_patcher.stop()

    def test_save_training_state(self):
        """Test saving training state"""
        # Save training state
        filepath = self.training_state_manager.save_training_state(
            model=self.mock_model,
            total_timesteps=1000,
            episode_count=10,
            episode_rewards=[1.0, 2.0, 3.0],
            episode_lengths=[100, 200, 300],
            config={"test": "config"},
            training_time=60.0,
        )

        # Verify file was created
        self.assertTrue(os.path.exists(filepath))

        # Load and verify content
        loaded_state = self.training_state_manager.load_training_state(filepath)

        self.assertEqual(loaded_state["total_timesteps"], 1000)
        self.assertEqual(loaded_state["episode_count"], 10)
        self.assertEqual(loaded_state["episode_rewards"], [1.0, 2.0, 3.0])
        self.assertEqual(loaded_state["episode_lengths"], [100, 200, 300])
        self.assertEqual(loaded_state["config"], {"test": "config"})
        self.assertEqual(loaded_state["training_time"], 60.0)
        self.assertEqual(loaded_state["version"], "1.0")

    def test_restore_training_state(self):
        """Test restoring training state to model"""
        # Create a fresh model to restore into
        fresh_model = Mock(spec=SAC)
        fresh_model.policy = Mock()
        fresh_model.policy.load_state_dict = Mock()
        fresh_model.policy.optimizer = Mock()
        fresh_model.policy.optimizer.load_state_dict = Mock()
        fresh_model.replay_buffer = Mock()
        fresh_model.replay_buffer.__dict__ = {}

        # Save state
        filepath = self.training_state_manager.save_training_state(
            model=self.mock_model, total_timesteps=1000, config={"test": "config"}
        )

        # Load state
        training_state = self.training_state_manager.load_training_state(filepath)

        # Restore to fresh model
        self.training_state_manager.restore_training_state(fresh_model, training_state)

        # Verify restoration calls were made
        fresh_model.policy.load_state_dict.assert_called_once()
        fresh_model.policy.optimizer.load_state_dict.assert_called_once()

    def test_resume_compatibility_validation(self):
        """Test validation of resume compatibility"""
        # Create training state with specific config
        saved_config = {
            "training": {
                "sac_hyperparameters": {
                    "learning_rate": 0.001,
                    "batch_size": 256,
                    "buffer_size": 1000000,
                },
                "environment_config": {
                    "window_size": 100,
                    "fee": 0.001,
                    "leverage": 1.0,
                },
            }
        }

        training_state: TrainingStateCheckpointData = {
            "model_state": {},
            "optimizer_state": {},
            "replay_buffer_state": None,
            "total_timesteps": 1000,
            "episode_count": 0,
            "episode_rewards": [],
            "episode_lengths": [],
            "random_state": (None, None, None),
            "config": saved_config,
            "timestamp": 0.0,
            "training_time": 0.0,
            "version": "1.0",
        }

        # Test compatible config
        compatible_config = saved_config.copy()
        validation = self.training_state_manager.validate_resume_compatibility(
            training_state, compatible_config
        )
        self.assertTrue(validation["compatible"])
        self.assertEqual(len(validation["errors"]), 0)

        # Test incompatible config
        incompatible_config = {
            "training": {
                "sac_hyperparameters": {
                    "learning_rate": 0.001,
                    "batch_size": 256,
                    "buffer_size": 1000000,
                },
                "environment_config": {
                    "window_size": 200,
                    "fee": 0.001,
                    "leverage": 1.0,
                },  # Changed from 100
            }
        }

        validation = self.training_state_manager.validate_resume_compatibility(
            training_state, incompatible_config
        )
        self.assertFalse(validation["compatible"])
        self.assertGreater(len(validation["errors"]), 0)

    def test_list_training_states(self):
        """Test listing available training states"""
        # Save multiple training states
        filepath1 = self.training_state_manager.save_training_state(
            model=self.mock_model, total_timesteps=1000, config={"algorithm": "SAC"}
        )

        filepath2 = self.training_state_manager.save_training_state(
            model=self.mock_model, total_timesteps=2000, config={"algorithm": "SAC"}
        )

        # List states
        states = self.training_state_manager.list_training_states()

        self.assertEqual(len(states), 2)
        # Should be sorted by timestamp (newest first)
        self.assertGreaterEqual(states[0]["timestamp"], states[1]["timestamp"])

    @patch("ztb.training.unified_trainer.algorithms.sac_trainer.HeavyTradingEnv")
    @patch("pandas.read_csv")
    def test_sac_trainer_resume_integration(self, mock_read_csv, mock_env_class):
        """Test SAC trainer resume integration"""
        # Mock dependencies
        mock_df = Mock()
        mock_read_csv.return_value = mock_df

        mock_env = Mock()
        mock_env_class.return_value = mock_env

        # Create config with resume path
        config = {
            "training": {
                "total_timesteps": 2000,
                "resume_from": "nonexistent_path.pkl",  # This should fail gracefully
                "data_config": {"data_path": "test.csv"},
                "sac_hyperparameters": {},
                "checkpoint_dir": str(self.temp_dir),
            },
            "model_name": "test_model",
        }

        trainer = SACTrainer(config)

        # Test with non-existent resume path (should start fresh training)
        # This is a simplified test - full integration test would require more mocking
        self.assertIsNotNone(trainer.training_state_manager)


class TestUnifiedResumeManager(unittest.TestCase):
    """Test cases for unified resume manager"""

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.temp_dir, ignore_errors=True)
        self.resume_manager = Mock()
        # In real implementation, this would be:
        # from ztb.training.unified_resume import UnifiedResumeManager
        # self.resume_manager = UnifiedResumeManager(self.temp_dir)

    def test_resume_options_creation(self):
        """Test creation of resume options"""
        from ztb.training.unified_resume import create_resume_options

        options = create_resume_options(
            training_state_path="/path/to/state.pkl",
            additional_timesteps=1000,
            validate_compatibility=True,
        )

        self.assertEqual(options.training_state_path, "/path/to/state.pkl")
        self.assertEqual(options.additional_timesteps, 1000)
        self.assertTrue(options.validate_compatibility)


if __name__ == "__main__":
    unittest.main()
