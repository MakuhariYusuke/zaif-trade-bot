"""
Unit tests for unified_trainer.py module.
"""

import tempfile
from pathlib import Path
from unittest.mock import MagicMock, Mock, patch

import pytest

from ztb.training.unified_trainer import (
    UnifiedAlgorithm,
    UnifiedTrainer,
    UnifiedTrainerConfig,
)


class TestUnifiedAlgorithm:
    """Test cases for UnifiedAlgorithm enum."""

    def test_algorithm_values(self):
        """Test that all expected algorithm values are present."""
        expected_values = ["ppo", "base_ml", "iterative", "ensemble", "curriculum"]

        for value in expected_values:
            assert UnifiedAlgorithm(value).value == value

    def test_algorithm_names(self):
        """Test algorithm enum names."""
        assert UnifiedAlgorithm.PPO.name == "PPO"
        assert UnifiedAlgorithm.BASE_ML.name == "BASE_ML"
        assert UnifiedAlgorithm.ITERATIVE.name == "ITERATIVE"
        assert UnifiedAlgorithm.ENSEMBLE.name == "ENSEMBLE"
        assert UnifiedAlgorithm.CURRICULUM.name == "CURRICULUM"


class TestUnifiedTrainerConfig:
    """Test cases for UnifiedTrainerConfig dataclass."""

    def test_default_config(self):
        """Test default configuration values."""
        config = UnifiedTrainerConfig(algorithm=UnifiedAlgorithm.PPO)

        assert config.algorithm == UnifiedAlgorithm.PPO
        assert config.force is False
        assert config.dry_run is False
        assert config.enable_streaming is False
        assert config.stream_batch_size == 256
        assert config.max_features is None
        assert config.offline_mode is False

    def test_custom_config(self):
        """Test custom configuration values."""
        config = UnifiedTrainerConfig(
            algorithm=UnifiedAlgorithm.ENSEMBLE,
            force=True,
            dry_run=True,
            enable_streaming=True,
            stream_batch_size=512,
            max_features=100,
            offline_mode=True,
        )

        assert config.algorithm == UnifiedAlgorithm.ENSEMBLE
        assert config.force is True
        assert config.dry_run is True
        assert config.enable_streaming is True
        assert config.stream_batch_size == 512
        assert config.max_features == 100
        assert config.offline_mode is True


class TestUnifiedTrainer:
    """Test cases for UnifiedTrainer class."""

    @pytest.fixture
    def sample_config(self):
        """Sample configuration for testing."""
        return {
            "algorithm": "ppo",
            "model_name": "test_model",
            "learning_rate": 3e-4,
            "batch_size": 64,
            "n_epochs": 10,
            "gamma": 0.99,
            "gae_lambda": 0.95,
            "clip_range": 0.2,
            "ent_coef": 0.0,
            "vf_coef": 0.5,
            "max_grad_norm": 0.5,
            "verbose": 0,
            "seed": 42,
            "total_timesteps": 100000,
            "eval_freq": 1000,
            "eval_episodes": 5,
            "save_freq": 10000,
            "log_interval": 1,
            "reset_num_timesteps": True,
            "progress_bar": False,
            "offline_mode": False,
        }

    def test_initialization_ppo_algorithm(self, sample_config):
        """Test UnifiedTrainer initialization with PPO algorithm."""
        trainer = UnifiedTrainer(sample_config)

        assert trainer.algorithm == "ppo"
        assert trainer.config_obj.algorithm == UnifiedAlgorithm.PPO
        assert trainer.force is False
        assert trainer.dry_run is False
        assert trainer.enable_streaming is False
        assert trainer.stream_batch_size == 256
        assert trainer.max_features is None

    def test_initialization_invalid_algorithm(self):
        """Test UnifiedTrainer initialization with invalid algorithm."""
        config = {"algorithm": "invalid_algorithm"}

        with pytest.raises(ValueError, match="Unknown algorithm: invalid_algorithm"):
            UnifiedTrainer(config)

    def test_initialization_all_algorithms(self):
        """Test UnifiedTrainer initialization with all valid algorithms."""
        algorithms = ["ppo", "base_ml", "iterative", "ensemble", "curriculum"]

        for algo in algorithms:
            config = {"algorithm": algo}
            trainer = UnifiedTrainer(config)
            assert trainer.algorithm == algo

    @patch('ztb.training.unified_trainer.DiscordNotifier')
    def test_initialization_offline_mode(self, mock_discord, sample_config):
        """Test UnifiedTrainer initialization in offline mode."""
        sample_config["offline_mode"] = True

        trainer = UnifiedTrainer(sample_config)

        # Check that DiscordNotifier was called with None webhook_url
        mock_discord.assert_called_with(webhook_url=None)

    @patch('ztb.training.unified_trainer.safe_operation')
    def test_train_method_calls_safe_operation(self, mock_safe_operation, sample_config):
        """Test that train method calls safe_operation."""
        trainer = UnifiedTrainer(sample_config)
        mock_safe_operation.return_value = "training_result"

        result = trainer.train()

        mock_safe_operation.assert_called_once()
        assert result == "training_result"

    @patch('ztb.training.unified_trainer.safe_operation')
    def test_train_method_with_force_flag(self, mock_safe_operation, sample_config):
        """Test train method with force flag."""
        trainer = UnifiedTrainer(sample_config, force=True)

        assert trainer.force is True
        assert trainer.config_obj.force is True

    @patch('ztb.training.unified_trainer.safe_operation')
    def test_train_method_with_dry_run(self, mock_safe_operation, sample_config):
        """Test train method with dry run flag."""
        trainer = UnifiedTrainer(sample_config, dry_run=True)

        assert trainer.dry_run is True
        assert trainer.config_obj.dry_run is True

    @patch('ztb.training.unified_trainer.safe_operation')
    def test_train_method_with_streaming(self, mock_safe_operation, sample_config):
        """Test train method with streaming enabled."""
        trainer = UnifiedTrainer(sample_config, enable_streaming=True, stream_batch_size=512)

        assert trainer.enable_streaming is True
        assert trainer.stream_batch_size == 512
        assert trainer.config_obj.enable_streaming is True
        assert trainer.config_obj.stream_batch_size == 512

    @patch('ztb.training.unified_trainer.safe_operation')
    def test_train_method_with_max_features(self, mock_safe_operation, sample_config):
        """Test train method with max features limit."""
        trainer = UnifiedTrainer(sample_config, max_features=100)

        assert trainer.max_features == 100
        assert trainer.config_obj.max_features == 100


class TestUnifiedTrainerTrainingMethods:
    """Test cases for training method implementations."""

    @pytest.fixture
    def sample_config(self):
        """Sample configuration for testing."""
        return {
            "algorithm": "ppo",
            "model_name": "test_model",
            "total_timesteps": 1000,
            "offline_mode": True,
        }

    @patch('ztb.training.unified_trainer.safe_operation')
    def test_train_ppo_algorithm(self, mock_safe_operation, sample_config):
        """Test PPO algorithm training."""
        trainer = UnifiedTrainer(sample_config)

        # Mock the _train_ppo method
        with patch.object(trainer, '_train_ppo', return_value="ppo_result") as mock_train_ppo:
            result = trainer._train_impl()

            mock_train_ppo.assert_called_once()
            assert result == "ppo_result"

    @patch('ztb.training.unified_trainer.safe_operation')
    def test_train_base_ml_algorithm(self, mock_safe_operation):
        """Test base ML algorithm training."""
        config = {"algorithm": "base_ml", "offline_mode": True}
        trainer = UnifiedTrainer(config)

        # Mock the _train_base_ml method
        with patch.object(trainer, '_train_base_ml', return_value="base_ml_result") as mock_train_base_ml:
            result = trainer._train_impl()

            mock_train_base_ml.assert_called_once()
            assert result == "base_ml_result"

    @patch('ztb.training.unified_trainer.safe_operation')
    def test_train_iterative_algorithm(self, mock_safe_operation):
        """Test iterative algorithm training."""
        config = {"algorithm": "iterative", "offline_mode": True}
        trainer = UnifiedTrainer(config)

        # Mock the _train_iterative method
        with patch.object(trainer, '_train_iterative', return_value="iterative_result") as mock_train_iterative:
            result = trainer._train_impl()

            mock_train_iterative.assert_called_once()
            assert result == "iterative_result"

    @patch('ztb.training.unified_trainer.safe_operation')
    def test_train_ensemble_algorithm(self, mock_safe_operation):
        """Test ensemble algorithm training."""
        config = {"algorithm": "ensemble", "offline_mode": True}
        trainer = UnifiedTrainer(config)

        # Mock the _train_ensemble method
        with patch.object(trainer, '_train_ensemble', return_value="ensemble_result") as mock_train_ensemble:
            result = trainer._train_impl()

            mock_train_ensemble.assert_called_once()
            assert result == "ensemble_result"

    @patch('ztb.training.unified_trainer.safe_operation')
    def test_train_curriculum_algorithm(self, mock_safe_operation):
        """Test curriculum algorithm training."""
        config = {"algorithm": "curriculum", "offline_mode": True}
        trainer = UnifiedTrainer(config)

        # Mock the _train_curriculum method
        with patch.object(trainer, '_train_curriculum', return_value="curriculum_result") as mock_train_curriculum:
            result = trainer._train_impl()

            mock_train_curriculum.assert_called_once()
            assert result == "curriculum_result"

    @patch('ztb.training.unified_trainer.safe_operation')
    def test_train_unknown_algorithm(self, mock_safe_operation):
        """Test training with unknown algorithm."""
        config = {"algorithm": "unknown", "offline_mode": True}

        # This should fail during initialization, not during training
        with pytest.raises(ValueError, match="Unknown algorithm: unknown"):
            UnifiedTrainer(config)


class TestUnifiedTrainerIntegration:
    """Integration tests for UnifiedTrainer."""

    def test_config_persistence(self):
        """Test that configuration is properly stored and accessible."""
        config = {
            "algorithm": "ppo",
            "model_name": "integration_test",
            "total_timesteps": 50000,
            "learning_rate": 1e-4,
            "batch_size": 128,
            "offline_mode": True,
        }

        trainer = UnifiedTrainer(config, force=True, max_features=50)

        # Check that original config is preserved
        assert trainer.config == config
        assert trainer.algorithm == "ppo"
        assert trainer.force is True
        assert trainer.max_features == 50

        # Check that config object has correct values
        assert trainer.config_obj.algorithm == UnifiedAlgorithm.PPO
        assert trainer.config_obj.force is True
        assert trainer.config_obj.max_features == 50

    def test_algorithm_enum_conversion(self):
        """Test conversion between string and enum algorithms."""
        test_cases = [
            ("ppo", UnifiedAlgorithm.PPO),
            ("base_ml", UnifiedAlgorithm.BASE_ML),
            ("iterative", UnifiedAlgorithm.ITERATIVE),
            ("ensemble", UnifiedAlgorithm.ENSEMBLE),
            ("curriculum", UnifiedAlgorithm.CURRICULUM),
        ]

        for string_val, enum_val in test_cases:
            config = {"algorithm": string_val, "offline_mode": True}
            trainer = UnifiedTrainer(config)

            assert trainer.config_obj.algorithm == enum_val
            assert trainer.algorithm == string_val

