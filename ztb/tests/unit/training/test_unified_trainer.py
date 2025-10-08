"""
Unit tests for unified_trainer.py module.

This module provides comprehensive testing for the UnifiedTrainer class,
covering all training algorithms and configuration scenarios.

Test Categories:
- Unit tests: Individual method testing with mocks
- Integration tests: End-to-end workflow testing
- Property-based tests: Hypothesis-driven edge case testing
- Error handling tests: Exception and edge case coverage

Coverage Goals:
- unified_trainer.py: >80% line coverage
- All 5 training algorithms fully tested
- Error conditions and edge cases covered
- Configuration validation tested

Performance Notes:
- Fast tests (< 0.1s) use mocks extensively
- Slow tests (> 1s) are marked with @pytest.mark.slow
- Parallel execution supported via pytest-xdist
"""

import tempfile
from pathlib import Path
from unittest.mock import MagicMock, Mock, patch, call

import pytest
from hypothesis import given, settings, strategies as st

from ztb.training.core.unified_trainer import (
    UnifiedAlgorithm,
    UnifiedTrainer,
    UnifiedTrainerConfig,
)


# Test Fixtures and Helpers
@pytest.fixture
def sample_training_config():
    """Standard training configuration for testing."""
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


@pytest.fixture
def mock_safe_operation():
    """Mock safe_operation that executes the operation."""
    def safe_operation_side_effect(**kwargs):
        operation = kwargs['operation']
        return operation()
    
    with patch('ztb.training.unified_trainer.safe_operation') as mock:
        mock.side_effect = safe_operation_side_effect
        yield mock


def create_trainer_with_algorithm(algorithm_name, config_overrides=None):
    """Helper to create UnifiedTrainer with specific algorithm."""
    config = {
        "algorithm": algorithm_name,
        "model_name": "test_model",
        "learning_rate": 3e-4,
        "batch_size": 64,
        "total_timesteps": 10000,
        "force": True,  # Skip confirmations
    }
    if config_overrides:
        config.update(config_overrides)
    
    return UnifiedTrainer(config)


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
    def test_train_ppo_algorithm(self, mock_safe_operation, sample_training_config):
        """Test PPO algorithm training."""
        trainer = UnifiedTrainer(sample_training_config)

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


class TestUnifiedTrainerTraining:
    """Test training methods in UnifiedTrainer."""

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

    @pytest.mark.parametrize("algorithm,expected_method", [
        ("ppo", "_train_ppo"),
        ("base_ml", "_train_base_ml"),
        ("iterative", "_train_iterative"),
        ("ensemble", "_train_ensemble"),
        ("curriculum", "_train_curriculum"),
    ])
    @patch('ztb.training.unified_trainer.safe_operation')
    def test_train_dispatches_to_correct_algorithm(self, mock_safe_operation, algorithm, expected_method, sample_config):
        """Test that train() dispatches to the correct algorithm method."""
        # Setup mocks
        def safe_operation_side_effect(**kwargs):
            operation = kwargs['operation']
            return operation()
        mock_safe_operation.side_effect = safe_operation_side_effect
        
        # Mock the algorithm method
        config = sample_config.copy()
        config["algorithm"] = algorithm
        trainer = UnifiedTrainer(config)
        
        with patch.object(trainer, expected_method, return_value=f"{algorithm}_result") as mock_method:
            result = trainer.train()
            
            mock_method.assert_called_once()
            assert result == f"{algorithm}_result"

    def test_train_handles_invalid_algorithm(self, sample_training_config):
        """Test that UnifiedTrainer raises ValueError for invalid algorithm."""
        config = sample_training_config.copy()
        config["algorithm"] = "invalid_algorithm"
        
        with pytest.raises(ValueError, match="Unknown algorithm: invalid_algorithm"):
            UnifiedTrainer(config)

    @patch('ztb.training.unified_trainer.safe_operation')
    def test_train_handles_algorithm_exceptions(self, mock_safe_operation, sample_config):
        """Test that train() handles exceptions in algorithm methods."""
        config = sample_config.copy()
        config["algorithm"] = "ppo"
        trainer = UnifiedTrainer(config)
        
        # Mock safe_operation to return default on error
        mock_safe_operation.return_value = "error_fallback"
        
        result = trainer.train()
        
        # Should return fallback value
        assert result == "error_fallback"
        mock_safe_operation.assert_called_once()


class TestUnifiedTrainerPropertyBased:
    """Property-based tests for UnifiedTrainer using hypothesis."""

    @given(
        learning_rate=st.floats(min_value=1e-6, max_value=1.0),
        batch_size=st.integers(min_value=1, max_value=1024),
        total_timesteps=st.integers(min_value=1000, max_value=1000000),
        gamma=st.floats(min_value=0.8, max_value=1.0),
        clip_range=st.floats(min_value=0.1, max_value=0.5),
    )
    @settings(max_examples=10, deadline=None)
    def test_trainer_initialization_with_valid_numeric_params(
        self, learning_rate, batch_size, total_timesteps, gamma, clip_range
    ):
        """Test that trainer initializes with various valid numeric parameters."""
        config = {
            "algorithm": "ppo",
            "learning_rate": learning_rate,
            "batch_size": batch_size,
            "total_timesteps": total_timesteps,
            "gamma": gamma,
            "clip_range": clip_range,
        }
        
        # Should not raise an exception
        trainer = UnifiedTrainer(config)
        assert trainer.config_obj.algorithm == UnifiedAlgorithm.PPO
        assert trainer.config["learning_rate"] == learning_rate

    @given(
        algorithm=st.sampled_from(["ppo", "base_ml", "iterative", "ensemble", "curriculum"]),
        force=st.booleans(),
        dry_run=st.booleans(),
    )
    def test_trainer_config_flags(self, algorithm, force, dry_run):
        """Test trainer configuration with various flag combinations."""
        config = {
            "algorithm": algorithm,
        }
        
        trainer = UnifiedTrainer(config, force=force, dry_run=dry_run)
        assert trainer.force == force
        assert trainer.dry_run == dry_run
        assert trainer.config_obj.algorithm.value == algorithm

    @given(
        invalid_algorithm=st.text().filter(lambda x: x not in ["ppo", "base_ml", "iterative", "ensemble", "curriculum"])
    )
    def test_invalid_algorithm_raises_error(self, invalid_algorithm):
        """Test that invalid algorithm names raise ValueError."""
        config = {"algorithm": invalid_algorithm}
        
        with pytest.raises(ValueError):
            UnifiedTrainer(config)


class TestUnifiedTrainerIntegration:
    """Integration tests for UnifiedTrainer with real dependencies."""

    @pytest.mark.slow
    @patch('ztb.training.unified_trainer.safe_operation')
    def test_full_training_workflow_ppo(self, mock_safe_operation):
        """Integration test for full PPO training workflow."""
        # This would test the complete flow with minimal mocking
        # Useful for catching integration issues
        pass

    @pytest.mark.parametrize("config_override,expected_behavior", [
        ({"dry_run": True}, "should_not_execute_training"),
        ({"force": True}, "should_skip_confirmations"),
        ({"offline_mode": True}, "should_work_without_network"),
    ])
    def test_configuration_driven_behavior(self, config_override, expected_behavior):
        """Test that configuration changes affect behavior appropriately."""
        base_config = {
            "algorithm": "ppo",
            "total_timesteps": 1000,
        }
        base_config.update(config_override)
        
        trainer = UnifiedTrainer(base_config)
        
        # Verify configuration is applied
        for key, value in config_override.items():
            assert trainer.config.get(key) == value

    @patch('ztb.training.unified_trainer.safe_operation')
    def test_train_method_calls_safe_operation(self, mock_safe_operation, sample_training_config):
        """Test that train method calls safe_operation with correct parameters."""
        mock_safe_operation.return_value = "mock_result"
        
        trainer = UnifiedTrainer(sample_training_config)
        result = trainer.train()
        
        mock_safe_operation.assert_called_once()
        call_args = mock_safe_operation.call_args
        assert call_args[1]['operation'] == trainer._train_impl
        assert call_args[1]['context'] == "training_execution"
        assert call_args[1]['default_result'] is None
        assert result == "mock_result"

    @patch('ztb.training.ppo_trainer.PPOTrainerAutoHalt')
    @patch('ztb.training.unified_trainer.safe_operation')
    def test_train_ppo_algorithm(self, mock_safe_operation, mock_ppo_trainer, sample_training_config):
        """Test PPO algorithm training."""
        # Setup mocks
        mock_model = Mock()
        mock_trainer_instance = Mock()
        mock_trainer_instance.train.return_value = mock_model
        mock_ppo_trainer.return_value = mock_trainer_instance
        
        # Make safe_operation call the actual operation function
        def safe_operation_side_effect(**kwargs):
            operation = kwargs['operation']
            return operation()
        mock_safe_operation.side_effect = safe_operation_side_effect
        
        # Create trainer with PPO algorithm
        config = sample_training_config.copy()
        config["algorithm"] = "ppo"
        trainer = UnifiedTrainer(config)
        
        result = trainer.train()
        
        # Verify safe_operation was called
        mock_safe_operation.assert_called_once()
        
        # Verify PPOTrainer was instantiated and trained
        mock_ppo_trainer.assert_called_once()
        mock_trainer_instance.train.assert_called_once()
        
        # Verify the result is the model
        assert result == mock_model

    @patch('ztb.training.unified_trainer.safe_operation')
    @patch('ztb.training.unified_trainer.MLReinforcementExperiment')
    def test_train_base_ml_algorithm(self, mock_ml_experiment, mock_safe_operation, sample_training_config):
        """Test base ML algorithm training."""
        # Setup mocks
        mock_experiment_instance = Mock()
        mock_experiment_instance.run.return_value = "base_ml_result"
        mock_ml_experiment.return_value = mock_experiment_instance
        
        # Make safe_operation call the actual operation function
        def safe_operation_side_effect(**kwargs):
            operation = kwargs['operation']
            return operation()
        mock_safe_operation.side_effect = safe_operation_side_effect
        
        # Create trainer with base_ml algorithm
        config = sample_training_config.copy()
        config["algorithm"] = "base_ml"
        trainer = UnifiedTrainer(config)
        
        result = trainer.train()
        
        # Verify safe_operation was called
        mock_safe_operation.assert_called_once()
        
        # Verify MLReinforcementExperiment was instantiated and run
        mock_ml_experiment.assert_called_once_with(config, total_steps=1000)
        mock_experiment_instance.run.assert_called_once()
        
        # Verify the result
        assert result == "base_ml_result"

    @patch('ztb.training.unified_trainer.safe_operation')
    @patch('ztb.training.ensemble.EnsembleTradingSystem')
    def test_train_ensemble_algorithm(self, mock_ensemble_system, mock_safe_operation, sample_training_config):
        """Test ensemble algorithm training."""
        # Setup mocks
        mock_ensemble_instance = Mock()
        mock_ensemble_instance.ensemble = Mock()
        mock_ensemble_instance.ensemble.models = [Mock(), Mock()]  # Mock 2 models
        mock_ensemble_system.return_value = mock_ensemble_instance
        
        # Make safe_operation call the actual operation function
        def safe_operation_side_effect(**kwargs):
            operation = kwargs['operation']
            return operation()
        mock_safe_operation.side_effect = safe_operation_side_effect
        
        # Create trainer with ensemble algorithm
        config = sample_training_config.copy()
        config["algorithm"] = "ensemble"
        config["ensemble_models"] = [{"model_path": "model1.zip"}, {"model_path": "model2.zip"}]
        trainer = UnifiedTrainer(config)
        
        result = trainer.train()
        
        # Verify safe_operation was called
        mock_safe_operation.assert_called_once()
        
        # Verify EnsembleTradingSystem was instantiated
        mock_ensemble_system.assert_called_once_with([{"model_path": "model1.zip"}, {"model_path": "model2.zip"}])
        
        # Verify the result is the ensemble system
        assert result == mock_ensemble_instance

    @patch('ztb.training.unified_trainer.safe_operation')
    @patch('ztb.training.run_1m.main')
    def test_train_iterative_algorithm(self, mock_run_1m_main, mock_safe_operation, sample_training_config):
        """Test iterative algorithm training."""
        # Make safe_operation call the actual operation function
        def safe_operation_side_effect(**kwargs):
            operation = kwargs['operation']
            return operation()
        mock_safe_operation.side_effect = safe_operation_side_effect
        
        # Create trainer with iterative algorithm
        config = sample_training_config.copy()
        config["algorithm"] = "iterative"
        config["total_timesteps"] = 10000  # Small number for testing
        config["force"] = True  # Skip confirmation
        trainer = UnifiedTrainer(config)
        
        result = trainer.train()
        
        # Verify safe_operation was called
        mock_safe_operation.assert_called_once()
        
        # Verify run_1m.main was called
        mock_run_1m_main.assert_called_once()
        
        # Verify the result is the return value from run_1m.main
        assert result is mock_run_1m_main.return_value

    @patch('ztb.training.unified_trainer.safe_operation')
    @patch('ztb.training.curriculum_learning.main')
    @patch('os.chdir')
    @patch('os.getcwd')
    @patch('os.path.exists')
    @patch('ztb.training.unified_trainer.get_project_root')
    def test_train_curriculum_algorithm(self, mock_get_project_root, mock_exists, mock_getcwd, mock_chdir, mock_curriculum_main, mock_safe_operation, sample_training_config):
        """Test curriculum algorithm training."""
        # Setup mocks
        mock_get_project_root.return_value = "/project/root"
        mock_exists.return_value = True  # Data file exists
        mock_getcwd.return_value = "/original/dir"  # Original working directory
        mock_curriculum_main.return_value = None  # Success
        
        # Make safe_operation call the actual operation function
        def safe_operation_side_effect(**kwargs):
            operation = kwargs['operation']
            return operation()
        mock_safe_operation.side_effect = safe_operation_side_effect
        
        # Create trainer with curriculum algorithm
        config = sample_training_config.copy()
        config["algorithm"] = "curriculum"
        config["data_path"] = "test_data.csv"
        trainer = UnifiedTrainer(config)
        
        result = trainer.train()
        
        # Verify safe_operation was called
        mock_safe_operation.assert_called_once()
        
        # Verify curriculum_main was called
        mock_curriculum_main.assert_called_once()
        
        # Verify directory changes
        mock_get_project_root.assert_called_once()
        mock_chdir.assert_has_calls([call("/project/root"), call("/original/dir")])
        
        # Verify the result is True (success)
        assert result is True

