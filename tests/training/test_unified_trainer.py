"""
Unit tests for UnifiedTrainer component integration.
"""

import pytest

from ztb.training.unified_trainer.trainer import UnifiedTrainer


class TestUnifiedTrainerComponents:
    """Test UnifiedTrainer component integration."""

    @pytest.fixture
    def sample_config(self):
        """Sample configuration for testing."""
        return {
            "training": {
                "algorithm": "ppo",
                "total_timesteps": 1000,
                "learning_rate": 3e-4,
                "batch_size": 64,
            },
            "model_name": "test_model",
            "data_path": "dummy_data.csv",
            "checkpoint_dir": "checkpoints",
            "output_dir": "models",
            "ppo": {
                "learning_rate": 3e-4,
                "n_steps": 2048,
                "batch_size": 64,
                "n_epochs": 10,
                "gamma": 0.99,
                "gae_lambda": 0.95,
                "clip_range": 0.2,
                "ent_coef": 0.0,
                "vf_coef": 0.5,
                "max_grad_norm": 0.5,
            }
        }

    def test_unified_trainer_initialization(self, sample_config):
        """Test UnifiedTrainer initialization with components."""
        trainer = UnifiedTrainer(sample_config)

        # Check that components are initialized
        assert hasattr(trainer, 'config_manager')
        assert hasattr(trainer, 'ui_manager')
        assert hasattr(trainer, 'reporter')
        assert trainer.config_manager is not None
        assert trainer.ui_manager is not None
        assert trainer.reporter is not None

    def test_config_manager_integration(self, sample_config):
        """Test that TrainingConfigManager is properly integrated."""
        trainer = UnifiedTrainer(sample_config)

        # Check that config is processed
        assert hasattr(trainer, 'config')
        assert trainer.config is not None
        assert 'training' in trainer.config
        assert trainer.config['training']['algorithm'] == 'ppo'

    def test_ui_manager_integration(self, sample_config):
        """Test that TrainingUIManager is properly integrated."""
        trainer = UnifiedTrainer(sample_config)

        # Check UI manager has logger
        assert trainer.ui_manager.logger is not None

    def test_reporter_integration(self, sample_config):
        """Test that TrainingReporter is properly integrated."""
        trainer = UnifiedTrainer(sample_config)

        # Check reporter has logger
        assert trainer.reporter.logger is not None

    def test_component_types(self, sample_config):
        """Test that components are of correct types."""
        trainer = UnifiedTrainer(sample_config)

        # Import component classes for type checking
        from ztb.training.unified_trainer.components.config_manager import TrainingConfigManager
        from ztb.training.unified_trainer.components.ui_manager import TrainingUIManager
        from ztb.training.unified_trainer.reporting import TrainingReporter

        assert isinstance(trainer.config_manager, TrainingConfigManager)
        assert isinstance(trainer.ui_manager, TrainingUIManager)
        assert isinstance(trainer.reporter, TrainingReporter)
