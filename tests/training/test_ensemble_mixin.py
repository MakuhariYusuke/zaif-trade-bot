#!/usr/bin/env python3
"""
Unit tests for EnsembleMixin functionality.

Tests cover:
- EnsembleMixin initialization and configuration
- Ensemble prediction capabilities
- Report generation functionality
- Integration with different trainers
"""

from unittest.mock import Mock, patch

import numpy as np
import pytest

from ztb.training.core.config_manager import ConfigManager
from ztb.training.unified_trainer.ensemble_mixin import EnsembleMixin


class MockTrainer(EnsembleMixin):
    """Mock trainer class for testing EnsembleMixin."""

    def __init__(self):
        super().__init__()
        self.logger = Mock()


class TestEnsembleMixin:
    """Test EnsembleMixin functionality."""

    @pytest.fixture
    def mock_trainer(self):
        """Create mock trainer with EnsembleMixin."""
        return MockTrainer()

    @pytest.fixture
    def ensemble_config(self):
        """Sample ensemble configuration."""
        return {
            "enabled": True,
            "num_members": 3,
            "voting_method": "majority",
            "specialization_enabled": True,
            "adaptation_enabled": True,
            "confidence_threshold": 0.6,
            "stability_weight": 0.3,
        }

    @pytest.fixture
    def training_config(self, ensemble_config):
        """Sample training configuration with ensemble enabled."""
        return {
            "model_name": "test_model",
            "algorithm": "ppo",
            "ensemble": ensemble_config,
            "total_timesteps": 1000,
        }

    def test_initialization_without_ensemble(self, mock_trainer):
        """Test initialization when ensemble is disabled."""
        config = {"ensemble": {"enabled": False}}

        mock_trainer.initialize_ensemble(config)

        assert not mock_trainer.ensemble_enabled
        assert mock_trainer.ensemble_system is None
        assert mock_trainer.ensemble_config is None

    @patch("ztb.training.unified_trainer.ensemble_mixin.EnsemblePredictor")
    @patch("ztb.training.unified_trainer.ensemble_mixin.EnsembleConfig")
    def test_initialization_with_ensemble(
        self,
        mock_ensemble_config_cls,
        mock_ensemble_predictor_cls,
        mock_trainer,
        training_config,
    ):
        """Test successful ensemble initialization."""
        # Setup mocks
        mock_config = Mock()
        mock_ensemble_config_cls.return_value = mock_config

        mock_predictor = Mock()
        mock_ensemble_predictor_cls.return_value = mock_predictor

        # Initialize ensemble
        mock_trainer.initialize_ensemble(training_config)

        # Verify ensemble is enabled and configured
        assert mock_trainer.ensemble_enabled
        assert mock_trainer.ensemble_system == mock_predictor
        assert mock_trainer.ensemble_config == mock_config

        # Verify classes were called correctly
        mock_ensemble_config_cls.assert_called_once_with(
            num_members=3,
            voting_method="majority",
            specialization_enabled=True,
            adaptation_enabled=True,
            confidence_threshold=0.6,
            stability_weight=0.3,
        )
        mock_ensemble_predictor_cls.assert_called_once_with(mock_config)

    @patch("ztb.training.unified_trainer.ensemble_mixin.EnsemblePredictor")
    def test_predict_with_ensemble_disabled(
        self, mock_ensemble_predictor_cls, mock_trainer
    ):
        """Test prediction when ensemble is disabled."""
        mock_trainer.ensemble_enabled = False

        obs = np.array([1, 2, 3])
        result = mock_trainer.predict_with_ensemble(obs)

        assert result is None
        mock_ensemble_predictor_cls.assert_not_called()

    @patch("ztb.training.unified_trainer.ensemble_mixin.EnsemblePredictor")
    def test_predict_with_ensemble_enabled(
        self, mock_ensemble_predictor_cls, mock_trainer
    ):
        """Test prediction when ensemble is enabled."""
        mock_trainer.ensemble_enabled = True
        mock_predictor = Mock()
        mock_predictor.predict.return_value = {"action": 1, "confidence": 0.8}
        mock_trainer.ensemble_system = mock_predictor

        obs = np.array([1, 2, 3])
        result = mock_trainer.predict_with_ensemble(obs)

        assert result == {"action": 1, "confidence": 0.8}
        mock_predictor.predict.assert_called_once_with(obs)

    @patch("ztb.training.unified_trainer.ensemble_mixin.EnsemblePredictor")
    def test_predict_with_ensemble_error(
        self, mock_ensemble_predictor_cls, mock_trainer
    ):
        """Test prediction error handling."""
        mock_trainer.ensemble_enabled = True
        mock_predictor = Mock()
        mock_predictor.predict.side_effect = Exception("Prediction failed")
        mock_trainer.ensemble_system = mock_predictor

        obs = np.array([1, 2, 3])
        result = mock_trainer.predict_with_ensemble(obs)

        assert result is None
        mock_trainer.logger.error.assert_called()

    @patch("ztb.training.unified_trainer.reporting.TrainingReporter")
    @patch("ztb.training.unified_trainer.ui.TrainingUI")
    def test_generate_report_with_ensemble_disabled(
        self, mock_ui_cls, mock_reporter_cls, mock_trainer
    ):
        """Test report generation when ensemble is disabled."""
        mock_trainer.ensemble_enabled = False

        result = mock_trainer.generate_ensemble_report(Mock(), Mock())

        assert result is None
        mock_reporter_cls.assert_not_called()
        mock_ui_cls.assert_not_called()

    @patch("ztb.training.unified_trainer.reporting.TrainingReporter")
    @patch("ztb.training.unified_trainer.ui.TrainingUI")
    def test_generate_report_with_ensemble_enabled(
        self, mock_ui_cls, mock_reporter_cls, mock_trainer, training_config
    ):
        """Test report generation when ensemble is enabled."""
        mock_trainer.ensemble_enabled = True
        mock_trainer.ensemble_config = Mock()
        mock_trainer.ensemble_system = Mock()

        mock_reporter = Mock()
        mock_reporter_cls.return_value = mock_reporter
        mock_reporter.generate_ensemble_report.return_value = "/path/to/report.md"

        mock_ui = Mock()
        mock_ui_cls.return_value = mock_ui

        with patch("os.path.exists", return_value=True):
            result = mock_trainer.generate_ensemble_report(mock_reporter, mock_ui)

        assert result == "/path/to/report.md"
        mock_reporter.generate_ensemble_report.assert_called_once()

    def test_print_ensemble_status_disabled(self, mock_trainer):
        """Test status printing when ensemble is disabled."""
        mock_trainer.ensemble_enabled = False

        ui = Mock()
        mock_trainer.print_ensemble_status(ui)

        ui.print_ensemble_status.assert_not_called()

    def test_print_ensemble_status_enabled(self, mock_trainer):
        """Test status printing when ensemble is enabled."""
        mock_trainer.ensemble_enabled = True
        mock_trainer.ensemble_config = Mock()
        mock_trainer.ensemble_system = Mock()

        ui = Mock()
        mock_trainer.print_ensemble_status(ui)

        ui.print_ensemble_status.assert_called_once()


class TestEnsembleIntegration:
    """Test ensemble integration with actual trainers."""

    @patch("ztb.training.unified_trainer.ensemble_mixin.EnsemblePredictor")
    def test_sac_trainer_integration(self, mock_predictor_cls):
        """Test ensemble integration with SAC trainer."""
        from ztb.training.trainers.sac_trainer import SACAlgorithmTrainer

        config_manager = Mock(spec=ConfigManager)
        trainer = SACAlgorithmTrainer(config_manager)

        # Verify it inherits from EnsembleMixin
        assert isinstance(trainer, EnsembleMixin)

        # Test ensemble initialization
        config = {"ensemble": {"enabled": True, "num_members": 2}}
        trainer.initialize_ensemble(config)

        assert trainer.ensemble_enabled

    @patch("ztb.training.unified_trainer.ensemble_mixin.EnsemblePredictor")
    def test_ppo_trainer_integration(self, mock_predictor_cls):
        """Test ensemble integration with PPO trainer."""
        from ztb.training.trainers.ppo_trainer import PPOAlgorithmTrainer

        config_manager = Mock(spec=ConfigManager)
        trainer = PPOAlgorithmTrainer(config_manager)

        # Verify it inherits from EnsembleMixin
        assert isinstance(trainer, EnsembleMixin)

        # Test ensemble initialization
        config = {"ensemble": {"enabled": True, "num_members": 2}}
        trainer.initialize_ensemble(config)

        assert trainer.ensemble_enabled
