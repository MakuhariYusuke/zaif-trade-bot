#!/usr/bin/env python3
"""
Tests for UnifiedTrainingConfigManager.

Tests configuration management functionality.
"""

from unittest.mock import Mock, patch

import pytest

from ztb.training.unified_trainer.components.config_manager import (
    UnifiedTrainingConfigManager,
)

# Import ConfigManager directly to avoid unified_trainer import issues


class TestUnifiedTrainingConfigManager:
    """Tests for UnifiedTrainingConfigManager."""

    def test_initialization(self):
        """Test UnifiedTrainingConfigManager initialization."""
        config_manager = UnifiedTrainingConfigManager()
        assert config_manager is not None
        assert hasattr(config_manager, "logger")

    def test_initialization_with_config_dir(self):
        """Current manager does not accept config_dir as a positional argument."""
        config_dir = "/tmp/test_config"
        with pytest.raises(TypeError):
            UnifiedTrainingConfigManager(config_dir)

    @patch("ztb.training.unified_trainer.components.config_manager.get_logger")
    def test_logger_initialization(self, mock_get_logger):
        """Test logger initialization."""
        mock_logger = Mock()
        mock_get_logger.return_value = mock_logger

        config_manager = UnifiedTrainingConfigManager()
        assert config_manager.logger == mock_logger
        mock_get_logger.assert_called()

    def test_process_config_basic(self):
        """Test basic config processing."""
        config_manager = UnifiedTrainingConfigManager()

        config = {
            "algorithm": "sac",
            "total_timesteps": 100000,
            "learning_rate": 0.001,
        }

        result = config_manager.process_config(config)

        assert isinstance(result, dict)
        assert "training" in result
        assert result["training"]["algorithm"] == "sac"

    def test_process_config_with_defaults(self):
        """Flat config dict should be wrapped under training."""
        config_manager = UnifiedTrainingConfigManager()

        config = {
            "algorithm": "ppo",
            "total_timesteps": 1000,
        }

        result = config_manager.process_config(config)

        assert result["training"]["algorithm"] == "ppo"
        assert result["training"]["total_timesteps"] == 1000

    def test_inheritance_from_config_manager(self):
        """Unified alias should expose the TrainingConfigManager interface."""
        config_manager = UnifiedTrainingConfigManager()
        assert hasattr(config_manager, "process_config")
        assert hasattr(config_manager, "get_algorithm_config")
