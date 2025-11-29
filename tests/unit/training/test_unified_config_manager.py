#!/usr/bin/env python3
"""
Tests for UnifiedTrainingConfigManager.

Tests configuration management functionality.
"""

from unittest.mock import Mock, patch

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
        """Test UnifiedTrainingConfigManager initialization with config directory."""
        config_dir = "/tmp/test_config"
        config_manager = UnifiedTrainingConfigManager(config_dir)
        assert config_manager is not None

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

        # Mock config object
        config = Mock()
        config.algorithm = "sac"
        config.total_timesteps = 100000
        config.learning_rate = 0.001

        result = config_manager.process_config(config)

        assert isinstance(result, dict)
        assert "algorithm" in result
        assert result["algorithm"] == "sac"

    def test_process_config_with_defaults(self):
        """Test config processing with default values."""
        config_manager = UnifiedTrainingConfigManager()

        # Config with minimal settings
        config = Mock()
        config.algorithm = "ppo"
        # Missing other required fields

        result = config_manager.process_config(config)

        assert result["algorithm"] == "ppo"
        # Should have default values for missing fields
        assert "total_timesteps" in result
        assert "learning_rate" in result

    def test_inheritance_from_config_manager(self):
        """Test that UnifiedTrainingConfigManager inherits from ConfigManager."""
        from ztb.utils.config_manager import ConfigManager

        config_manager = UnifiedTrainingConfigManager()
        assert isinstance(config_manager, ConfigManager)
