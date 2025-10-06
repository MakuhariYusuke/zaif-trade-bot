"""
Unit tests for logging_utils.py module.
"""

import logging
from unittest.mock import patch

from ztb.utils.logging_utils import get_logger, setup_logging


class TestSetupLogging:
    """Test cases for setup_logging function."""

    @patch("logging.basicConfig")
    def test_setup_logging_default_parameters(self, mock_basic_config):
        """Test setup_logging with default parameters."""
        setup_logging()

        mock_basic_config.assert_called_once_with(
            level=logging.INFO,
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        )

    @patch("logging.basicConfig")
    def test_setup_logging_custom_level(self, mock_basic_config):
        """Test setup_logging with custom logging level."""
        setup_logging(level=logging.DEBUG)

        mock_basic_config.assert_called_once_with(
            level=logging.DEBUG,
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        )

    @patch("logging.basicConfig")
    def test_setup_logging_custom_format(self, mock_basic_config):
        """Test setup_logging with custom format string."""
        custom_format = "%(levelname)s: %(message)s"
        setup_logging(format_string=custom_format)

        mock_basic_config.assert_called_once_with(
            level=logging.INFO, format=custom_format
        )

    @patch("logging.basicConfig")
    def test_setup_logging_custom_level_and_format(self, mock_basic_config):
        """Test setup_logging with both custom level and format."""
        custom_format = "%(levelname)s: %(message)s"
        setup_logging(level=logging.WARNING, format_string=custom_format)

        mock_basic_config.assert_called_once_with(
            level=logging.WARNING, format=custom_format
        )


class TestGetLogger:
    """Test cases for get_logger function."""

    @patch("logging.getLogger")
    def test_get_logger_basic(self, mock_get_logger):
        """Test get_logger returns logger from logging.getLogger."""
        mock_logger = logging.Logger("test_logger")
        mock_get_logger.return_value = mock_logger

        result = get_logger("test_logger")

        assert result == mock_logger
        mock_get_logger.assert_called_once_with("test_logger")

    @patch("logging.getLogger")
    def test_get_logger_different_names(self, mock_get_logger):
        """Test get_logger with different logger names."""
        test_names = ["module.logger", "another.logger", "root"]

        for name in test_names:
            mock_logger = logging.Logger(name)
            mock_get_logger.return_value = mock_logger

            result = get_logger(name)

            assert result == mock_logger
            mock_get_logger.assert_called_with(name)

    @patch("logging.getLogger")
    def test_get_logger_empty_name(self, mock_get_logger):
        """Test get_logger with empty name."""
        mock_logger = logging.Logger("")
        mock_get_logger.return_value = mock_logger

        result = get_logger("")

        assert result == mock_logger
        mock_get_logger.assert_called_once_with("")

    def test_get_logger_integration(self):
        """Integration test to verify get_logger returns actual Logger instance."""
        logger = get_logger("test.integration")

        assert isinstance(logger, logging.Logger)
        assert logger.name == "test.integration"
