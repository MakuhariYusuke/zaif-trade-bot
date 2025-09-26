"""
Unit tests for LoggerManager
"""
import pytest
from unittest.mock import patch
from ztb.utils.logger import LoggerManager


def test_logger_manager_init():
    """Test LoggerManager initialization"""
    logger = LoggerManager()
    assert logger.discord_webhook is None
    assert logger.log_file is None


def test_log_experiment_start():
    """Test log_experiment_start calls logging"""
    logger = LoggerManager()
    with patch.object(logger.logger, 'info') as mock_info:
        logger.log_experiment_start("test_exp", {"param": "value"})
        mock_info.assert_called()


def test_log_experiment_end():
    """Test log_experiment_end calls logging"""
    logger = LoggerManager()
    with patch.object(logger.logger, 'info') as mock_info:
        logger.log_experiment_end({"result": "success"})
        mock_info.assert_called()


def test_log_error():
    """Test log_error calls logging"""
    logger = LoggerManager()
    with patch.object(logger.logger, 'error') as mock_error:
        logger.log_error("test error")
        mock_error.assert_called()