"""
Unit tests for logging_utils.py module.
Tests for RotatingFileHandler and logging setup.
"""

import logging
import logging.handlers
import tempfile
from pathlib import Path

from ztb.utils.logging_utils import get_logger, setup_logging


class TestSetupLogging:
    """Test cases for setup_logging function with RotatingFileHandler."""

    def test_setup_logging_console_only(self):
        """Test setup_logging with console handler only (no file)."""
        # Clear existing handlers
        root_logger = logging.getLogger()
        root_logger.handlers.clear()

        setup_logging(level=logging.INFO)

        # Should have exactly 1 handler (console)
        assert len(root_logger.handlers) == 1
        assert isinstance(root_logger.handlers[0], logging.StreamHandler)
        assert root_logger.level == logging.INFO

    def test_setup_logging_with_file_rotation(self):
        """Test setup_logging with RotatingFileHandler."""
        with tempfile.TemporaryDirectory() as tmpdir:
            log_file = Path(tmpdir) / "test.log"

            # Clear existing handlers
            root_logger = logging.getLogger()
            root_logger.handlers.clear()

            setup_logging(
                level=logging.DEBUG,
                log_file=str(log_file),
                max_bytes=1024,
                backup_count=3,
            )

            # Should have 2 handlers (console + file)
            assert len(root_logger.handlers) == 2

            # Find the RotatingFileHandler
            file_handler = None
            for handler in root_logger.handlers:
                if isinstance(handler, logging.handlers.RotatingFileHandler):
                    file_handler = handler
                    break

            assert file_handler is not None, "RotatingFileHandler not found"
            assert file_handler.maxBytes == 1024
            assert file_handler.backupCount == 3
            assert log_file.exists()

            # Close handlers to release file locks (Windows)
            for handler in root_logger.handlers[:]:
                handler.close()
                root_logger.removeHandler(handler)

    def test_setup_logging_custom_format(self):
        """Test setup_logging with custom format string."""
        root_logger = logging.getLogger()
        root_logger.handlers.clear()

        custom_format = "%(levelname)s: %(message)s"
        setup_logging(format_string=custom_format)

        # Check formatter
        handler = root_logger.handlers[0]
        assert handler.formatter._fmt == custom_format

    def test_setup_logging_creates_log_directory(self):
        """Test that setup_logging creates log directory if missing."""
        with tempfile.TemporaryDirectory() as tmpdir:
            log_file = Path(tmpdir) / "nested" / "dir" / "test.log"
            assert not log_file.parent.exists()

            root_logger = logging.getLogger()
            root_logger.handlers.clear()

            setup_logging(log_file=str(log_file))

            assert log_file.parent.exists()
            assert log_file.exists()

            # Close handlers to release file locks (Windows)
            for handler in root_logger.handlers[:]:
                handler.close()
                root_logger.removeHandler(handler)

    def test_setup_logging_clears_existing_handlers(self):
        """Test that setup_logging clears existing handlers."""
        root_logger = logging.getLogger()

        # Add dummy handler
        dummy_handler = logging.StreamHandler()
        root_logger.addHandler(dummy_handler)
        assert len(root_logger.handlers) > 0

        # Setup logging should clear and recreate
        setup_logging()

        # Old handler should be removed
        assert dummy_handler not in root_logger.handlers


class TestGetLogger:
    """Test cases for get_logger function."""

    def test_get_logger_returns_logger_instance(self):
        """Test get_logger returns Logger instance."""
        logger = get_logger("test.module")

        assert isinstance(logger, logging.Logger)
        assert logger.name == "test.module"

    def test_get_logger_different_names(self):
        """Test get_logger with different logger names."""
        test_names = ["module.logger", "another.logger", "root"]

        for name in test_names:
            logger = get_logger(name)

            assert isinstance(logger, logging.Logger)
            assert logger.name == name

    def test_get_logger_empty_name(self):
        """Test get_logger with empty name (returns root logger)."""
        logger = get_logger("")

        assert isinstance(logger, logging.Logger)
        # Empty name returns root logger
        assert logger.name == "root"

    def test_get_logger_returns_same_instance(self):
        """Test get_logger returns the same instance for the same name."""
        logger1 = get_logger("test.same")
        logger2 = get_logger("test.same")

        assert logger1 is logger2
