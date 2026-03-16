#!/usr/bin/env python3
"""Logging utilities for consistent logging setup across the codebase."""

from __future__ import annotations

import json
import logging
import time
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import TYPE_CHECKING, Mapping

from ztb.utils.types import LoggerProtocol

if TYPE_CHECKING:
    from ztb.types.common import ConfigDict

BYTES_PER_MB = 1024 * 1024

def setup_logging(
    level: int = logging.INFO,
    format_string: str | None = None,
    log_file: str | None = None,
    max_bytes: int = 10 * BYTES_PER_MB,  # 10MB
    backup_count: int = 5,
) -> None:
    """
    set up basic logging configuration with optional file rotation.

    Args:
        level: Logging level (default: logging.INFO)
        format_string: Custom format string (optional)
        log_file: Log file path for file logging with rotation (optional)
        max_bytes: Maximum log file size before rotation (default: 10MB)
        backup_count: Number of backup log files to keep (default: 5)
    """
    if format_string is None:
        format_string = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

    # Create formatter
    formatter = logging.Formatter(format_string)

    # Get root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(level)

    # Remove existing handlers to avoid duplicates
    root_logger.handlers.clear()

    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)

    # File handler with rotation (Bug #40 fix)
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)

        file_handler = RotatingFileHandler(
            filename=log_file,
            maxBytes=max_bytes,
            backupCount=backup_count,
            encoding="utf-8",
        )
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)

def setup_logging_from_config(config: Mapping[str, object]) -> None:
    """
    set up logging from configuration dictionary.

    Args:
        config: Configuration dictionary with logging settings
    """
    from ztb.utils.config_helpers import get_dict, get_int, get_string

    logging_config = get_dict(config, "logging")

    level_str = get_string(logging_config, "level", "INFO").upper()
    level = getattr(logging, level_str, logging.INFO)

    format_string = get_string(logging_config, "format")
    log_file = get_string(logging_config, "file")
    max_bytes = get_int(logging_config, "max_bytes", 10 * BYTES_PER_MB)
    backup_count = get_int(logging_config, "backup_count", 5)

    setup_logging(
        level=level,
        format_string=format_string,
        log_file=log_file,
        max_bytes=max_bytes,
        backup_count=backup_count,
    )

def configure_log_levels(config: Mapping[str, object]) -> None:
    """
    Configure specific log levels for different modules.

    Args:
        config: Configuration dictionary with module log levels
    """
    from ztb.utils.config_helpers import get_dict

    logging_config = get_dict(config, "logging")
    module_levels = get_dict(logging_config, "module_levels")

    for module, level_val in module_levels.items():
        if isinstance(level_val, str):
            level = getattr(logging, level_val.upper(), logging.INFO)
            logging.getLogger(module).setLevel(level)

def get_logger(name: str) -> logging.Logger:
    """
    Get a logger instance for the given name.

    Logger names should follow the module hierarchy (e.g., 'ztb.trading.order') to enable granular log level control.
    It is recommended to use fully qualified module paths for logger names to maintain consistency and leverage Python's logger hierarchy.

    Args:
        name: Logger name (use module path, e.g., 'ztb.trading.order')

    Returns:
        Logger instance
    """
    return logging.getLogger(name)

class StructuredLogger(LoggerProtocol):
    """
    Structured logging utility for consistent log formatting.

    Supports JSON output and contextual logging for better log analysis.

    Example:
        logger = StructuredLogger("trade", json_format=True)
        logger.set_context(user_id=123, session="abc")
        logger.info("Order executed", extra={"order_id": "xyz"})

    Recommended Use Cases:
        - When you need structured logs for downstream analysis (e.g., ELK stack, BigQuery)
        - When you want to include contextual information (user/session/order IDs) in every log
        - For services requiring both human-readable and machine-readable logs
    """

    def __init__(self, name: str, json_format: bool = False):
        self.logger = get_logger(name)
        self.json_format = json_format
        self.context: dict[str, object] = {}

    def set_context(self, **kwargs: object) -> None:
        """set logging context that will be included in all log messages."""
        self.context.update(kwargs)

    def clear_context(self) -> None:
        """Clear all logging context."""
        self.context.clear()

    def _format_message(
        self, message: str, extra: dict[str, object] | None = None
    ) -> str:
        """Format message with context and extra data."""
        log_data = {**self.context}
        if extra:
            log_data.update(extra)

        if self.json_format:
            return json.dumps(
                {"message": message, "timestamp": time.time(), **log_data}
            )
        else:
            if log_data:
                context_str = ", ".join(f"{k}={v}" for k, v in log_data.items())
                return f"{message} [{context_str}]"
            return message

    def info(self, message: str, extra: dict[str, object] | None = None) -> None:
        """Log info message with context."""
        self.logger.info(self._format_message(message, extra))

    def warning(self, message: str, extra: dict[str, object] | None = None) -> None:
        """Log warning message with context."""
        self.logger.warning(self._format_message(message, extra))

    def error(self, message: str, extra: dict[str, object] | None = None) -> None:
        """Log error message with context."""
        self.logger.error(self._format_message(message, extra))

    def debug(self, message: str, extra: dict[str, object] | None = None) -> None:
        """Log debug message with context."""
        self.logger.debug(self._format_message(message, extra))

def create_structured_logger(name: str, json_format: bool = False) -> StructuredLogger:
    """
    Create a structured logger instance.

    Args:
        name: Logger name
        json_format: Whether to output logs in JSON format

    Returns:
        StructuredLogger instance
    """
    return StructuredLogger(name, json_format)

# Standardized logging functions for common operations
def log_data_loading(logger: logging.Logger, file_path: str, rows: int) -> None:
    """Standardized logging for data loading operations."""
    logger.info(f"✅ Data loaded: {rows:,} rows from {file_path}")

def log_model_loading(logger: logging.Logger, model_path: str) -> None:
    """Standardized logging for model loading operations."""
    logger.info(f"✅ Model loaded from {model_path}")

def log_training_start(
    logger: logging.Logger, model_name: str, config_summary: str = ""
) -> None:
    """Standardized logging for training start."""
    message = f"🚀 Starting training for {model_name}"
    if config_summary:
        message += f" with config: {config_summary}"
    logger.info(message)

def log_backtest_start(
    logger: logging.Logger, model_name: str, data_info: str = ""
) -> None:
    """Standardized logging for backtest start."""
    message = f"🚀 Starting backtest for {model_name}"
    if data_info:
        message += f" using {data_info}"
    logger.info(message)

def log_analysis_start(
    logger: logging.Logger, analysis_type: str, target: str = ""
) -> None:
    """Standardized logging for analysis start."""
    message = f"📊 Starting {analysis_type} analysis"
    if target:
        message += f" for {target}"
    logger.info(message)

def log_success(logger: logging.Logger, operation: str, details: str = "") -> None:
    """Standardized logging for successful operations."""
    message = f"✅ {operation} completed successfully"
    if details:
        message += f": {details}"
    logger.info(message)

def log_error(logger: logging.Logger, operation: str, error: str) -> None:
    """Standardized logging for errors."""
    logger.error(f"❌ {operation} failed: {error}")

def log_warning(logger: logging.Logger, message: str) -> None:
    """Standardized logging for warnings."""
    logger.warning(f"⚠️ {message}")
