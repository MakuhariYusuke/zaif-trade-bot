#!/usr/bin/env python3
"""
Logging utilities for consistent logging setup across the codebase.
"""

import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Any, Dict, Optional


def setup_logging(
    level: int = logging.INFO,
    format_string: Optional[str] = None,
    log_file: Optional[str] = None,
    max_bytes: int = 10 * 1024 * 1024,  # 10MB
    backup_count: int = 5,
) -> None:
    """
    Set up basic logging configuration with optional file rotation.

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
        

def setup_logging_from_config(config: Dict[str, Any]) -> None:
    """
    Set up logging from configuration dictionary.

    Args:
        config: Configuration dictionary with logging settings
    """
    logging_config = config.get("logging", {})

    level_str = logging_config.get("level", "INFO").upper()
    level = getattr(logging, level_str, logging.INFO)

    format_string = logging_config.get("format")
    log_file = logging_config.get("file")
    max_bytes = logging_config.get("max_bytes", 10 * 1024 * 1024)
    backup_count = logging_config.get("backup_count", 5)

    setup_logging(
        level=level,
        format_string=format_string,
        log_file=log_file,
        max_bytes=max_bytes,
        backup_count=backup_count
    )


def configure_log_levels(config: Dict[str, Any]) -> None:
    """
    Configure specific log levels for different modules.

    Args:
        config: Configuration dictionary with module log levels
    """
    module_levels = config.get("logging", {}).get("module_levels", {})

    for module, level_str in module_levels.items():
        level = getattr(logging, level_str.upper(), logging.INFO)
        logging.getLogger(module).setLevel(level)


def get_logger(name: str) -> logging.Logger:
    """
    Get a logger instance for the given name.
    
    Args:
        name: Logger name
        
    Returns:
        Logger instance
    """
    return logging.getLogger(name)
