#!/usr/bin/env python3
"""
Logging utilities for consistent logging setup across the codebase.
"""

import logging
from typing import Optional


def setup_logging(level: int = logging.INFO, format_string: Optional[str] = None) -> None:
    """
    Set up basic logging configuration.

    Args:
        level: Logging level (default: logging.INFO)
        format_string: Custom format string (optional)
    """
    if format_string is None:
        format_string = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

    logging.basicConfig(level=level, format=format_string)


def get_logger(name: str) -> logging.Logger:
    """
    Get a logger with the specified name.

    Args:
        name: Logger name

    Returns:
        Configured logger
    """
    return logging.getLogger(name)