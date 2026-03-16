#!/usr/bin/env python3
"""
Utility functions for live trading.
"""

import logging
from datetime import datetime

from ztb.utils.logging_utils import get_logger
from ztb.utils.path_utils import get_project_root

PROJECT_ROOT = get_project_root()

def _configure_live_logging(level: str) -> logging.Logger:
    """Configure logging for live trading run and return module logger."""
    numeric_level = getattr(logging, level.upper(), logging.INFO)

    logging.basicConfig(
        level=numeric_level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    logger = _setup_live_logger()
    logger.setLevel(numeric_level)
    return logger

def _setup_live_logger() -> logging.Logger:
    """Ensure module logger writes to rotating log file and console exactly once."""
    logger = get_logger(__name__)
    if getattr(logger, "_ztb_live_configured", False):
        return logger

    log_dir = PROJECT_ROOT / "logs"
    log_dir.mkdir(exist_ok=True)
    log_file = (
        log_dir / f"live_trading_{datetime.now().strftime('%Y%m%d_%H-%M-%S_%f')}.log"
    )
    logger.info(f"log_file: {log_file}")

    file_handler = logging.FileHandler(str(log_file))
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(
        logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    )

    # Avoid duplicate handlers when running multiple sessions in same interpreter
    logger.handlers.clear()
    logger.addHandler(file_handler)

    if not any(isinstance(h, logging.StreamHandler) for h in logger.handlers):
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        console_handler.setFormatter(
            logging.Formatter("%(asctime)s - %(name)s - %(message)s")
        )
        logger.addHandler(console_handler)

    logger.propagate = False
    setattr(logger, "_ztb_live_configured", True)
    setattr(logger, "log_file", log_file)
    logger.info("Live trading log file: %s", log_file)
    return logger
