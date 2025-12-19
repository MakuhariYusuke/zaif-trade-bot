"""Base trainer class with common functionality."""

import logging
from typing import Any, Dict

from ztb.training.utils.logging_utils import get_logger


class BaseTrainer:
    """Base class for all trainers."""

    def __init__(self, config: Dict[str, Any]) -> None:
        """
        Initialize base trainer.

        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.logger = get_logger(self.__class__.__name__)