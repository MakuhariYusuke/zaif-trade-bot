#!/usr/bin/env python3
"""
Algorithm-specific training implementations for Unified Trainer.
"""

import logging
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional

from ztb.types.common import ConfigDict
from ztb.utils.logging_utils import get_logger


class BaseAlgorithmTrainer(ABC):
    """
    Abstract base class for algorithm-specific trainers.
    """

    def __init__(self, config: ConfigDict, logger: Optional[logging.Logger] = None):
        self.config = config
        self.logger = logger or get_logger(__name__)

    @abstractmethod
    def validate_config(self) -> bool:
        """Validate trainer configuration."""
        pass

    @abstractmethod
    def train(self) -> bool:
        """Execute training."""
        pass

    @abstractmethod
    def get_training_stats(self) -> Dict[str, Any]:
        """Get training statistics."""
        pass
