"""
Base components for unified training system.
"""

from .base_trainer import BaseAlgorithmTrainer
from .callbacks import TrainingProgressCallback

__all__ = [
    "TrainingProgressCallback",
    "BaseAlgorithmTrainer",
]
