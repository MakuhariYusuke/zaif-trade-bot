#!/usr/bin/env python3
"""
Unified Trainer module for Zaif Trade Bot.
"""

from ztb.config.schema import TrainingConfig, ZaifTradeBotConfig
from ztb.training.unified_trainer.config import get_training_config, load_config
from ztb.training.unified_trainer.trainer import UnifiedTrainer
from ztb.training.unified_trainer.utils import configure_progress_bar

__all__ = [
    "ZaifTradeBotConfig",
    "TrainingConfig",
    "UnifiedTrainer",
    "configure_progress_bar",
    "load_config",
    "get_training_config",
]

# Backwards-compatible alias expected by some modules/tests
UnifiedAlgorithm = UnifiedTrainer
__all__.append("UnifiedAlgorithm")
# Backwards-compatible name used in some legacy tests/scripts
UnifiedTrainerConfig = TrainingConfig
__all__.append("UnifiedTrainerConfig")
