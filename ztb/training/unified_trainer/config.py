#!/usr/bin/env python3
"""
Configuration classes for Unified Trainer.
"""

from ztb.config.managers.ztb_manager import ZaifTradeBotConfigManager as ConfigManager
from ztb.config.schemas.zaif import TrainingConfig, ZaifTradeBotConfig
from dataclasses import dataclass
from typing import Any, Dict, Optional


@dataclass
class UnifiedTrainerConfig:
    """Configuration for Unified Trainer."""
    algorithm: str = "SAC"
    total_timesteps: int = 100000
    config_path: Optional[str] = None
    model_save_path: Optional[str] = None
    log_dir: Optional[str] = None
    additional_params: Optional[Dict[str, Any]] = None


def load_config(config_path: str) -> ZaifTradeBotConfig:
    """
    Load configuration using the new ConfigManager.

    Args:
        config_path: Path to configuration file

    Returns:
        Validated ZaifTradeBotConfig object
    """
    config_manager = ConfigManager.get_instance()
    return config_manager.load_config(config_path)


def get_training_config(config: ZaifTradeBotConfig) -> TrainingConfig:
    """
    Extract training configuration from global config.

    Args:
        config: Global configuration object

    Returns:
        Training configuration
    """
    if config.training is None:
        raise ValueError("Training configuration not found in global config")
    return config.training
