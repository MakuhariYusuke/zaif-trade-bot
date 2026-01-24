"""
Configuration management system for Zaif Trade Bot.

This module provides centralized configuration management with support for:
- YAML configuration files
- Environment variable overrides
- Configuration validation and type safety
- Dynamic configuration reloading
"""

from ztb.config.managers.ztb_manager import ZaifTradeBotConfigManager as ConfigManager
from ztb.config.schemas import GlobalConfig

__all__ = [
    "ConfigManager",
    "GlobalConfig",
]
