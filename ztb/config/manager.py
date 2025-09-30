"""
Centralized configuration management.
"""

import os
from pathlib import Path
from typing import Any, Dict, Optional

from .loader import ConfigLoader
from .schema import GlobalConfig


class ConfigManager:
    """Centralized configuration manager."""

    _instance: Optional["ConfigManager"] = None
    _config: Optional[GlobalConfig] = None

    def __init__(self):
        if ConfigManager._instance is not None:
            raise RuntimeError("ConfigManager is a singleton")
        ConfigManager._instance = self
        self.loader = ConfigLoader()

    @classmethod
    def get_instance(cls) -> "ConfigManager":
        """Get singleton instance."""
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def load_config(self, config_path: Optional[str] = None) -> GlobalConfig:
        """Load and merge configuration from all sources."""
        if config_path:
            self.loader.load_yaml(config_path)

        self.loader.load_env()
        # CLI args would be loaded here if available

        merged = self.loader.merge_configs()
        self._config = GlobalConfig(**merged)
        return self._config

    def get_config(self) -> GlobalConfig:
        """Get current configuration."""
        if self._config is None:
            self.load_config()
        return self._config

    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value by key."""
        config = self.get_config()
        keys = key.split(".")
        value = config.dict()
        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default
        return value

    def set(self, key: str, value: Any) -> None:
        """Set configuration value."""
        # This would update the config, but for simplicity, just update the dict
        config_dict = self.get_config().dict()
        keys = key.split(".")
        d = config_dict
        for k in keys[:-1]:
            d = d.setdefault(k, {})
        d[keys[-1]] = value
        self._config = GlobalConfig(**config_dict)


# Global instance
config_manager = ConfigManager.get_instance()