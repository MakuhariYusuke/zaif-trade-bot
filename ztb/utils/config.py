#!/usr/bin/env python3
"""
config.py
Central configuration management for ZTB system
"""

import logging
import os
from typing import Any

logger = logging.getLogger(__name__)


class ZTBConfig:
    """Central configuration management for all ZTB components"""

    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value from environment variables"""
        return os.getenv(key, default)

    def get_bool(self, key: str, default: bool = False) -> bool:
        """Get boolean configuration value"""
        value = os.getenv(key)
        if value is None:
            return default
        return value.lower() in ("true", "1", "yes", "on")

    def get_int(self, key: str, default: int = 0) -> int:
        """Get integer configuration value"""
        value = os.getenv(key)
        if value is None:
            return default
        try:
            return int(value)
        except ValueError:
            logger.warning(
                f"Invalid integer value for {key}: {value}, using default {default}"
            )
            return default

    def get_float(self, key: str, default: float = 0.0) -> float:
        """Get float configuration value with type validation"""
        value = os.getenv(key)
        if value is None:
            return default
        try:
            return float(value)
        except ValueError:
            logger.warning(
                f"Invalid float value for {key}: {value}, using default {default}"
            )
            return default

    def log_config(self) -> None:
        """Log current configuration for debugging"""
        config_vars = [
            "ZTB_MEM_PROFILE",
            "ZTB_CUDA_WARN_GB",
            "ZTB_LOG_LEVEL",
            "ZTB_CHECKPOINT_INTERVAL",
            "ZTB_MAX_MEMORY_GB",
            "ZTB_TEST_ISOLATION",
            "ZTB_FLOAT_TOLERANCE",
        ]
        logger.info("Current ZTB Configuration:")
        for var in config_vars:
            value = os.getenv(var)
            if value is not None:
                logger.info(f"  {var}={value}")


def get_config_value(config_dict: dict[str, Any], key: str, expected_type: type, default: Any = None) -> Any:
    """
    Safely extract and convert configuration values from dict with type validation.
    
    Args:
        config_dict: Configuration dictionary
        key: Key to extract
        expected_type: Expected type (str, int, float, bool)
        default: Default value if key not found or conversion fails
        
    Returns:
        Converted value or default
    """
    raw_value = config_dict.get(key)
    try:
        if raw_value is None:
            return default
            
        if expected_type == str:
            return str(raw_value)
        elif expected_type == int:
            return int(raw_value) if isinstance(raw_value, (int, str)) else default
        elif expected_type == float:
            return float(raw_value) if isinstance(raw_value, (int, float, str)) else default
        elif expected_type == bool:
            if isinstance(raw_value, bool):
                return raw_value
            elif isinstance(raw_value, str):
                return raw_value.lower() in ("true", "1", "yes", "on")
            else:
                return default
        else:
            return raw_value
    except (ValueError, TypeError):
        logger.warning(f"Failed to convert config value for {key}: {raw_value}, using default {default}")
        return default


# Global instance
config = ZTBConfig()
