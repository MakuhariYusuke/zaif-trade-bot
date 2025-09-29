#!/usr/bin/env python3
"""
config.py
Central configuration management for ZTB system
"""

import os
from typing import Any, Optional


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
        return value.lower() in ('true', '1', 'yes', 'on')

    def get_int(self, key: str, default: int = 0) -> int:
        """Get integer configuration value"""
        value = os.getenv(key)
        if value is None:
            return default
        try:
            return int(value)
        except ValueError:
            print(f"Warning: Invalid integer value for {key}: {value}, using default {default}")
            return default

    def get_float(self, key: str, default: float = 0.0) -> float:
        """Get float configuration value with type validation"""
        value = os.getenv(key)
        if value is None:
            return default
        try:
            return float(value)
        except ValueError:
            print(f"Warning: Invalid float value for {key}: {value}, using default {default}")
            return default

    def log_config(self) -> None:
        """Log current configuration for debugging"""
        config_vars = [
            'ZTB_MEM_PROFILE', 'ZTB_CUDA_WARN_GB', 'ZTB_LOG_LEVEL',
            'ZTB_CHECKPOINT_INTERVAL', 'ZTB_MAX_MEMORY_GB',
            'ZTB_TEST_ISOLATION', 'ZTB_FLOAT_TOLERANCE'
        ]
        print("Current ZTB Configuration:")
        for var in config_vars:
            value = os.getenv(var)
            if value is not None:
                print(f"  {var}={value}")


# Global instance
config = ZTBConfig()