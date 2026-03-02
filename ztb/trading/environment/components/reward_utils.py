"""
Reward Calculator Utilities - Helper functions for reward calculation.

This module contains utility functions and helper methods used by the reward calculator.
"""

from typing import Any

class RewardUtils:
    """Utility functions for reward calculation settings and common operations."""

    @staticmethod
    def get_setting_int(
        settings: dict[str, Any] | None, key: str, default: int
    ) -> int:
        """Get integer reward setting with fallback."""
        if settings and key in settings:
            value = settings[key]
            if isinstance(value, (int, float)):
                return int(value)
        return default

    @staticmethod
    def get_setting_float(
        settings: dict[str, Any] | None, key: str, default: float
    ) -> float:
        """Get float reward setting with fallback."""
        if settings and key in settings:
            value = settings[key]
            if isinstance(value, (int, float)):
                return float(value)
        return default

    @staticmethod
    def get_setting_bool(
        settings: dict[str, Any] | None, key: str, default: bool
    ) -> bool:
        """Get boolean reward setting with fallback."""
        if settings and key in settings:
            value = settings[key]
            if isinstance(value, bool):
                return value
            if isinstance(value, (int, float)):
                return bool(value)
            if isinstance(value, str):
                return value.lower() in {"true", "1", "yes", "y", "on"}
        return default

    @staticmethod
    def safe_divide(
        numerator: float, denominator: float, default: float = 0.0
    ) -> float:
        """Safely divide two numbers, returning default if denominator is zero."""
        return numerator / denominator if denominator != 0 else default

    @staticmethod
    def clamp(value: float, min_val: float, max_val: float) -> float:
        """Clamp a value between min and max."""
        return max(min_val, min(max_val, value))
