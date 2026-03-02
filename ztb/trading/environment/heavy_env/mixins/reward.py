"""Reward configuration helpers for HeavyTradingEnv."""

from __future__ import annotations

from typing import Any

def _get_reward_setting_int(self: Any, key: str, default: int) -> int:
    """Type-safe getter for integer reward settings."""
    if hasattr(self.reward_settings, "get"):
        value = self.reward_settings.get(key, default)
    else:
        value = getattr(self.reward_settings, key, default)

    if isinstance(value, (int, float)):
        return int(value)
    return default

def _get_reward_setting_float(self: Any, key: str, default: float) -> float:
    """Type-safe getter for float reward settings."""
    if hasattr(self.reward_settings, "get"):
        value = self.reward_settings.get(key, default)
    else:
        value = getattr(self.reward_settings, key, default)

    if isinstance(value, (int, float)):
        return float(value)
    return default

def _get_reward_setting_bool(self: Any, key: str, default: bool) -> bool:
    """Type-safe getter for boolean reward settings."""
    value = self.reward_settings.get(key, default)
    if isinstance(value, bool):
        return value
    return default
