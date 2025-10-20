"""Backward-compatible exports for trading environments."""

from __future__ import annotations

import gymnasium as gym

from ztb.trading.environment.heavy_env.core import FlipHeavyTradingEnv, HeavyTradingEnv  # noqa: F401
from ztb.trading.environment.utils.config import EnvironmentConfig, RewardSettings

__all__ = [
    "HeavyTradingEnv",
    "FlipHeavyTradingEnv",
    "EnvironmentConfig",
    "RewardSettings",
]

gym.register(
    id="HeavyTradingEnv",
    entry_point="ztb.trading.environment:HeavyTradingEnv",
    kwargs={},
)
