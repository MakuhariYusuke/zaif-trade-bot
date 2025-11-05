"""
Configuration module for Feature Weight Adjustment System
"""

from .adjustment_config import (
    AdjustmentConfig,
    AdjustmentStrategyType,
    AdjustmentFrequency,
    DEFAULT_BACKTEST_CONFIG,
    DEFAULT_LIVE_TRADING_CONFIG,
    DEFAULT_AGGRESSIVE_CONFIG,
)

__all__ = [
    'AdjustmentConfig',
    'AdjustmentStrategyType',
    'AdjustmentFrequency',
    'DEFAULT_BACKTEST_CONFIG',
    'DEFAULT_LIVE_TRADING_CONFIG',
    'DEFAULT_AGGRESSIVE_CONFIG',
]