#!/usr/bin/env python3
"""
Action Signal Guide Backtest Configuration

This module contains configuration settings for the ActionSignalGuide backtest.
"""

import logging
from typing import Optional

from ztb.trading.strategies.action_signal_guide.action_signal_guide import (
    ActionSignalGuideConfig,
    GuidanceLevel,
)

logger = logging.getLogger(__name__)

# Valid pattern names for validation
VALID_PATTERNS = [
    "candlestick",
    "fibonacci",
    "gann",
    "wave",
    "harmonic",
    "oscillator",
    "volume",
    "bollinger",
    "adx",
    "granville",
    "heikin_ashi",
    "dow_theory",
]


def validate_pattern_name(pattern_name: str) -> bool:
    """Validate if pattern name is supported."""
    if pattern_name not in VALID_PATTERNS:
        logger.warning(
            f"Unsupported pattern name: {pattern_name}. Valid patterns: {VALID_PATTERNS}"
        )
        return False
    return True


def get_backtest_config_for_pattern(
    pattern_name: Optional[str] = None,
) -> ActionSignalGuideConfig:
    """
    Get ActionSignalGuide configuration for backtesting specific pattern.

    Args:
        pattern_name: Name of the pattern to enable. If None, all patterns disabled.

    Returns:
        Configured ActionSignalGuideConfig instance.
    """
    # Base config with all patterns disabled
    config = ActionSignalGuideConfig(
        debug_short_mode=False,
        guidance_level=GuidanceLevel.WEAK,
        error_suppression_threshold=0,  # Suppress all error logs
        enable_candlestick_patterns=False,
        enable_fibonacci_patterns=False,
        enable_gann_patterns=False,
        enable_wave_patterns=False,
        enable_harmonic_patterns=False,
        enable_oscillator_patterns=False,
        enable_volume_patterns=False,
        enable_bollinger_patterns=False,
        enable_adx_patterns=False,
        enable_granville_patterns=False,
        enable_heikin_ashi_patterns=False,
        enable_dow_theory_patterns=False,
    )

    # Enable specific pattern if requested
    if pattern_name:
        if not validate_pattern_name(pattern_name):
            logger.error(
                f"Invalid pattern name '{pattern_name}', returning config with all patterns disabled"
            )
            return config

        pattern_attr = f"enable_{pattern_name}_patterns"
        if hasattr(config, pattern_attr):
            setattr(config, pattern_attr, True)
            logger.info(f"Enabled {pattern_name} patterns for backtesting")
        else:
            logger.error(f"Pattern attribute {pattern_attr} not found in config")

    # Log enabled patterns for debugging
    enabled_patterns = [
        name
        for name in VALID_PATTERNS
        if getattr(config, f"enable_{name}_patterns", False)
    ]
    if enabled_patterns:
        logger.info(f"Enabled patterns: {enabled_patterns}")
    else:
        logger.info("All patterns disabled (baseline config)")

    return config


def get_backtest_config() -> ActionSignalGuideConfig:
    """Get ActionSignalGuide configuration for backtesting (legacy function)."""
    return get_backtest_config_for_pattern()  # All disabled by default


def get_engine_config() -> dict:
    """Get backtest engine configuration."""
    return {
        "initial_capital": 100000.0,  # $100k starting capital
        "commission": 0.001,  # 0.1% commission
        "slippage": 0.0005,  # 0.05% slippage
        "enable_risk_management": True,
        "max_position_size": 0.1,  # Max 10% of capital per position
        "stop_loss": 0.05,  # 5% stop loss
        "take_profit": 0.1,  # 10% take profit
        "max_drawdown": 0.2,  # 20% max drawdown
    }
    """Get backtest engine configuration."""
    return {
        "initial_capital": 100000.0,  # $100k starting capital
        "commission": 0.001,  # 0.1% commission
        "slippage": 0.0005,  # 0.05% slippage
        "enable_risk_management": True,
        "max_position_size": 0.1,  # Max 10% of capital per position
        "stop_loss": 0.05,  # 5% stop loss
        "take_profit": 0.1,  # 10% take profit
        "max_drawdown": 0.2,  # 20% max drawdown
    }
