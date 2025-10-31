#!/usr/bin/env python3
"""
Action Signal Guide Backtest Configuration

This module contains configuration settings for the ActionSignalGuide backtest.
"""

from ztb.trading.strategies.action_signal_guide.action_signal_guide import (
    ActionSignalGuideConfig,
    GuidanceLevel,
)


def get_backtest_config_for_pattern(pattern_name: str = None):
    """Get ActionSignalGuide configuration for backtesting specific pattern."""
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
        pattern_attr = f"enable_{pattern_name}_patterns"
        if hasattr(config, pattern_attr):
            setattr(config, pattern_attr, True)

    return config


def get_backtest_config():
    """Get ActionSignalGuide configuration for backtesting (legacy function)."""
    return get_backtest_config_for_pattern()  # All disabled by default


def get_engine_config():
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
