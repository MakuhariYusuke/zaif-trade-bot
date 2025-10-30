#!/usr/bin/env python3
"""
Action Signal Guide Backtest Package

This package contains modules for backtesting the ActionSignalGuide strategy.
"""

from .config import get_backtest_config, get_engine_config
from .data_generator import generate_synthetic_data, generate_trending_data
from .results_runner import (
    display_backtest_results,
    display_signal_statistics,
    save_results_to_file,
)

__all__ = [
    "get_backtest_config",
    "get_engine_config",
    "generate_synthetic_data",
    "generate_trending_data",
    "display_backtest_results",
    "display_signal_statistics",
    "save_results_to_file",
]
