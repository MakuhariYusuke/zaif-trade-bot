#!/usr/bin/env python3
"""
Unified Backtest Framework

A comprehensive backtesting framework that supports multiple trading strategies
including SAC models, Action Signal Guide, and hybrid approaches. Designed to
leverage SAC learning outcomes for enhanced analysis and strategy evaluation.
"""

from .config import get_backtest_config, get_engine_config
from .data_generator import generate_synthetic_data, generate_trending_data
from .results_runner import (
    display_backtest_results,
    display_signal_statistics,
    save_results_to_file,
)
from .unified_backtester import UnifiedBacktester
from .strategies import SACStrategy, ActionSignalGuideStrategy, HybridStrategy

__all__ = [
    # Legacy Action Signal Guide functions
    "get_backtest_config",
    "get_engine_config",
    "generate_synthetic_data",
    "generate_trending_data",
    "display_backtest_results",
    "display_signal_statistics",
    "save_results_to_file",
    # Unified backtest framework
    "UnifiedBacktester",
    "SACStrategy",
    "ActionSignalGuideStrategy",
    "HybridStrategy",
]
