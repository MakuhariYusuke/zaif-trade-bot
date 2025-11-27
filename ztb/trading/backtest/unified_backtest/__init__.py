#!/usr/bin/env python3
"""
Unified Backtest Framework

A comprehensive backtesting framework that supports multiple trading strategies
including SAC models, Action Signal Guide, and hybrid approaches. Designed to
leverage SAC learning outcomes for enhanced analysis and strategy evaluation.
"""

from .unified_backtester import (
    UnifiedBacktester,
    BacktestConfig,
    BacktestResult,
)
from .strategy_base import (
    BaseTradingStrategy,
    MLTradingStrategy,
    SignalBasedStrategy,
    TradingStrategy,
)
from .sac_strategy import SACStrategy
from .action_signal_guide_strategy import ActionSignalGuideStrategy
from .analyzer import BacktestAnalyzer
from .signal_performance import (
    BacktestPerformanceAnalyzer,
    BacktestSignalPerformanceAnalyzer,
    SignalTracker,
)

__version__ = "1.0.0"
__all__ = [
    "UnifiedBacktester",
    "BacktestConfig",
    "BacktestResult",
    "TradingStrategy",
    "BaseTradingStrategy",
    "MLTradingStrategy",
    "SignalBasedStrategy",
    "SACStrategy",
    "ActionSignalGuideStrategy",
    "BacktestAnalyzer",
    "BacktestPerformanceAnalyzer",
    "BacktestSignalPerformanceAnalyzer",
    "SignalTracker",
]