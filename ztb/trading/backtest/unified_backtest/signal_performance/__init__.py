"""
Signal Performance Analysis for Unified Backtest Framework

This package provides signal performance analysis capabilities integrated
with the unified backtest framework, enabling quantitative evaluation of
Action Signal Guide signals during backtesting.
"""

from .backtest_integration import BacktestSignalPerformanceAnalyzer
from .signal_tracker import SignalTracker
from .performance_analyzer import BacktestPerformanceAnalyzer

__all__ = [
    "BacktestSignalPerformanceAnalyzer",
    "SignalTracker",
    "BacktestPerformanceAnalyzer",
]
