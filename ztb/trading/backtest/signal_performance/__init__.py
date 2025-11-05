"""
Signal Performance Analysis Package

Provides comprehensive signal performance analysis for trading strategies,
specifically designed for backtest integration.
"""

from .backtest_integration import BacktestSignalPerformanceAnalyzer
from .performance_analyzer import BacktestPerformanceAnalyzer
from .signal_tracker import SignalTracker, TrackedSignal

__all__ = [
    'BacktestSignalPerformanceAnalyzer',
    'BacktestPerformanceAnalyzer',
    'SignalTracker',
    'TrackedSignal'
]