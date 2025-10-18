"""
Trading Analysis Tools

This module provides comprehensive analysis tools for trading strategies including:
- Backtest analysis with risk metrics
- Performance evaluation and reporting
- Market condition analysis
- Robustness testing across different market regimes
"""

from .analyze_backtest import BacktestAnalyzer

__all__ = [
    "BacktestAnalyzer",
]