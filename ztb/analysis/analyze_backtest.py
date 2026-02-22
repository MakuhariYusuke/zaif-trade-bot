"""Compatibility shim for analyze_backtest

Provides the older import path `ztb.analysis.analyze_backtest` by
re-exporting symbols from `ztb.analysis.backtest.analyze_backtest`.
"""
from .backtest.analyze_backtest import BacktestAnalyzer

__all__ = ["BacktestAnalyzer"]
