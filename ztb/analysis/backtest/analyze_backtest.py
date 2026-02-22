"""Compatibility shim for legacy import path.

Re-export BacktestAnalyzer from the current comparative analyze module.
"""

from ztb.analysis.comparative.analyze_backtest import BacktestAnalyzer

__all__ = ["BacktestAnalyzer"]
