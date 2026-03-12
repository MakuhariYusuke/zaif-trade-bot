"""
Deprecated backtest metrics module.

Use ztb.metrics.metrics for BacktestMetrics and MetricsCalculator.
"""

from ztb.metrics.metrics import BacktestMetrics
from ztb.metrics.metrics import MetricsCalculator

__all__ = ["BacktestMetrics", "MetricsCalculator"]
