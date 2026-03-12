"""
Deprecated shim for statistical helpers.

Use ztb.metrics.metrics as the single source of truth.
"""

from __future__ import annotations

import warnings

from ztb.metrics.metrics import calculate_atr
from ztb.metrics.metrics import calculate_autocorrelation
from ztb.metrics.metrics import calculate_distribution_stats
from ztb.metrics.metrics import calculate_volatility
from ztb.metrics.metrics import detect_outliers_iqr
from ztb.metrics.metrics import p_mean_method
from ztb.metrics.metrics import rolling_statistics

warnings.warn(
    "ztb.metrics.statistics is deprecated; use ztb.metrics.metrics",
    DeprecationWarning,
    stacklevel=2,
)

__all__ = [
    "calculate_atr",
    "calculate_autocorrelation",
    "calculate_distribution_stats",
    "calculate_volatility",
    "detect_outliers_iqr",
    "p_mean_method",
    "rolling_statistics",
]
