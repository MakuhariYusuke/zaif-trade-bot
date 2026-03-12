"""
Deprecated shim for trading metrics.

Use ztb.metrics.metrics as the single source of truth.
"""

from __future__ import annotations

import warnings

from ztb.metrics.metrics import action_distribution
from ztb.metrics.metrics import calmar_ratio
from ztb.metrics.metrics import calculate_delta_sharpe
from ztb.metrics.metrics import calculate_feature_metrics
from ztb.metrics.metrics import max_drawdown
from ztb.metrics.metrics import sharpe_ratio
from ztb.metrics.metrics import sharpe_with_stats
from ztb.metrics.metrics import sortino_ratio
from ztb.metrics.metrics import validate_ablation_results
from ztb.metrics.metrics import win_rate

warnings.warn(
    "ztb.utils.trading_metrics is deprecated; use ztb.metrics.metrics",
    DeprecationWarning,
    stacklevel=2,
)

__all__ = [
    "action_distribution",
    "calmar_ratio",
    "calculate_delta_sharpe",
    "calculate_feature_metrics",
    "max_drawdown",
    "sharpe_ratio",
    "sharpe_with_stats",
    "sortino_ratio",
    "validate_ablation_results",
    "win_rate",
]
