"""
Deprecated shim for statistical significance helpers.

Use ztb.metrics.metrics as the single source of truth.
"""

from __future__ import annotations

import warnings

from ztb.metrics.metrics import (
    calculate_bootstrap_pvalue,
    calculate_deflated_sharpe_ratio,
)

warnings.warn(
    "ztb.metrics.statistical_significance is deprecated; use ztb.metrics.metrics",
    DeprecationWarning,
    stacklevel=2,
)

__all__ = [
    "calculate_deflated_sharpe_ratio",
    "calculate_bootstrap_pvalue",
]
