"""
Deprecated shim for regime evaluation.

Use ztb.analysis.regime.regime_eval as the source of truth.
"""

from __future__ import annotations

import warnings

from ztb.analysis.regime.regime_eval import RegimeDetector, RegimeEvaluator

warnings.warn(
    "ztb.evaluation.regime_eval is deprecated; "
    "use ztb.analysis.regime.regime_eval",
    DeprecationWarning,
    stacklevel=2,
)

__all__ = ["RegimeDetector", "RegimeEvaluator"]
