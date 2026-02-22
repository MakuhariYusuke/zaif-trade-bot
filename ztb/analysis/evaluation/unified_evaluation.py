"""
Deprecated shim for unified evaluation.

Use ztb.evaluation.unified_evaluation as the source of truth.
"""

from __future__ import annotations

import warnings

from ztb.evaluation.unified_evaluation import (
    ComprehensiveEvaluation,
    EvaluationMetric,
    EvaluationType,
    UnifiedEvaluator,
)

warnings.warn(
    "ztb.analysis.evaluation.unified_evaluation is deprecated; "
    "use ztb.evaluation.unified_evaluation",
    DeprecationWarning,
    stacklevel=2,
)

__all__ = [
    "ComprehensiveEvaluation",
    "EvaluationMetric",
    "EvaluationType",
    "UnifiedEvaluator",
]
