#!/usr/bin/env python3
"""
Trading Evaluator module for Zaif Trade Bot.
"""

import warnings

from ztb.analysis.evaluator.evaluator import TradingEvaluator
from ztb.analysis.evaluator.types import (
    EvaluationResult,
    SingleEpisodeResultDict,
)
from ztb.types.common import ConfigDict

__all__ = [
    "TradingEvaluator",
    "EvaluationResult",
    "ConfigDict",
    "SingleEpisodeResultDict",
]

warnings.warn(
    "ztb.analysis.evaluator is deprecated; use ztb.evaluation.unified_evaluation",
    DeprecationWarning,
    stacklevel=2,
)
