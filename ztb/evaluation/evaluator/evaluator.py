"""Backward compatibility shim re-exporting `ztb.analysis.evaluator.TradingEvaluator`.
"""

from ztb.analysis.evaluator.evaluator import (
    EvaluationResult,
    SingleEpisodeResultDict,
    TradingEvaluator,
)

__all__ = ["TradingEvaluator", "EvaluationResult", "SingleEpisodeResultDict"]
