"""
統合評価フレームワーク

バックテスト、Walk-Forward、クロスバリデーション等の
包括的評価を提供するモジュール。
"""

import warnings

from .unified_evaluation import EvaluationMetric, EvaluationType, UnifiedEvaluator
from .walk_forward_adapter import (
    WalkForwardAggregationStats,
    WalkForwardUnifiedEvaluator,
)
from .walk_forward_integration_pipeline import WalkForwardEvaluationPipeline

warnings.warn(
    "ztb.analysis.evaluation is legacy for unified evaluation; "
    "prefer ztb.evaluation for core evaluators. "
    "Walk-forward adapters remain here for compatibility.",
    DeprecationWarning,
    stacklevel=2,
)

__all__ = [
    "UnifiedEvaluator",
    "EvaluationMetric",
    "EvaluationType",
    "WalkForwardUnifiedEvaluator",
    "WalkForwardAggregationStats",
    "WalkForwardEvaluationPipeline",
]
