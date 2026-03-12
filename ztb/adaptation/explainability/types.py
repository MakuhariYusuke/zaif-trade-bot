"""
Type definitions for Explainability Module
説明可能性モジュールの型定義
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any

class ExplanationType(Enum):
    """説明タイプ"""

    FEATURE_IMPORTANCE = "feature_importance"
    DECISION_REASON = "decision_reason"
    CONTRIBUTION_BREAKDOWN = "contribution_breakdown"
    COUNTERFACTUAL = "counterfactual"

@dataclass
class ExplainabilityFeatureImportance:
    """特徴量重要度"""

    feature_name: str
    importance_score: float
    feature_category: str | None = None
    description: str | None = None
    confidence: float | None = None

# Backwards compatibility: alias name expected by other modules
FeatureImportance = ExplainabilityFeatureImportance

@dataclass
class DecisionExplanation:
    """決定説明"""

    decision_type: str  # BUY, SELL, HOLD
    confidence_score: float
    primary_factors: list[ExplainabilityFeatureImportance]
    contributing_factors: list[ExplainabilityFeatureImportance]
    natural_language_explanation: str | None = None
    visualization_data: dict[str, Any] | None = None

@dataclass
class VisualizationResult:
    """可視化結果"""

    plots: dict[str, Any]  # プロット名 -> プロットデータ
    timestamp: datetime
    format: str  # png, svg, htmlなど

    def to_dict(self) -> dict[str, Any]:
        """辞書形式に変換"""
        return {
            "plots": self.plots,
            "timestamp": self.timestamp.isoformat(),
            "format": self.format,
        }

@dataclass
class ExplanationResult:
    """説明結果"""

    explanation_id: str
    timestamp: datetime
    model_version: str
    explanation_type: ExplanationType
    target_prediction: Any
    feature_importance: list[ExplainabilityFeatureImportance]
    decision_explanation: DecisionExplanation | None = None
    visualization: VisualizationResult | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    processing_time_seconds: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        """辞書形式に変換"""
        return {
            "explanation_id": self.explanation_id,
            "timestamp": self.timestamp.isoformat(),
            "model_version": self.model_version,
            "explanation_type": self.explanation_type.value,
            "target_prediction": self.target_prediction,
            "feature_importance": [
                {
                    "feature_name": fi.feature_name,
                    "importance_score": fi.importance_score,
                    "feature_category": fi.feature_category,
                    "description": fi.description,
                    "confidence": fi.confidence,
                }
                for fi in self.feature_importance
            ],
            "decision_explanation": self.decision_explanation.__dict__
            if self.decision_explanation
            else None,
            "metadata": self.metadata,
            "processing_time_seconds": self.processing_time_seconds,
        }

@dataclass
class ExplanationCache:
    """説明キャッシュ"""

    explanation_id: str
    result: ExplanationResult
    created_at: datetime
    ttl_seconds: int

    @property
    def is_expired(self) -> bool:
        """キャッシュが期限切れかどうか"""
        from datetime import timedelta

        return datetime.now() - self.created_at > timedelta(seconds=self.ttl_seconds)

@dataclass
class VisualizationResult:
    """可視化結果"""

    plots: dict[str, Any]  # プロット名 -> プロットデータ
    timestamp: datetime
    format: str  # png, svg, htmlなど

    def to_dict(self) -> dict[str, Any]:
        """辞書形式に変換"""
        # Removed duplicated VisualizationResult and explanation cache declarations (they are defined above as ExplanationCache)

    total_explanations: int
    explanation_types: dict[str, int]
    top_features: list[ExplainabilityFeatureImportance]
    model_performance_insights: dict[str, Any]
    recommendations: list[str]

    def to_dict(self) -> dict[str, Any]:
        """辞書形式に変換"""
        return {
            "report_id": self.report_id,
            "generated_at": self.generated_at.isoformat(),
            "period_start": self.period_start.isoformat(),
            "period_end": self.period_end.isoformat(),
            "total_explanations": self.total_explanations,
            "explanation_types": self.explanation_types,
            "top_features": [
                {
                    "feature_name": tf.feature_name,
                    "importance_score": tf.importance_score,
                    "feature_category": tf.feature_category,
                }
                for tf in self.top_features
            ],
            "model_performance_insights": self.model_performance_insights,
            "recommendations": self.recommendations,
        }

__all__ = [
    "ExplanationCache",
    "ExplanationResult",
    "ExplainabilityFeatureImportance",
    "DecisionExplanation",
    "VisualizationResult",
    "ExplanationReport",
]
