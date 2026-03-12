"""
Machine Learning Integration Interfaces for Action Signal Guide.

Defines interfaces for ML-based pattern optimization and ensemble prediction.
"""

from __future__ import annotations

from abc import abstractmethod
from dataclasses import dataclass
from enum import Enum

import pandas as pd

from ztb.trading.strategies.action_signal_guide.interfaces.common_types import (
    FeatureData,
    GenericData,
    IActionSignalGuideInterface,
    MetadataMap,
    MetricsMap,
    PayloadMap,
    TargetData,
)

class MLPredictionModel(Enum):
    """Types of ML prediction models."""

    LINEAR_REGRESSION = "linear_regression"
    RANDOM_FOREST = "random_forest"
    GRADIENT_BOOSTING = "gradient_boosting"
    NEURAL_NETWORK = "neural_network"
    ENSEMBLE = "ensemble"

class OptimizationTarget(Enum):
    """Optimization targets for ML models."""

    SIGNAL_ACCURACY = "signal_accuracy"
    RISK_ADJUSTED_RETURN = "risk_adjusted_return"
    SHARPE_RATIO = "sharpe_ratio"
    MAX_DRAWDOWN = "max_drawdown"
    WIN_RATE = "win_rate"

@dataclass
class MLTrainingData:
    """Training data structure for ML models."""

    features: FeatureData
    target: TargetData
    feature_names: list[str] | None = None
    target_name: str | None = None
    metadata: MetadataMap | None = None

@dataclass
class MLPredictionResult:
    """Result of ML model prediction."""

    prediction: float
    confidence: float
    feature_importance: dict[str, float]
    model_name: str
    timestamp: float

@dataclass
class MLResult:
    """Generic result structure for ML operations."""

    success: bool
    data: GenericData
    message: str
    metadata: MetadataMap

class IPatternOptimizer(IActionSignalGuideInterface):
    """Interface for pattern combination optimization using ML."""

    @abstractmethod
    def optimize_pattern_combination(
        self,
        available_patterns: list[str],
        historical_performance: MetricsMap,
        market_conditions: PayloadMap,
    ) -> dict[str, float]:
        """Optimize pattern combination weights using ML."""

    @abstractmethod
    def update_model(self, new_data: MLTrainingData) -> bool:
        """Update the optimization model with new training data."""

    @abstractmethod
    def get_optimization_metrics(self) -> MetricsMap:
        """Get current optimization model metrics."""

class IOnlineLearner(IActionSignalGuideInterface):
    """Interface for online learning and parameter adaptation."""

    @abstractmethod
    def learn_from_feedback(
        self,
        prediction: MLPredictionResult,
        actual_outcome: float,
        market_context: PayloadMap,
    ) -> None:
        """Learn from prediction feedback for online adaptation."""

    @abstractmethod
    def adapt_parameters(self, current_performance: MetricsMap) -> PayloadMap:
        """Adapt model parameters based on current performance."""

    @abstractmethod
    def get_learning_progress(self) -> MetricsMap:
        """Get online learning progress metrics."""

class IEnsemblePredictor(IActionSignalGuideInterface):
    """Interface for ensemble prediction combining multiple ML models."""

    @abstractmethod
    def predict_ensemble(
        self, features: pd.DataFrame, model_predictions: list[MLPredictionResult]
    ) -> MLPredictionResult:
        """Generate ensemble prediction from multiple model predictions."""

    @abstractmethod
    def update_ensemble_weights(self, model_performance: dict[str, float]) -> None:
        """Update ensemble model weights based on individual model performance."""

    @abstractmethod
    def get_ensemble_statistics(self) -> MetricsMap:
        """Get ensemble model statistics."""

class IFeatureEngineer(IActionSignalGuideInterface):
    """Interface for feature engineering and selection."""

    @abstractmethod
    def engineer_features(
        self, raw_data: pd.DataFrame, pattern_signals: PayloadMap
    ) -> pd.DataFrame:
        """Engineer features from raw market data and pattern signals."""

    @abstractmethod
    def select_features(
        self, feature_matrix: pd.DataFrame, target: pd.Series, max_features: int = 50
    ) -> list[str]:
        """Select most important features using statistical methods."""

    @abstractmethod
    def get_feature_importance(self) -> dict[str, float]:
        """Get feature importance scores."""

