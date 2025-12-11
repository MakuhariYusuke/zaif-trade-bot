"""
Machine Learning Integration Interfaces for Action Signal Guide.

This module defines interfaces for ML-based pattern optimization and ensemble prediction.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional

import pandas as pd


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

    features: Any  # Can be DataFrame, ndarray, or dict
    target: Any  # Target values
    feature_names: Optional[List[str]] = None
    target_name: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None


@dataclass
class MLPredictionResult:
    """Result of ML model prediction."""

    prediction: float
    confidence: float
    feature_importance: Dict[str, float]
    model_name: str
    timestamp: float


@dataclass
class MLResult:
    """Generic result structure for ML operations."""

    success: bool
    data: Any
    message: str
    metadata: Dict[str, Any]


class IPatternOptimizer(ABC):
    """Interface for pattern combination optimization using ML."""

    @abstractmethod
    def optimize_pattern_combination(
        self,
        available_patterns: List[str],
        historical_performance: Dict[str, Any],
        market_conditions: Dict[str, Any],
    ) -> Dict[str, float]:
        """
        Optimize pattern combination weights using ML.

        Args:
            available_patterns: List of available pattern names
            historical_performance: Historical performance data
            market_conditions: Current market conditions

        Returns:
            Dictionary mapping pattern names to optimized weights
        """
        pass

    @abstractmethod
    def update_model(self, new_data: MLTrainingData) -> bool:
        """
        Update the optimization model with new training data.

        Args:
            new_data: New training data

        Returns:
            True if update was successful
        """
        pass

    @abstractmethod
    def get_optimization_metrics(self) -> Dict[str, Any]:
        """Get current optimization model metrics."""
        pass


class IOnlineLearner(ABC):
    """Interface for online learning and parameter adaptation."""

    @abstractmethod
    def learn_from_feedback(
        self,
        prediction: MLPredictionResult,
        actual_outcome: float,
        market_context: Dict[str, Any],
    ) -> None:
        """
        Learn from prediction feedback for online adaptation.

        Args:
            prediction: Model prediction result
            actual_outcome: Actual market outcome
            market_context: Market context at prediction time
        """
        pass

    @abstractmethod
    def adapt_parameters(self, current_performance: Dict[str, Any]) -> Dict[str, Any]:
        """
        Adapt model parameters based on current performance.

        Args:
            current_performance: Current model performance metrics

        Returns:
            Updated parameters
        """
        pass

    @abstractmethod
    def get_learning_progress(self) -> Dict[str, Any]:
        """Get online learning progress metrics."""
        pass


class IEnsemblePredictor(ABC):
    """Interface for ensemble prediction combining multiple ML models."""

    @abstractmethod
    def predict_ensemble(
        self, features: pd.DataFrame, model_predictions: List[MLPredictionResult]
    ) -> MLPredictionResult:
        """
        Generate ensemble prediction from multiple model predictions.

        Args:
            features: Input features
            model_predictions: List of individual model predictions

        Returns:
            Ensemble prediction result
        """
        pass

    @abstractmethod
    def update_ensemble_weights(self, model_performance: Dict[str, float]) -> None:
        """
        Update ensemble model weights based on individual model performance.

        Args:
            model_performance: Performance metrics for each model
        """
        pass

    @abstractmethod
    def get_ensemble_statistics(self) -> Dict[str, Any]:
        """Get ensemble model statistics."""
        pass


class IFeatureEngineer(ABC):
    """Interface for feature engineering and selection."""

    @abstractmethod
    def engineer_features(
        self, raw_data: pd.DataFrame, pattern_signals: Dict[str, Any]
    ) -> pd.DataFrame:
        """
        Engineer features from raw market data and pattern signals.

        Args:
            raw_data: Raw OHLCV market data
            pattern_signals: Pattern recognition signals

        Returns:
            Engineered features DataFrame
        """
        pass

    @abstractmethod
    def select_features(
        self, feature_matrix: pd.DataFrame, target: pd.Series, max_features: int = 50
    ) -> List[str]:
        """
        Select most important features using statistical methods.

        Args:
            feature_matrix: Feature matrix
            target: Target variable
            max_features: Maximum number of features to select

        Returns:
            List of selected feature names
        """
        pass

    @abstractmethod
    def get_feature_importance(self) -> Dict[str, float]:
        """Get feature importance scores."""
        pass
