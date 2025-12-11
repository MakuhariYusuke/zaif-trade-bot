"""
Pattern Optimizer Implementation for Action Signal Guide.

This module implements ML-based pattern optimization using various algorithms
to improve signal recognition and prediction accuracy.
"""

import logging
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.preprocessing import StandardScaler

from ..config.asg_ml_config import MLIntegrationConfig, PatternOptimizerConfig
from ..interfaces.ml_interfaces import (
    IPatternOptimizer,
    MLPredictionModel,
    MLResult,
    MLTrainingData,
)

logger = logging.getLogger(__name__)


@dataclass
class PatternOptimizationResult:
    """Result of pattern optimization."""

    optimized_patterns: Dict[str, Any] = field(default_factory=dict)
    performance_metrics: Dict[str, float] = field(default_factory=dict)
    feature_importance: Dict[str, float] = field(default_factory=dict)
    model_accuracy: float = 0.0
    optimization_score: float = 0.0
    training_time: float = 0.0
    validation_score: float = 0.0


class BasePatternOptimizer(IPatternOptimizer):
    """Base implementation of pattern optimizer."""

    def __init__(self, config: PatternOptimizerConfig):
        self.config = config
        self.models: Dict[str, Any] = {}
        self.scalers: Dict[str, StandardScaler] = {}
        self.feature_importance: Dict[str, Dict[str, float]] = {}
        self.is_trained = False

    def optimize_patterns(self, training_data: MLTrainingData) -> MLResult:
        """Optimize patterns using ML algorithms."""
        start_time = time.time()

        try:
            # Prepare data
            X, y = self._prepare_data(training_data)

            # Train models
            results = {}
            for model_type in self.config.model_types:
                model_result = self._train_model(model_type, X, y, training_data)
                results[model_type.value] = model_result

            # Select best model
            best_model_type, best_result = self._select_best_model(results)

            # Create optimization result
            optimization_result = PatternOptimizationResult(
                optimized_patterns=best_result.get("patterns", {}),
                performance_metrics=best_result.get("metrics", {}),
                feature_importance=best_result.get("importance", {}),
                model_accuracy=best_result.get("accuracy", 0.0),
                optimization_score=best_result.get("score", 0.0),
                training_time=time.time() - start_time,
                validation_score=best_result.get("validation", 0.0),
            )

            self.is_trained = True

            return MLResult(
                success=True,
                data=optimization_result,
                message=f"Pattern optimization completed with {best_model_type}",
                metadata={
                    "best_model": best_model_type,
                    "training_time": optimization_result.training_time,
                    "model_count": len(results),
                },
            )

        except Exception as e:
            logger.error(f"Pattern optimization failed: {e}")
            return MLResult(
                success=False,
                data=None,
                message=f"Pattern optimization failed: {str(e)}",
                metadata={"error": str(e)},
            )

    def predict_patterns(self, features: Dict[str, Any]) -> MLResult:
        """Predict optimized patterns for new data."""
        if not self.is_trained:
            return MLResult(
                success=False, data=None, message="Model not trained yet", metadata={}
            )

        try:
            # Prepare features
            X = self._prepare_features(features)

            # Get predictions from all models
            predictions = {}
            for model_name, model in self.models.items():
                pred = model.predict(X)
                predictions[model_name] = pred[0] if len(pred) == 1 else pred

            # Ensemble prediction
            final_prediction = self._ensemble_predictions(predictions)

            return MLResult(
                success=True,
                data=final_prediction,
                message="Pattern prediction completed",
                metadata={"model_predictions": predictions},
            )

        except Exception as e:
            logger.error(f"Pattern prediction failed: {e}")
            return MLResult(
                success=False,
                data=None,
                message=f"Pattern prediction failed: {str(e)}",
                metadata={"error": str(e)},
            )

    def update_model(self, new_data: MLTrainingData) -> MLResult:
        """Update model with new training data."""
        if not self.is_trained:
            return self.optimize_patterns(new_data)

        try:
            # Incremental learning or retraining
            X, y = self._prepare_data(new_data)

            for model_name, model in self.models.items():
                if hasattr(model, "partial_fit"):
                    model.partial_fit(X, y)
                else:
                    # Retrain with combined data
                    model.fit(X, y)

            return MLResult(
                success=True,
                data=None,
                message="Model updated successfully",
                metadata={},
            )

        except Exception as e:
            logger.error(f"Model update failed: {e}")
            return MLResult(
                success=False,
                data=None,
                message=f"Model update failed: {str(e)}",
                metadata={"error": str(e)},
            )

    def get_model_info(self) -> Dict[str, Any]:
        """Get information about trained models."""
        return {
            "is_trained": self.is_trained,
            "models": list(self.models.keys()),
            "feature_importance": self.feature_importance,
            "config": self.config.__dict__,
        }

    def _prepare_data(
        self, training_data: MLTrainingData
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Prepare training data for ML models."""
        # Convert to DataFrame if needed
        if isinstance(training_data.features, dict):
            df = pd.DataFrame(training_data.features)
        else:
            df = training_data.features

        # Extract target
        if isinstance(training_data.target, str):
            y = df[training_data.target].values
            X = df.drop(columns=[training_data.target]).values
        else:
            y = np.array(training_data.target)
            X = df.values if isinstance(df, pd.DataFrame) else np.array(df)

        # Scale features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        # Store scaler
        self.scalers["main"] = scaler

        return X_scaled, y

    def _prepare_features(self, features: Dict[str, Any]) -> np.ndarray:
        """Prepare features for prediction."""
        df = pd.DataFrame([features])
        X = df.values

        # Scale using stored scaler
        if "main" in self.scalers:
            X = self.scalers["main"].transform(X)

        return X

    def _train_model(
        self,
        model_type: MLPredictionModel,
        X: np.ndarray,
        y: np.ndarray,
        training_data: MLTrainingData,
    ) -> Dict[str, Any]:
        """Train a specific ML model."""
        try:
            # Create model
            model = self._create_model(model_type)

            # Cross-validation
            tscv = TimeSeriesSplit(n_splits=5)
            cv_scores = cross_val_score(
                model, X, y, cv=tscv, scoring="neg_mean_squared_error"
            )
            cv_score = -cv_scores.mean()

            # Train final model
            model.fit(X, y)

            # Store model
            self.models[model_type.value] = model

            # Get feature importance if available
            importance = {}
            if hasattr(model, "feature_importances_"):
                feature_names = (
                    training_data.feature_names
                    if training_data.feature_names
                    else [f"feature_{i}" for i in range(X.shape[1])]
                )
                importance = dict(zip(feature_names, model.feature_importances_))
            elif hasattr(model, "coef_"):
                feature_names = (
                    training_data.feature_names
                    if training_data.feature_names
                    else [f"feature_{i}" for i in range(X.shape[1])]
                )
                importance = dict(zip(feature_names, np.abs(model.coef_)))

            self.feature_importance[model_type.value] = importance

            # Calculate metrics
            y_pred = model.predict(X)
            mse = mean_squared_error(y, y_pred)
            r2 = r2_score(y, y_pred)

            return {
                "model": model,
                "cv_score": cv_score,
                "mse": mse,
                "r2": r2,
                "accuracy": r2,  # Using R2 as accuracy metric
                "score": r2 - cv_score,  # Combined score
                "importance": importance,
                "patterns": {},  # Will be filled by subclasses
                "metrics": {"mse": mse, "r2": r2, "cv_score": cv_score},
                "validation": cv_score,
            }

        except Exception as e:
            logger.error(f"Failed to train {model_type.value}: {e}")
            return {
                "error": str(e),
                "accuracy": 0.0,
                "score": float("-inf"),
                "importance": {},
                "patterns": {},
                "metrics": {},
                "validation": 0.0,
            }

    def _create_model(self, model_type: MLPredictionModel) -> Any:
        """Create ML model instance."""
        if model_type == MLPredictionModel.LINEAR_REGRESSION:
            return LinearRegression()
        elif model_type == MLPredictionModel.RANDOM_FOREST:
            return RandomForestRegressor(
                n_estimators=self.config.random_forest_estimators,
                max_depth=self.config.random_forest_max_depth,
                random_state=42,
            )
        elif model_type == MLPredictionModel.GRADIENT_BOOSTING:
            return GradientBoostingRegressor(
                n_estimators=self.config.gradient_boosting_estimators,
                learning_rate=self.config.gradient_boosting_learning_rate,
                max_depth=self.config.gradient_boosting_max_depth,
                random_state=42,
            )
        else:
            raise ValueError(f"Unsupported model type: {model_type}")

    def _select_best_model(
        self, results: Dict[str, Dict[str, Any]]
    ) -> Tuple[str, Dict[str, Any]]:
        """Select the best performing model."""
        best_model = None
        best_score = float("-inf")
        best_result = None

        for model_name, result in results.items():
            if (
                "error" not in result
                and result.get("score", float("-inf")) > best_score
            ):
                best_score = result["score"]
                best_model = model_name
                best_result = result

        if best_model is None:
            # Fallback to first model
            best_model = list(results.keys())[0]
            best_result = results[best_model]

        return best_model, best_result

    def _ensemble_predictions(
        self, predictions: Dict[str, np.ndarray]
    ) -> Dict[str, Any]:
        """Create ensemble prediction from multiple models."""
        if not predictions:
            return {}

        # Simple averaging for regression
        pred_values = list(predictions.values())
        ensemble_pred = np.mean(pred_values, axis=0)

        return {
            "ensemble_prediction": float(ensemble_pred),
            "individual_predictions": {k: float(v) for k, v in predictions.items()},
            "prediction_variance": float(np.var(pred_values)),
        }


class AdvancedPatternOptimizer(BasePatternOptimizer):
    """Advanced pattern optimizer with additional features."""

    def __init__(self, config: PatternOptimizerConfig):
        super().__init__(config)
        self.pattern_library: Dict[str, Dict[str, Any]] = {}
        self.performance_history: List[Dict[str, Any]] = []

    def optimize_patterns(self, training_data: MLTrainingData) -> MLResult:
        """Enhanced pattern optimization with pattern discovery."""
        base_result = super().optimize_patterns(training_data)

        if not base_result.success:
            return base_result

        try:
            # Extract patterns from trained models
            patterns = self._extract_patterns(training_data)

            # Update pattern library
            self.pattern_library.update(patterns)

            # Update result with patterns
            if isinstance(base_result.data, PatternOptimizationResult):
                base_result.data.optimized_patterns = patterns

            return base_result

        except Exception as e:
            logger.error(f"Pattern extraction failed: {e}")
            return base_result

    def _extract_patterns(self, training_data: MLTrainingData) -> Dict[str, Any]:
        """Extract meaningful patterns from training data and models."""
        patterns = {}

        # Analyze feature correlations
        if isinstance(training_data.features, pd.DataFrame):
            corr_matrix = training_data.features.corr()
            strong_correlations = self._find_strong_correlations(corr_matrix)
            patterns["correlations"] = strong_correlations

        # Extract decision rules from tree-based models
        for model_name, model in self.models.items():
            if hasattr(model, "estimators_"):  # Ensemble model
                rules = self._extract_decision_rules(model)
                patterns[f"{model_name}_rules"] = rules

        # Identify key features
        key_features = self._identify_key_features()
        patterns["key_features"] = key_features

        return patterns

    def _find_strong_correlations(
        self, corr_matrix: pd.DataFrame, threshold: float = 0.7
    ) -> Dict[str, List[str]]:
        """Find strongly correlated feature groups."""
        correlations = {}
        for col in corr_matrix.columns:
            correlated = corr_matrix[col][
                abs(corr_matrix[col]) > threshold
            ].index.tolist()
            correlated.remove(col)  # Remove self-correlation
            if correlated:
                correlations[col] = correlated
        return correlations

    def _extract_decision_rules(self, model: Any, max_rules: int = 10) -> List[str]:
        """Extract decision rules from tree-based models."""
        rules = []
        try:
            # This is a simplified implementation
            # In practice, you'd use libraries like sklearn.tree.export_text
            if hasattr(model, "estimators_"):
                for estimator in model.estimators_[:max_rules]:
                    # Simplified rule extraction
                    rule = f"Tree {len(rules) + 1}: feature_importance = {estimator.feature_importances_[:3]}"
                    rules.append(rule)
        except Exception as e:
            logger.warning(f"Rule extraction failed: {e}")

        return rules

    def _identify_key_features(self) -> List[str]:
        """Identify most important features across all models."""
        all_importance = {}
        for model_name, importance in self.feature_importance.items():
            for feature, imp in importance.items():
                if feature not in all_importance:
                    all_importance[feature] = []
                all_importance[feature].append(imp)

        # Average importance across models
        avg_importance = {k: np.mean(v) for k, v in all_importance.items()}

        # Sort by importance
        sorted_features = sorted(
            avg_importance.items(), key=lambda x: x[1], reverse=True
        )

        return [f for f, _ in sorted_features[:10]]  # Top 10 features

    def optimize_pattern_combination(
        self,
        available_patterns: List[str],
        historical_performance: Dict[str, Any],
        market_conditions: Dict[str, Any],
    ) -> Dict[str, float]:
        """Optimize pattern combination weights using advanced ML techniques."""
        if not available_patterns:
            return {}

        try:
            # Use ML models to predict optimal weights
            features = self._create_combination_features(
                available_patterns, historical_performance, market_conditions
            )

            weights = {}
            for pattern in available_patterns:
                # Predict weight using ensemble of models
                pattern_features = features.get(pattern, {})
                weight = self._predict_pattern_weight(pattern, pattern_features)
                weights[pattern] = max(0.0, min(1.0, weight))  # Clamp to [0, 1]

            # Normalize weights
            total_weight = sum(weights.values())
            if total_weight > 0:
                weights = {k: v / total_weight for k, v in weights.items()}

            return weights

        except Exception as e:
            logger.error(f"Pattern combination optimization failed: {e}")
            # Fallback to equal weights
            return {
                pattern: 1.0 / len(available_patterns) for pattern in available_patterns
            }

    def get_optimization_metrics(self) -> Dict[str, Any]:
        """Get comprehensive optimization metrics."""
        metrics = {
            "total_patterns": len(self.pattern_library),
            "performance_history_length": len(self.performance_history),
            "feature_importance_summary": self._summarize_feature_importance(),
            "model_performance": self._get_model_performance_metrics(),
            "pattern_discovery_stats": self._get_pattern_discovery_stats(),
        }

        return metrics

    def _create_combination_features(
        self,
        available_patterns: List[str],
        historical_performance: Dict[str, Any],
        market_conditions: Dict[str, Any],
    ) -> Dict[str, Dict[str, float]]:
        """Create features for pattern combination optimization."""
        features = {}

        for pattern in available_patterns:
            pattern_features = {}

            # Historical performance features
            perf_data = historical_performance.get(pattern, {})
            pattern_features.update(
                {
                    "avg_return": perf_data.get("avg_return", 0.0),
                    "volatility": perf_data.get("volatility", 0.0),
                    "sharpe_ratio": perf_data.get("sharpe_ratio", 0.0),
                    "win_rate": perf_data.get("win_rate", 0.0),
                    "max_drawdown": perf_data.get("max_drawdown", 0.0),
                }
            )

            # Market condition features
            pattern_features.update(
                {
                    "market_volatility": market_conditions.get("volatility", 0.0),
                    "trend_strength": market_conditions.get("trend_strength", 0.0),
                    "liquidity": market_conditions.get("liquidity", 1.0),
                }
            )

            features[pattern] = pattern_features

        return features

    def _predict_pattern_weight(
        self, pattern: str, features: Dict[str, float]
    ) -> float:
        """Predict optimal weight for a pattern using ML models."""
        try:
            # Simple weighted combination for now
            # In practice, this would use trained ML models
            weight = 0.0

            # Base weight from performance
            weight += features.get("sharpe_ratio", 0.0) * 0.4
            weight += features.get("win_rate", 0.0) * 0.3
            weight += (1.0 - abs(features.get("max_drawdown", 0.0))) * 0.3

            # Adjust for market conditions
            volatility_penalty = features.get("market_volatility", 0.0) * 0.1
            weight = max(0.0, weight - volatility_penalty)

            return weight

        except Exception as e:
            logger.warning(f"Weight prediction failed for {pattern}: {e}")
            return 0.1  # Default weight

    def _summarize_feature_importance(self) -> Dict[str, float]:
        """Summarize feature importance across all models."""
        if not self.feature_importance:
            return {}

        all_importance = {}
        for model_name, importance in self.feature_importance.items():
            for feature, imp in importance.items():
                if feature not in all_importance:
                    all_importance[feature] = []
                all_importance[feature].append(imp)

        # Average importance
        avg_importance = {k: np.mean(v) for k, v in all_importance.items()}

        # Sort by importance
        sorted_importance = sorted(
            avg_importance.items(), key=lambda x: x[1], reverse=True
        )

        return dict(sorted_importance[:10])  # Top 10

    def _get_model_performance_metrics(self) -> Dict[str, Any]:
        """Get performance metrics for all models."""
        metrics = {}
        for model_name, model in self.models.items():
            try:
                # Basic metrics - in practice, you'd track more detailed metrics
                metrics[model_name] = {
                    "trained": True,
                    "feature_count": len(self.feature_importance.get(model_name, {})),
                }
            except Exception as e:
                metrics[model_name] = {"error": str(e)}

        return metrics

    def _get_pattern_discovery_stats(self) -> Dict[str, Any]:
        """Get statistics about pattern discovery."""
        return {
            "patterns_discovered": len(self.pattern_library),
            "correlation_patterns": len(
                [p for p in self.pattern_library.keys() if "correlation" in p]
            ),
            "rule_patterns": len(
                [p for p in self.pattern_library.keys() if "rule" in p]
            ),
            "feature_patterns": len(
                [p for p in self.pattern_library.keys() if "feature" in p]
            ),
        }


def create_pattern_optimizer(config: MLIntegrationConfig) -> IPatternOptimizer:
    """Factory function to create pattern optimizer."""
    if config.pattern_optimizer.advanced_features:
        return AdvancedPatternOptimizer(config.pattern_optimizer)
    else:
        return BasePatternOptimizer(config.pattern_optimizer)
