"""
Pattern Optimizer Implementation for Action Signal Guide.

This module implements ML-based pattern optimization using various algorithms
to improve signal recognition and prediction accuracy.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import TypedDict, cast

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.preprocessing import StandardScaler

from ..config.asg_ml_config import MLIntegrationConfig, PatternOptimizerConfig
from ..components.history_helpers import append_with_compaction
from ..interfaces.ml_interfaces import (
    IPatternOptimizer,
    MLPredictionModel,
    MLResult,
    MLTrainingData,
)

logger = logging.getLogger(__name__)

RegressorModel = LinearRegression | RandomForestRegressor | GradientBoostingRegressor

class ModelTrainingSuccess(TypedDict):
    model: RegressorModel
    cv_score: float
    mse: float
    r2: float
    accuracy: float
    score: float
    importance: dict[str, float]
    patterns: dict[str, object]
    metrics: dict[str, float]
    validation: float

class ModelTrainingError(TypedDict):
    error: str
    accuracy: float
    score: float
    importance: dict[str, float]
    patterns: dict[str, object]
    metrics: dict[str, float]
    validation: float

ModelTrainingResult = ModelTrainingSuccess | ModelTrainingError

@dataclass
class PatternOptimizationResult:
    """Result of pattern optimization."""

    optimized_patterns: dict[str, object] = field(default_factory=dict)
    performance_metrics: dict[str, float] = field(default_factory=dict)
    feature_importance: dict[str, float] = field(default_factory=dict)
    model_accuracy: float = 0.0
    optimization_score: float = 0.0
    training_time: float = 0.0
    validation_score: float = 0.0

class BasePatternOptimizer(IPatternOptimizer):
    """Base implementation of pattern optimizer."""

    def __init__(self, config: PatternOptimizerConfig):
        self.config = config
        self.models: dict[str, RegressorModel] = {}
        self.scalers: dict[str, StandardScaler] = {}
        self.feature_importance: dict[str, dict[str, float]] = {}
        self.is_trained = False

        # Keep feature ordering stable between training and inference.
        self._feature_names: list[str] = []
        self._last_training_results: dict[str, ModelTrainingResult] = {}

    def optimize_patterns(self, training_data: MLTrainingData) -> MLResult:
        """Optimize patterns using ML algorithms."""
        start_time = time.time()

        try:
            X, y = self._prepare_data(training_data)

            results: dict[str, ModelTrainingResult] = {}
            for model_type in self.config.model_types:
                model_result = self._train_model(model_type, X, y, training_data)
                results[model_type.value] = model_result

            successful_models = [
                name for name, result in results.items() if not self._is_error_result(result)
            ]
            if not successful_models:
                self.is_trained = False
                self._last_training_results = results
                return MLResult(
                    success=False,
                    data=None,
                    message="Pattern optimization failed: no model trained successfully",
                    metadata={
                        "model_errors": {
                            name: cast(ModelTrainingError, result).get("error", "unknown")
                            for name, result in results.items()
                        }
                    },
                )

            best_model_type, best_result = self._select_best_model(results)

            optimization_result = PatternOptimizationResult(
                optimized_patterns=dict(best_result.get("patterns", {})),
                performance_metrics=dict(best_result.get("metrics", {})),
                feature_importance=dict(best_result.get("importance", {})),
                model_accuracy=float(best_result.get("accuracy", 0.0)),
                optimization_score=float(best_result.get("score", 0.0)),
                training_time=time.time() - start_time,
                validation_score=float(best_result.get("validation", 0.0)),
            )

            self.is_trained = True
            self._last_training_results = results

            return MLResult(
                success=True,
                data=optimization_result,
                message=f"Pattern optimization completed with {best_model_type}",
                metadata={
                    "best_model": best_model_type,
                    "training_time": optimization_result.training_time,
                    "model_count": len(results),
                    "successful_models": len(successful_models),
                },
            )

        except Exception as e:
            logger.error(f"Pattern optimization failed: {e}")
            self.is_trained = False
            return MLResult(
                success=False,
                data=None,
                message=f"Pattern optimization failed: {str(e)}",
                metadata={"error": str(e)},
            )

    def predict_patterns(self, features: dict[str, object]) -> MLResult:
        """Predict optimized patterns for new data."""
        if not self.is_trained or not self.models:
            return MLResult(
                success=False, data=None, message="Model not trained yet", metadata={}
            )

        try:
            X = self._prepare_features(features)

            predictions: dict[str, float] = {}
            for model_name, model in self.models.items():
                pred = np.asarray(model.predict(X), dtype=float).reshape(-1)
                if pred.size == 0:
                    continue
                predictions[model_name] = float(pred[0])

            if not predictions:
                return MLResult(
                    success=False,
                    data=None,
                    message="Pattern prediction failed: no model produced a prediction",
                    metadata={},
                )

            final_prediction = self._ensemble_predictions(predictions)

            return MLResult(
                success=True,
                data=final_prediction,
                message="Pattern prediction completed",
                metadata={"model_predictions": dict(predictions)},
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
            X, y = self._prepare_data(new_data)

            for model in self.models.values():
                if hasattr(model, "partial_fit"):
                    model.partial_fit(X, y)  # type: ignore[attr-defined]
                else:
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

    def optimize_pattern_combination(
        self,
        available_patterns: list[str],
        historical_performance: dict[str, object],
        market_conditions: dict[str, object],
    ) -> dict[str, float]:
        """Provide a safe baseline optimizer for pattern combination weights."""
        if not available_patterns:
            return {}

        weights: dict[str, float] = {}
        volatility_penalty = max(0.0, self._to_float(market_conditions.get("volatility"), 0.0))

        for pattern in available_patterns:
            perf_raw = historical_performance.get(pattern, {})
            perf = perf_raw if isinstance(perf_raw, Mapping) else {}

            sharpe = self._to_float(perf.get("sharpe_ratio"), 0.0)
            win_rate = self._to_float(perf.get("win_rate"), 0.0)
            drawdown = abs(self._to_float(perf.get("max_drawdown"), 0.0))

            score = sharpe * 0.5 + win_rate * 0.4 + (1.0 - drawdown) * 0.1
            score = max(0.0, score - volatility_penalty * 0.05)
            weights[pattern] = score

        total = sum(weights.values())
        if total <= 0:
            equal_weight = 1.0 / len(available_patterns)
            return {pattern: equal_weight for pattern in available_patterns}

        return {pattern: value / total for pattern, value in weights.items()}

    def get_optimization_metrics(self) -> dict[str, object]:
        """Get optimization metrics for the current base optimizer state."""
        training_scores: dict[str, float] = {}
        for model_name, result in self._last_training_results.items():
            if not self._is_error_result(result):
                training_scores[model_name] = float(result.get("score", 0.0))

        return {
            "is_trained": self.is_trained,
            "model_count": len(self.models),
            "feature_importance_models": len(self.feature_importance),
            "training_scores": training_scores,
        }

    def get_model_info(self) -> dict[str, object]:
        """Get information about trained models."""
        return {
            "is_trained": self.is_trained,
            "models": list(self.models.keys()),
            "feature_importance": self.feature_importance,
            "config": self.config.__dict__,
            "feature_names": list(self._feature_names),
        }

    def _prepare_data(
        self, training_data: MLTrainingData
    ) -> tuple[np.ndarray, np.ndarray]:
        """Prepare training data for ML models."""
        df = self._coerce_features_to_dataframe(training_data.features)

        if training_data.feature_names and len(training_data.feature_names) == df.shape[1]:
            df.columns = [str(name) for name in training_data.feature_names]
        else:
            df.columns = [str(col) for col in df.columns]

        if isinstance(training_data.target, str):
            target_col = training_data.target
            if target_col not in df.columns:
                raise ValueError(f"Target column '{target_col}' not found in training data")
            y = df[target_col].to_numpy(dtype=float)
            X_df = df.drop(columns=[target_col])
        else:
            y = np.asarray(training_data.target, dtype=float).reshape(-1)
            X_df = df

        X = X_df.to_numpy(dtype=float, copy=False)
        if X.shape[0] != y.shape[0]:
            raise ValueError(
                f"Feature/target length mismatch: X={X.shape[0]}, y={y.shape[0]}"
            )
        if X.shape[0] < 3:
            raise ValueError("At least 3 samples are required for pattern optimization")

        self._feature_names = [str(col) for col in X_df.columns]

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        self.scalers["main"] = scaler

        return X_scaled, y

    def _coerce_features_to_dataframe(self, features: object) -> pd.DataFrame:
        """Normalize training features into a DataFrame."""
        if isinstance(features, pd.DataFrame):
            return features.copy()

        if isinstance(features, Mapping):
            if not features:
                return pd.DataFrame()
            if all(np.isscalar(v) for v in features.values()):
                return pd.DataFrame([dict(features)])
            return pd.DataFrame(features)

        return pd.DataFrame(features)

    def _prepare_features(self, features: dict[str, object]) -> np.ndarray:
        """Prepare features for prediction using the training feature order."""
        if self._feature_names:
            row = [self._to_float(features.get(name), 0.0) for name in self._feature_names]
        else:
            ordered_keys = sorted(features.keys())
            row = [self._to_float(features[key], 0.0) for key in ordered_keys]

        X = np.asarray([row], dtype=float)

        scaler = self.scalers.get("main")
        if scaler is not None:
            expected_features = getattr(scaler, "n_features_in_", X.shape[1])
            if int(expected_features) != X.shape[1]:
                raise ValueError(
                    f"Feature size mismatch for prediction: expected {expected_features}, got {X.shape[1]}"
                )
            X = scaler.transform(X)

        return X

    def _train_model(
        self,
        model_type: MLPredictionModel,
        X: np.ndarray,
        y: np.ndarray,
        training_data: MLTrainingData,
    ) -> ModelTrainingResult:
        """Train a specific ML model."""
        try:
            model = self._create_model(model_type)
            cv_score = self._compute_cv_score(model, X, y)

            model.fit(X, y)
            self.models[model_type.value] = model

            importance = self._extract_feature_importance(model, training_data, X.shape[1])
            self.feature_importance[model_type.value] = importance

            y_pred = np.asarray(model.predict(X), dtype=float).reshape(-1)
            mse = float(mean_squared_error(y, y_pred))
            r2 = float(r2_score(y, y_pred))

            validation_score = cv_score if np.isfinite(cv_score) else mse
            score = r2 - validation_score

            return {
                "model": model,
                "cv_score": validation_score,
                "mse": mse,
                "r2": r2,
                "accuracy": r2,
                "score": score,
                "importance": importance,
                "patterns": {},
                "metrics": {
                    "mse": mse,
                    "r2": r2,
                    "cv_score": validation_score,
                },
                "validation": validation_score,
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

    def _compute_cv_score(
        self, model: RegressorModel, X: np.ndarray, y: np.ndarray
    ) -> float:
        """Compute CV score with sample-aware split sizing."""
        sample_count = X.shape[0]
        n_splits = min(5, sample_count - 1)
        if n_splits < 2:
            return float("nan")

        try:
            tscv = TimeSeriesSplit(n_splits=n_splits)
            cv_scores = cross_val_score(
                model,
                X,
                y,
                cv=tscv,
                scoring="neg_mean_squared_error",
            )
            return float(-np.mean(cv_scores))
        except Exception as e:
            logger.warning(f"Cross-validation failed for model {type(model).__name__}: {e}")
            return float("nan")

    def _extract_feature_importance(
        self,
        model: RegressorModel,
        training_data: MLTrainingData,
        feature_count: int,
    ) -> dict[str, float]:
        """Extract feature importance from model if available."""
        feature_names = self._resolve_feature_names(training_data, feature_count)

        if hasattr(model, "feature_importances_"):
            raw = np.asarray(getattr(model, "feature_importances_"), dtype=float)
            return {
                name: float(value)
                for name, value in zip(feature_names, raw, strict=False)
            }

        if hasattr(model, "coef_"):
            raw = np.abs(np.asarray(getattr(model, "coef_"), dtype=float).reshape(-1))
            return {
                name: float(value)
                for name, value in zip(feature_names, raw, strict=False)
            }

        return {}

    def _resolve_feature_names(
        self, training_data: MLTrainingData, feature_count: int
    ) -> list[str]:
        """Resolve feature names for reporting/importances."""
        if training_data.feature_names and len(training_data.feature_names) == feature_count:
            return [str(name) for name in training_data.feature_names]

        if self._feature_names and len(self._feature_names) == feature_count:
            return list(self._feature_names)

        return [f"feature_{i}" for i in range(feature_count)]

    def _create_model(self, model_type: MLPredictionModel) -> RegressorModel:
        """Create ML model instance."""
        if model_type == MLPredictionModel.LINEAR_REGRESSION:
            return LinearRegression()
        if model_type == MLPredictionModel.RANDOM_FOREST:
            return RandomForestRegressor(
                n_estimators=self.config.random_forest_estimators,
                max_depth=self.config.random_forest_max_depth,
                random_state=42,
            )
        if model_type == MLPredictionModel.GRADIENT_BOOSTING:
            return GradientBoostingRegressor(
                n_estimators=self.config.gradient_boosting_estimators,
                learning_rate=self.config.gradient_boosting_learning_rate,
                max_depth=self.config.gradient_boosting_max_depth,
                random_state=42,
            )
        raise ValueError(f"Unsupported model type: {model_type}")

    def _select_best_model(
        self, results: dict[str, ModelTrainingResult]
    ) -> tuple[str, ModelTrainingSuccess]:
        """Select the best performing model."""
        best_model = ""
        best_score = float("-inf")
        best_result: ModelTrainingSuccess | None = None

        for model_name, result in results.items():
            if self._is_error_result(result):
                continue

            score = float(result.get("score", float("-inf")))
            if score > best_score:
                best_score = score
                best_model = model_name
                best_result = result

        if best_result is None:
            raise ValueError("No successful model results available")

        return best_model, best_result

    def _ensemble_predictions(self, predictions: dict[str, float]) -> dict[str, object]:
        """Create ensemble prediction from multiple models."""
        if not predictions:
            return {}

        pred_values = np.asarray(list(predictions.values()), dtype=float)
        ensemble_pred = float(np.mean(pred_values))

        return {
            "ensemble_prediction": ensemble_pred,
            "individual_predictions": dict(predictions),
            "prediction_variance": float(np.var(pred_values)),
        }

    @staticmethod
    def _is_error_result(result: ModelTrainingResult) -> bool:
        return "error" in result

    @staticmethod
    def _to_float(value: object, default: float = 0.0) -> float:
        try:
            if value is None:
                return default
            return float(value)
        except (TypeError, ValueError):
            return default

class AdvancedPatternOptimizer(BasePatternOptimizer):
    """Advanced pattern optimizer with additional features."""

    def __init__(self, config: PatternOptimizerConfig):
        super().__init__(config)
        self.pattern_library: dict[str, object] = {}
        self.performance_history: list[dict[str, float]] = []

    def optimize_patterns(self, training_data: MLTrainingData) -> MLResult:
        """Enhanced pattern optimization with pattern discovery."""
        base_result = super().optimize_patterns(training_data)

        if not base_result.success:
            return base_result

        try:
            if isinstance(base_result.data, PatternOptimizationResult):
                self._append_performance_snapshot(base_result.data)

            patterns = self._extract_patterns(training_data)
            self.pattern_library.update(patterns)

            if isinstance(base_result.data, PatternOptimizationResult):
                base_result.data.optimized_patterns = patterns

            return base_result

        except Exception as e:
            logger.error(f"Pattern extraction failed: {e}")
            return base_result

    def _append_performance_snapshot(self, result: PatternOptimizationResult) -> None:
        """Store bounded performance history for long-running processes."""
        history_limit = max(100, int(self.config.max_training_samples))
        append_with_compaction(
            self.performance_history,
            {
                "timestamp": float(time.time()),
                "model_accuracy": float(result.model_accuracy),
                "optimization_score": float(result.optimization_score),
                "validation_score": float(result.validation_score),
            },
            high_water=history_limit * 2,
            retain=history_limit,
        )

    def _extract_patterns(self, training_data: MLTrainingData) -> dict[str, object]:
        """Extract meaningful patterns from training data and models."""
        patterns: dict[str, object] = {}

        if isinstance(training_data.features, pd.DataFrame):
            corr_matrix = training_data.features.corr()
            strong_correlations = self._find_strong_correlations(corr_matrix)
            patterns["correlations"] = strong_correlations

        for model_name, model in self.models.items():
            if hasattr(model, "estimators_"):
                rules = self._extract_decision_rules(model)
                if rules:
                    patterns[f"{model_name}_rules"] = rules

        key_features = self._identify_key_features()
        patterns["key_features"] = key_features

        return patterns

    def _find_strong_correlations(
        self, corr_matrix: pd.DataFrame, threshold: float = 0.7
    ) -> dict[str, list[str]]:
        """Find strongly correlated feature groups."""
        correlations: dict[str, list[str]] = {}
        for col in corr_matrix.columns:
            correlated = corr_matrix[col][
                abs(corr_matrix[col]) > threshold
            ].index.tolist()
            if col in correlated:
                correlated.remove(col)
            if correlated:
                correlations[str(col)] = [str(item) for item in correlated]
        return correlations

    def _extract_decision_rules(
        self, model: RegressorModel, max_rules: int = 10
    ) -> list[str]:
        """Extract compact decision-rule summaries from tree-based models."""
        rules: list[str] = []
        try:
            estimators_obj = getattr(model, "estimators_", None)
            if estimators_obj is None:
                return rules

            if isinstance(estimators_obj, np.ndarray):
                estimators = [e for e in estimators_obj.ravel().tolist() if e is not None]
            elif isinstance(estimators_obj, list):
                estimators = list(estimators_obj)
            else:
                estimators = []

            for estimator in estimators[:max_rules]:
                importances = getattr(estimator, "feature_importances_", None)
                if importances is None:
                    continue

                top = np.asarray(importances, dtype=float).reshape(-1)[:3]
                top_summary = ", ".join(f"{value:.4f}" for value in top)
                rules.append(
                    f"Tree {len(rules) + 1}: feature_importance=[{top_summary}]"
                )

        except Exception as e:
            logger.warning(f"Rule extraction failed: {e}")

        return rules

    def _identify_key_features(self) -> list[str]:
        """Identify most important features across all models."""
        all_importance: dict[str, list[float]] = {}
        for importance in self.feature_importance.values():
            for feature, imp in importance.items():
                all_importance.setdefault(feature, []).append(float(imp))

        avg_importance = {k: float(np.mean(v)) for k, v in all_importance.items()}
        sorted_features = sorted(
            avg_importance.items(), key=lambda x: x[1], reverse=True
        )

        return [feature for feature, _ in sorted_features[:10]]

    def optimize_pattern_combination(
        self,
        available_patterns: list[str],
        historical_performance: dict[str, object],
        market_conditions: dict[str, object],
    ) -> dict[str, float]:
        """Optimize pattern combination weights using advanced feature synthesis."""
        if not available_patterns:
            return {}

        try:
            features = self._create_combination_features(
                available_patterns,
                historical_performance,
                market_conditions,
            )

            weights: dict[str, float] = {}
            for pattern in available_patterns:
                pattern_features = features.get(pattern, {})
                weight = self._predict_pattern_weight(pattern, pattern_features)
                weights[pattern] = max(0.0, min(1.0, weight))

            total_weight = sum(weights.values())
            if total_weight <= 0:
                equal = 1.0 / len(available_patterns)
                return {pattern: equal for pattern in available_patterns}

            return {k: v / total_weight for k, v in weights.items()}

        except Exception as e:
            logger.error(f"Pattern combination optimization failed: {e}")
            equal = 1.0 / len(available_patterns)
            return {pattern: equal for pattern in available_patterns}

    def get_optimization_metrics(self) -> dict[str, object]:
        """Get comprehensive optimization metrics."""
        return {
            "total_patterns": len(self.pattern_library),
            "performance_history_length": len(self.performance_history),
            "feature_importance_summary": self._summarize_feature_importance(),
            "model_performance": self._get_model_performance_metrics(),
            "pattern_discovery_stats": self._get_pattern_discovery_stats(),
        }

    def _create_combination_features(
        self,
        available_patterns: list[str],
        historical_performance: dict[str, object],
        market_conditions: dict[str, object],
    ) -> dict[str, dict[str, float]]:
        """Create features for pattern combination optimization."""
        features: dict[str, dict[str, float]] = {}

        for pattern in available_patterns:
            perf_raw = historical_performance.get(pattern, {})
            perf_data = perf_raw if isinstance(perf_raw, Mapping) else {}

            pattern_features = {
                "avg_return": self._to_float(perf_data.get("avg_return"), 0.0),
                "volatility": self._to_float(perf_data.get("volatility"), 0.0),
                "sharpe_ratio": self._to_float(perf_data.get("sharpe_ratio"), 0.0),
                "win_rate": self._to_float(perf_data.get("win_rate"), 0.0),
                "max_drawdown": self._to_float(perf_data.get("max_drawdown"), 0.0),
                "market_volatility": self._to_float(
                    market_conditions.get("volatility"), 0.0
                ),
                "trend_strength": self._to_float(
                    market_conditions.get("trend_strength"), 0.0
                ),
                "liquidity": self._to_float(market_conditions.get("liquidity"), 1.0),
            }

            features[pattern] = pattern_features

        return features

    def _predict_pattern_weight(self, pattern: str, features: dict[str, float]) -> float:
        """Predict optimal weight for a pattern using lightweight heuristics."""
        try:
            weight = 0.0
            weight += features.get("sharpe_ratio", 0.0) * 0.4
            weight += features.get("win_rate", 0.0) * 0.3
            weight += (1.0 - abs(features.get("max_drawdown", 0.0))) * 0.3

            volatility_penalty = features.get("market_volatility", 0.0) * 0.1
            return max(0.0, weight - volatility_penalty)

        except Exception as e:
            logger.warning(f"Weight prediction failed for {pattern}: {e}")
            return 0.1

    def _summarize_feature_importance(self) -> dict[str, float]:
        """Summarize feature importance across all models."""
        if not self.feature_importance:
            return {}

        all_importance: dict[str, list[float]] = {}
        for importance in self.feature_importance.values():
            for feature, imp in importance.items():
                all_importance.setdefault(feature, []).append(float(imp))

        avg_importance = {k: float(np.mean(v)) for k, v in all_importance.items()}
        sorted_importance = sorted(
            avg_importance.items(), key=lambda x: x[1], reverse=True
        )

        return dict(sorted_importance[:10])

    def _get_model_performance_metrics(self) -> dict[str, object]:
        """Get performance metrics for all models."""
        metrics: dict[str, object] = {}
        for model_name in self.models.keys():
            metrics[model_name] = {
                "trained": True,
                "feature_count": len(self.feature_importance.get(model_name, {})),
            }

        return metrics

    def _get_pattern_discovery_stats(self) -> dict[str, int]:
        """Get statistics about pattern discovery."""
        keys = list(self.pattern_library.keys())
        return {
            "patterns_discovered": len(keys),
            "correlation_patterns": len([name for name in keys if "correlation" in name]),
            "rule_patterns": len([name for name in keys if "rule" in name]),
            "feature_patterns": len([name for name in keys if "feature" in name]),
        }

def create_pattern_optimizer(config: MLIntegrationConfig) -> IPatternOptimizer:
    """Factory function to create pattern optimizer."""
    if config.pattern_optimizer.advanced_features:
        return AdvancedPatternOptimizer(config.pattern_optimizer)
    return BasePatternOptimizer(config.pattern_optimizer)
