"""
Performance Evaluator

Evaluates the effectiveness of weight adjustments and provides
performance metrics for the adjustment system.
"""

import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta

from ztb.utils.logging_utils import get_logger

from ..utils.data_processor import DataProcessor
from ..utils.validation_utils import ValidationUtils

logger = get_logger(__name__)


class PerformanceEvaluator:
    """
    Evaluates the performance of weight adjustments.

    Provides metrics and analysis to determine the effectiveness
    of different weight adjustment strategies.
    """

    def __init__(self, evaluation_window: int = 100):
        """
        Initialize PerformanceEvaluator.

        Args:
            evaluation_window: Number of data points to use for evaluation
        """
        self.evaluation_window = evaluation_window
        self.data_processor = DataProcessor()
        self.validator = ValidationUtils()

        # Performance tracking
        self.performance_history: List[Dict[str, Any]] = []
        self.baseline_performance: Optional[Dict[str, float]] = None

    def evaluate_adjustment_impact(
        self,
        before_weights: Dict[str, float],
        after_weights: Dict[str, float],
        performance_data: Dict[str, Any],
        feature_importance: Dict[str, float]
    ) -> Dict[str, Any]:
        """
        Evaluate the impact of a weight adjustment.

        Args:
            before_weights: Weights before adjustment
            after_weights: Weights after adjustment
            performance_data: Performance data after adjustment
            feature_importance: Feature importance scores

        Returns:
            Dictionary containing impact analysis
        """
        try:
            # Calculate weight changes
            weight_changes = self._calculate_weight_changes(before_weights, after_weights)

            # Evaluate performance impact
            performance_impact = self._evaluate_performance_impact(
                performance_data, weight_changes
            )

            # Calculate feature contribution changes
            contribution_changes = self._calculate_contribution_changes(
                before_weights, after_weights, feature_importance
            )

            # Overall impact score
            impact_score = self._calculate_impact_score(
                weight_changes, performance_impact, contribution_changes
            )

            result = {
                "weight_changes": weight_changes,
                "performance_impact": performance_impact,
                "contribution_changes": contribution_changes,
                "impact_score": impact_score,
                "evaluation_timestamp": datetime.now().isoformat(),
                "recommendation": self._generate_recommendation(impact_score),
            }

            # Store in history
            self.performance_history.append(result)

            # Maintain history size
            if len(self.performance_history) > self.evaluation_window:
                self.performance_history = self.performance_history[-self.evaluation_window:]

            return result

        except Exception as e:
            logger.error(f"Failed to evaluate adjustment impact: {e}")
            return {
                "error": str(e),
                "impact_score": 0.0,
                "recommendation": "evaluation_failed",
            }

    def set_baseline_performance(self, performance_data: Dict[str, Any]) -> None:
        """
        Set baseline performance for comparison.

        Args:
            performance_data: Baseline performance data
        """
        if self.validator.validate_performance_data(performance_data):
            self.baseline_performance = {
                key: value for key, value in performance_data.items()
                if isinstance(value, (int, float))
            }
            logger.info("Baseline performance set")
        else:
            logger.error("Invalid baseline performance data")

    def get_performance_trends(self, window: Optional[int] = None) -> Dict[str, Any]:
        """
        Get performance trends over time.

        Args:
            window: Number of recent evaluations to analyze

        Returns:
            Dictionary containing trend analysis
        """
        if not self.performance_history:
            return {"error": "No performance history available"}

        # Use specified window or all history
        history = self.performance_history[-window:] if window else self.performance_history

        if len(history) < 2:
            return {"error": "Insufficient history for trend analysis"}

        # Extract impact scores
        impact_scores = [entry.get("impact_score", 0.0) for entry in history]
        timestamps = [entry.get("evaluation_timestamp") for entry in history]

        # Calculate trends
        trends = self._calculate_trends(impact_scores, timestamps)

        # Performance stability
        stability = self._calculate_stability_metrics(impact_scores)

        return {
            "trend_analysis": trends,
            "stability_metrics": stability,
            "sample_size": len(history),
            "analysis_window": window,
        }

    def get_adjustment_effectiveness(self) -> Dict[str, Any]:
        """
        Get overall effectiveness metrics for weight adjustments.

        Returns:
            Dictionary containing effectiveness analysis
        """
        if not self.performance_history:
            return {"error": "No performance history available"}

        # Aggregate impact scores
        impact_scores = [entry.get("impact_score", 0.0) for entry in self.performance_history]

        # Calculate effectiveness metrics
        effectiveness = {
            "average_impact": np.mean(impact_scores),
            "median_impact": np.median(impact_scores),
            "best_impact": np.max(impact_scores),
            "worst_impact": np.min(impact_scores),
            "impact_std": np.std(impact_scores),
            "positive_adjustments": sum(1 for score in impact_scores if score > 0),
            "negative_adjustments": sum(1 for score in impact_scores if score < 0),
            "neutral_adjustments": sum(1 for score in impact_scores if score == 0),
            "total_adjustments": len(impact_scores),
        }

        # Success rate
        effectiveness["success_rate"] = (
            effectiveness["positive_adjustments"] / effectiveness["total_adjustments"]
            if effectiveness["total_adjustments"] > 0 else 0.0
        )

        # Effectiveness rating
        effectiveness["effectiveness_rating"] = self._rate_effectiveness(effectiveness)

        return effectiveness

    def _calculate_weight_changes(
        self,
        before_weights: Dict[str, float],
        after_weights: Dict[str, float]
    ) -> Dict[str, float]:
        """
        Calculate changes in feature weights.

        Args:
            before_weights: Weights before adjustment
            after_weights: Weights after adjustment

        Returns:
            Dictionary of weight changes
        """
        changes = {}

        for feature in before_weights:
            if feature in after_weights:
                changes[feature] = after_weights[feature] - before_weights[feature]
            else:
                changes[feature] = -before_weights[feature]  # Feature removed

        # Add new features
        for feature in after_weights:
            if feature not in before_weights:
                changes[feature] = after_weights[feature]

        return changes

    def _evaluate_performance_impact(
        self,
        performance_data: Dict[str, Any],
        weight_changes: Dict[str, float]
    ) -> Dict[str, Any]:
        """
        Evaluate how weight changes impacted performance.

        Args:
            performance_data: Current performance data
            weight_changes: Weight changes made

        Returns:
            Dictionary containing performance impact analysis
        """
        impact = {
            "win_rate_change": 0.0,
            "return_change": 0.0,
            "volatility_change": 0.0,
            "max_drawdown_change": 0.0,
        }

        # Compare with baseline if available
        if self.baseline_performance:
            for metric in impact.keys():
                if metric in performance_data and metric in self.baseline_performance:
                    current = performance_data[metric]
                    baseline = self.baseline_performance[metric]
                    if baseline != 0:
                        impact[metric.replace("_change", "_change")] = (current - baseline) / abs(baseline)

        # Weight change magnitude correlation
        total_weight_change = sum(abs(change) for change in weight_changes.values())
        impact["total_weight_change"] = total_weight_change

        # Feature-specific impact (simplified)
        impact["features_adjusted"] = len([f for f in weight_changes.values() if abs(f) > 0.01])

        return impact

    def _calculate_contribution_changes(
        self,
        before_weights: Dict[str, float],
        after_weights: Dict[str, float],
        feature_importance: Dict[str, float]
    ) -> Dict[str, float]:
        """
        Calculate changes in feature contributions.

        Args:
            before_weights: Weights before adjustment
            after_weights: Weights after adjustment
            feature_importance: Feature importance scores

        Returns:
            Dictionary of contribution changes
        """
        changes = {}

        for feature in before_weights:
            before_contribution = before_weights[feature] * feature_importance.get(feature, 0.0)
            after_contribution = after_weights.get(feature, 0.0) * feature_importance.get(feature, 0.0)
            changes[feature] = after_contribution - before_contribution

        # Add new features
        for feature in after_weights:
            if feature not in before_weights:
                contribution = after_weights[feature] * feature_importance.get(feature, 0.0)
                changes[feature] = contribution

        return changes

    def _calculate_impact_score(
        self,
        weight_changes: Dict[str, float],
        performance_impact: Dict[str, Any],
        contribution_changes: Dict[str, float]
    ) -> float:
        """
        Calculate overall impact score for the adjustment.

        Args:
            weight_changes: Changes in feature weights
            performance_impact: Performance impact metrics
            contribution_changes: Changes in feature contributions

        Returns:
            Impact score between -1 and 1
        """
        # Weight the different factors
        weights = {
            "performance": 0.5,
            "contribution": 0.3,
            "stability": 0.2,
        }

        # Performance score
        performance_score = (
            performance_impact.get("win_rate_change", 0.0) * 0.4 +
            performance_impact.get("return_change", 0.0) * 0.4 +
            -abs(performance_impact.get("volatility_change", 0.0)) * 0.2  # Penalize volatility increase
        )

        # Contribution score
        total_contribution_change = sum(abs(change) for change in contribution_changes.values())
        contribution_score = min(total_contribution_change, 1.0)  # Cap at 1.0

        # Stability score (inverse of weight change magnitude)
        total_weight_change = sum(abs(change) for change in weight_changes.values())
        stability_score = max(0.0, 1.0 - total_weight_change)  # Less change = more stable

        # Combine scores
        impact_score = (
            weights["performance"] * performance_score +
            weights["contribution"] * contribution_score +
            weights["stability"] * stability_score
        )

        # Normalize to [-1, 1] range
        return max(-1.0, min(1.0, impact_score))

    def _calculate_trends(
        self,
        impact_scores: List[float],
        timestamps: List[str]
    ) -> Dict[str, Any]:
        """
        Calculate trends in impact scores.

        Args:
            impact_scores: List of impact scores
            timestamps: Corresponding timestamps

        Returns:
            Dictionary containing trend analysis
        """
        if len(impact_scores) < 2:
            return {"error": "Insufficient data for trend analysis"}

        # Simple linear trend using numpy
        x = np.arange(len(impact_scores))
        y = np.array(impact_scores)

        # Linear regression
        slope, intercept = np.polyfit(x, y, 1)
        r_value = np.corrcoef(x, y)[0, 1]
        p_value = 0.0  # Simplified, not calculating actual p-value
        std_err = np.std(y) / np.sqrt(len(y))

        # Moving averages
        window_size = min(10, len(impact_scores) // 2)
        if window_size >= 2:
            moving_avg = pd.Series(impact_scores).rolling(window=window_size).mean().iloc[-1]
            moving_std = pd.Series(impact_scores).rolling(window=window_size).std().iloc[-1]
        else:
            moving_avg = np.mean(impact_scores)
            moving_std = np.std(impact_scores)

        return {
            "linear_trend_slope": slope,
            "linear_trend_r_squared": r_value ** 2,
            "linear_trend_p_value": p_value,
            "moving_average": moving_avg,
            "moving_std": moving_std,
            "trend_direction": "improving" if slope > 0.01 else "declining" if slope < -0.01 else "stable",
        }

    def _calculate_stability_metrics(self, impact_scores: List[float]) -> Dict[str, Any]:
        """
        Calculate stability metrics for impact scores.

        Args:
            impact_scores: List of impact scores

        Returns:
            Dictionary of stability metrics
        """
        if len(impact_scores) < 2:
            return {"error": "Insufficient data for stability analysis"}

        scores_array = np.array(impact_scores)

        return {
            "mean": np.mean(scores_array),
            "std": np.std(scores_array),
            "cv": np.std(scores_array) / abs(np.mean(scores_array)) if np.mean(scores_array) != 0 else 0.0,
            "min": np.min(scores_array),
            "max": np.max(scores_array),
            "range": np.max(scores_array) - np.min(scores_array),
            "stability_score": 1.0 / (1.0 + np.std(scores_array)),  # Higher = more stable
        }

    def _rate_effectiveness(self, effectiveness: Dict[str, Any]) -> str:
        """
        Rate the overall effectiveness of adjustments.

        Args:
            effectiveness: Effectiveness metrics

        Returns:
            Effectiveness rating
        """
        success_rate = effectiveness.get("success_rate", 0.0)
        average_impact = effectiveness.get("average_impact", 0.0)

        if success_rate > 0.7 and average_impact > 0.1:
            return "excellent"
        elif success_rate > 0.5 and average_impact > 0.0:
            return "good"
        elif success_rate > 0.3:
            return "moderate"
        elif success_rate < 0.2 or average_impact < -0.1:
            return "poor"
        else:
            return "neutral"

    def _generate_recommendation(self, impact_score: float) -> str:
        """
        Generate recommendation based on impact score.

        Args:
            impact_score: Calculated impact score

        Returns:
            Recommendation string
        """
        if impact_score > 0.2:
            return "continue_adjustment"
        elif impact_score > 0.0:
            return "minor_adjustment"
        elif impact_score > -0.2:
            return "stabilize_weights"
        else:
            return "rollback_adjustment"