"""
Statistical Analyzer for Market Regime Detection.

This module provides statistical analysis capabilities for market regime detection,
including validation, performance metrics, and statistical significance testing.
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from scipy import stats


@dataclass
class StatisticalTestResult:
    """Result of statistical test."""

    test_name: str
    statistic: float
    p_value: float
    significant: bool
    alpha: float = 0.05
    interpretation: str = ""


@dataclass
class RegimeValidationMetrics:
    """Validation metrics for regime detection."""

    accuracy: float
    precision: float
    recall: float
    f1_score: float
    sharpe_ratio: float  # Add Sharpe ratio
    confusion_matrix: Dict[str, Dict[str, int]]


class StatisticalAnalyzer:
    """
    Statistical analyzer for market regime detection validation and analysis.

    This class provides comprehensive statistical analysis including:
    - Regime detection accuracy validation
    - Statistical significance testing
    - Performance metrics calculation
    - Cross-validation analysis
    """

    def __init__(self, confidence_level: float = 0.95):
        """
        Initialize the statistical analyzer.

        Args:
            confidence_level: Confidence level for statistical tests
        """
        self.confidence_level = confidence_level
        self.alpha = 1 - confidence_level

    def validate_regime_detection(
        self,
        predicted_regimes: List[str],
        actual_regimes: List[str],
        regime_labels: Optional[List[str]] = None,
    ) -> RegimeValidationMetrics:
        """
        Validate regime detection performance.

        Args:
            predicted_regimes: List of predicted regime labels
            actual_regimes: List of actual regime labels
            regime_labels: Optional list of all possible regime labels

        Returns:
            Validation metrics
        """
        if len(predicted_regimes) != len(actual_regimes):
            raise ValueError("Predicted and actual regimes must have same length")

        if regime_labels is None:
            regime_labels = list(set(predicted_regimes + actual_regimes))

        # Calculate confusion matrix
        confusion_matrix = {
            label: dict.fromkeys(regime_labels, 0) for label in regime_labels
        }

        correct_predictions = 0
        total_predictions = len(predicted_regimes)

        for pred, actual in zip(predicted_regimes, actual_regimes):
            confusion_matrix[actual][pred] += 1
            if pred == actual:
                correct_predictions += 1

        # Calculate metrics
        accuracy = (
            correct_predictions / total_predictions if total_predictions > 0 else 0
        )

        # Calculate precision, recall, F1 for each regime
        precisions, recalls, f1_scores = [], [], []

        for regime in regime_labels:
            # True positives, false positives, false negatives
            tp = confusion_matrix[regime][regime]
            fp = sum(
                confusion_matrix[other][regime]
                for other in regime_labels
                if other != regime
            )
            fn = sum(
                confusion_matrix[regime][other]
                for other in regime_labels
                if other != regime
            )

            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = (
                2 * precision * recall / (precision + recall)
                if (precision + recall) > 0
                else 0
            )

            precisions.append(precision)
            recalls.append(recall)
            f1_scores.append(f1)

        # Macro-averaged metrics
        avg_precision = np.mean(precisions)
        avg_recall = np.mean(recalls)
        avg_f1 = np.mean(f1_scores)

        # Calculate Sharpe ratio (simplified - would need actual returns data)
        # For now, use a placeholder calculation
        regime_sharpe = 0.0  # Placeholder - would need actual return series

        return RegimeValidationMetrics(
            accuracy=accuracy,
            precision=avg_precision,
            recall=avg_recall,
            f1_score=avg_f1,
            sharpe_ratio=regime_sharpe,
            confusion_matrix=confusion_matrix,
        )

    def test_regime_stability(
        self, regime_sequence: List[str], window_size: int = 20
    ) -> StatisticalTestResult:
        """
        Test regime detection stability using runs test.

        Args:
            regime_sequence: Sequence of regime detections
            window_size: Window size for stability analysis

        Returns:
            Statistical test result
        """
        if len(regime_sequence) < window_size:
            return StatisticalTestResult(
                test_name="regime_stability",
                statistic=0.0,
                p_value=1.0,
                significant=False,
                interpretation="Insufficient data for stability test",
            )

        # Convert regime sequence to binary (change vs no change)
        changes = []
        for i in range(1, len(regime_sequence)):
            changes.append(1 if regime_sequence[i] != regime_sequence[i - 1] else 0)

        # Runs test for randomness (stability)
        n_runs = 1
        for i in range(1, len(changes)):
            if changes[i] != changes[i - 1]:
                n_runs += 1

        n_changes = sum(changes)
        n_no_changes = len(changes) - n_changes

        # Expected runs and variance
        expected_runs = (2 * n_changes * n_no_changes) / len(changes) + 1
        variance = (expected_runs - 1) * (expected_runs - 2) / (len(changes) - 1)

        if variance > 0:
            z_statistic = (n_runs - expected_runs) / np.sqrt(variance)
            p_value = 2 * (1 - stats.norm.cdf(abs(z_statistic)))
        else:
            z_statistic = 0
            p_value = 1.0

        significant = p_value < self.alpha

        interpretation = (
            "Regime detection is stable (low frequency of changes)"
            if significant
            else "Regime detection shows normal variation"
        )

        return StatisticalTestResult(
            test_name="regime_stability_runs_test",
            statistic=z_statistic,
            p_value=p_value,
            significant=significant,
            interpretation=interpretation,
        )

    def analyze_indicator_significance(
        self, indicators: Dict[str, List[float]], regime_labels: List[str]
    ) -> Dict[str, StatisticalTestResult]:
        """
        Analyze statistical significance of indicators for regime classification.

        Args:
            indicators: Dictionary of indicator values by regime
            regime_labels: List of regime labels

        Returns:
            Dictionary of statistical test results for each indicator
        """
        results = {}

        for indicator_name, values in indicators.items():
            if len(values) != len(regime_labels):
                continue

            # ANOVA test for differences between regimes
            unique_regimes = list(set(regime_labels))
            if len(unique_regimes) < 2:
                continue

            groups = []
            for regime in unique_regimes:
                regime_values = [
                    v for v, r in zip(values, regime_labels) if r == regime
                ]
                if len(regime_values) > 1:  # Need at least 2 values for t-test
                    groups.append(regime_values)

            if len(groups) >= 2:
                try:
                    f_statistic, p_value = stats.f_oneway(*groups)
                    significant = p_value < self.alpha

                    interpretation = (
                        f"Indicator '{indicator_name}' shows significant differences between regimes"
                        if significant
                        else f"Indicator '{indicator_name}' does not show significant differences between regimes"
                    )

                    results[indicator_name] = StatisticalTestResult(
                        test_name=f"anova_{indicator_name}",
                        statistic=f_statistic,
                        p_value=p_value,
                        significant=significant,
                        interpretation=interpretation,
                    )
                except Exception:
                    continue

        return results

    def calculate_regime_transition_matrix(
        self, regime_sequence: List[str]
    ) -> Dict[str, Dict[str, float]]:
        """
        Calculate regime transition probability matrix.

        Args:
            regime_sequence: Sequence of regime detections

        Returns:
            Transition probability matrix
        """
        if len(regime_sequence) < 2:
            return {}

        unique_regimes = list(set(regime_sequence))
        transition_matrix: Dict[str, Dict[str, float]] = {
            regime: dict.fromkeys(unique_regimes, 0.0)
            for regime in unique_regimes
        }

        # Count transitions
        for i in range(len(regime_sequence) - 1):
            current = regime_sequence[i]
            next_regime = regime_sequence[i + 1]
            transition_matrix[current][next_regime] += 1

        # Convert to probabilities
        for current in unique_regimes:
            total_transitions = sum(transition_matrix[current].values())
            if total_transitions > 0:
                for next_regime in unique_regimes:
                    transition_matrix[current][next_regime] /= total_transitions

        return transition_matrix

    def perform_cross_validation(
        self, regime_data: List[Tuple[Dict[str, float], str]], n_folds: int = 5
    ) -> Dict[str, Any]:
        """
        Perform cross-validation on regime detection.

        Args:
            regime_data: List of (indicators, regime) tuples
            n_folds: Number of cross-validation folds

        Returns:
            Cross-validation results
        """
        if len(regime_data) < n_folds:
            return {"error": "Insufficient data for cross-validation"}

        # Simple cross-validation (can be enhanced with actual ML model)
        fold_size = len(regime_data) // n_folds
        accuracies = []

        for fold in range(n_folds):
            test_start = fold * fold_size
            test_end = (
                (fold + 1) * fold_size if fold < n_folds - 1 else len(regime_data)
            )

            # Simple validation: check if indicators are consistent within regimes
            test_data = regime_data[test_start:test_end]
            correct = 0

            for indicators, regime in test_data:
                # This is a simplified validation - in practice, you'd train a model
                predicted = self._simple_regime_prediction(indicators)
                if predicted == regime:
                    correct += 1

            accuracy = correct / len(test_data) if test_data else 0
            accuracies.append(accuracy)

        return {
            "mean_accuracy": np.mean(accuracies),
            "std_accuracy": np.std(accuracies),
            "min_accuracy": min(accuracies),
            "max_accuracy": max(accuracies),
            "cv_folds": n_folds,
        }

    def _simple_regime_prediction(self, indicators: Dict[str, float]) -> str:
        """
        Simple regime prediction based on indicator thresholds.
        This is a placeholder for actual ML-based prediction.
        """
        # Simplified prediction logic
        rsi = indicators.get("rsi", 50)
        adx = indicators.get("adx", 25)
        volatility = indicators.get("volatility", 0.01)
        momentum = indicators.get("momentum", 0)

        if volatility > 0.03:
            return "extreme_volatility"
        elif adx > 30 and abs(momentum) > 0.02:
            return "strong_bull_trend" if momentum > 0 else "strong_bear_trend"
        elif adx > 25 and abs(momentum) > 0.01:
            return "moderate_bull_trend" if momentum > 0 else "moderate_bear_trend"
        elif volatility > 0.02:
            return "high_volatility_ranging"
        elif adx < 20:
            return "consolidation"
        else:
            return "weak_bull_trend" if momentum > 0 else "weak_bear_trend"

    def analyze_regime_detection_quality(
        self, detection_results: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Comprehensive analysis of regime detection quality.

        Args:
            detection_results: List of regime detection results

        Returns:
            Quality analysis results
        """
        if not detection_results:
            return {"error": "No detection results provided"}

        # Extract confidence scores and regimes
        confidences = [r.get("confidence", 0) for r in detection_results]
        regimes = [r.get("regime", "unknown") for r in detection_results]

        # Basic statistics
        analysis = {
            "total_detections": len(detection_results),
            "unique_regimes": len(set(regimes)),
            "mean_confidence": np.mean(confidences),
            "std_confidence": np.std(confidences),
            "min_confidence": min(confidences),
            "max_confidence": max(confidences),
            "regime_distribution": {},
        }

        # Regime distribution
        for regime in set(regimes):
            count = regimes.count(regime)
            analysis["regime_distribution"][regime] = {
                "count": count,
                "percentage": count / len(regimes) * 100,
            }

        # Confidence analysis by regime
        analysis["confidence_by_regime"] = {}
        for regime in set(regimes):
            regime_confidences = [
                c for c, r in zip(confidences, regimes) if r == regime
            ]
            if regime_confidences:
                analysis["confidence_by_regime"][regime] = {
                    "mean": np.mean(regime_confidences),
                    "std": np.std(regime_confidences),
                    "min": min(regime_confidences),
                    "max": max(regime_confidences),
                }

        return analysis
