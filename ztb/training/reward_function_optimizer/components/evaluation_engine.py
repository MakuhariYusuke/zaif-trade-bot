"""
Evaluation Engine - Handles evaluation logic for reward function optimization.

This module separates evaluation-related logic from the main optimizer class,
including performance measurement, cross-validation, and result analysis.
"""

from collections import deque
from typing import Callable

from ztb.metrics.metrics import coefficient_of_variation
from ztb.utils.logging_utils import get_logger
from ztb.utils.safety import ensure_dict, safe_to_float

logger = get_logger(__name__)

ConfigMap = dict[str, object]
ScoreMap = dict[str, float]
EvaluationRecord = dict[str, object]

class EvaluationEngine:
    """
    Handles evaluation of reward function configurations.

    This class manages:
    - Performance evaluation across different market conditions
    - Cross-validation execution
    - Result aggregation and analysis
    - Statistical significance testing
    """

    def __init__(self, history_limit: int = 2000):
        """Initialize EvaluationEngine."""
        self.logger = get_logger(__name__)
        self.evaluation_history: deque[EvaluationRecord] = deque(maxlen=history_limit)

    def evaluate_configuration(
        self,
        config: ConfigMap,
        evaluation_function: Callable[[ConfigMap], ScoreMap],
        market_conditions: list[ConfigMap] | None = None,
    ) -> ConfigMap:
        """
        Evaluate a reward function configuration.

        Args:
            config: Configuration to evaluate
            evaluation_function: Function to perform evaluation
            market_conditions: Different market conditions to test

        Returns:
            Evaluation results

        Raises:
            RuntimeError: If evaluation fails
        """
        try:
            if market_conditions is None:
                market_conditions = [{}]  # Default single evaluation

            all_results: list[ConfigMap] = []

            for i, condition in enumerate(market_conditions):
                self.logger.debug(
                    f"Evaluating condition {i + 1}/{len(market_conditions)}"
                )

                # Merge config with market condition
                eval_config = {**config, **condition}

                # Run evaluation
                raw_result = evaluation_function(eval_config)
                result = ensure_dict(raw_result)
                result["condition"] = condition
                result["condition_index"] = i

                all_results.append(result)

            # Aggregate results
            aggregated_results = self._aggregate_results(all_results)

            evaluation_record: EvaluationRecord = {
                "config": config,
                "results": aggregated_results,
                "individual_results": all_results,
                "n_conditions": len(market_conditions),
                "timestamp": self._get_timestamp(),
            }

            self.evaluation_history.append(evaluation_record)

            self.logger.info(
                f"Configuration evaluation completed. Score: {aggregated_results.get('mean_score', 'N/A')}"
            )
            return aggregated_results

        except Exception as e:
            self.logger.error(f"Configuration evaluation failed: {e}")
            raise RuntimeError(f"Configuration evaluation failed: {e}") from e

    def cross_validate(
        self,
        config: ConfigMap,
        evaluation_function: Callable[[ConfigMap], ScoreMap],
        n_folds: int = 5,
        validation_function: Callable[[ConfigMap], bool] | None = None,
    ) -> ConfigMap:
        """
        Perform cross-validation on configuration.

        Args:
            config: Configuration to validate
            evaluation_function: Function to perform evaluation
            n_folds: Number of cross-validation folds
            validation_function: Optional validation function

        Returns:
            Cross-validation results

        Raises:
            RuntimeError: If cross-validation fails
        """
        try:
            fold_results: list[ConfigMap] = []

            for fold in range(n_folds):
                self.logger.debug(f"Cross-validation fold {fold + 1}/{n_folds}")

                # Run evaluation for this fold
                result = ensure_dict(evaluation_function(config))
                result["fold"] = fold

                fold_results.append(result)

            # Calculate cross-validation statistics
            cv_results = self._calculate_cv_statistics(fold_results)

            cv_record: EvaluationRecord = {
                "config": config,
                "cv_results": cv_results,
                "fold_results": fold_results,
                "n_folds": n_folds,
                "timestamp": self._get_timestamp(),
            }

            self.evaluation_history.append(cv_record)

            self.logger.info(
                f"Cross-validation completed. CV score: {cv_results.get('mean_score', 'N/A')}"
            )
            return cv_results

        except Exception as e:
            self.logger.error(f"Cross-validation failed: {e}")
            raise RuntimeError(f"Cross-validation failed: {e}") from e

    def compare_configurations(
        self,
        configs: list[ConfigMap],
        evaluation_function: Callable[[ConfigMap], ScoreMap],
        statistical_test: bool = True,
    ) -> ConfigMap:
        """
        Compare multiple configurations.

        Args:
            configs: list of configurations to compare
            evaluation_function: Function to evaluate configurations
            statistical_test: Whether to perform statistical significance testing

        Returns:
            Comparison results

        Raises:
            RuntimeError: If comparison fails
        """
        try:
            comparison_results: list[ConfigMap] = []

            for i, config in enumerate(configs):
                self.logger.debug(f"Evaluating configuration {i + 1}/{len(configs)}")

                result = self.evaluate_configuration(config, evaluation_function)
                result["config_index"] = i
                result["config"] = config

                comparison_results.append(result)

            # Sort by performance
            comparison_results.sort(
                key=lambda x: safe_to_float(x.get("mean_score", 0.0), 0.0),
                reverse=True,
            )

            # Perform statistical testing if requested
            if statistical_test and len(comparison_results) > 1:
                statistical_results = self._perform_statistical_testing(
                    comparison_results
                )
            else:
                statistical_results = {}

            comparison_summary = {
                "ranked_configs": comparison_results,
                "best_config": comparison_results[0] if comparison_results else None,
                "statistical_analysis": statistical_results,
                "n_configs": len(configs),
                "timestamp": self._get_timestamp(),
            }

            self.logger.info(
                f"Configuration comparison completed. Best score: {comparison_results[0].get('mean_score', 'N/A') if comparison_results else 'N/A'}"
            )
            return comparison_summary

        except Exception as e:
            self.logger.error(f"Configuration comparison failed: {e}")
            raise RuntimeError(f"Configuration comparison failed: {e}") from e

    def _aggregate_results(self, results: list[ConfigMap]) -> ConfigMap:
        """Aggregate multiple evaluation results."""
        try:
            if not results:
                return {}

            import numpy as np

            # Extract scores
            scores = [self._extract_primary_score(r) for r in results]

            if not scores:
                return {"error": "No scores found in results"}

            aggregated = {
                "mean_score": float(np.mean(scores)),
                "std_score": float(np.std(scores)),
                "min_score": float(np.min(scores)),
                "max_score": float(np.max(scores)),
                "median_score": float(np.median(scores)),
                "n_results": len(results),
                "n_scores": len(scores),
                "scores": [float(s) for s in scores],
            }

            # Add additional metrics if available
            for metric_name in ["profit", "sharpe_ratio", "win_rate", "max_drawdown"]:
                metric_values = [
                    safe_to_float(r.get(metric_name, 0.0), 0.0)
                    for r in results
                    if metric_name in r
                ]
                if metric_values:
                    aggregated[f"mean_{metric_name}"] = float(np.mean(metric_values))
                    aggregated[f"std_{metric_name}"] = float(np.std(metric_values))

            return aggregated

        except Exception as e:
            self.logger.error(f"Failed to aggregate results: {e}")
            return {"error": f"Aggregation failed: {e}"}

    def _calculate_cv_statistics(
        self, fold_results: list[ConfigMap]
    ) -> ConfigMap:
        """Calculate cross-validation statistics."""
        try:
            if not fold_results:
                return {}

            import numpy as np

            scores = [self._extract_primary_score(r) for r in fold_results]

            if not scores:
                return {"error": "No scores found in fold results"}

            cv_stats = {
                "cv_mean_score": float(np.mean(scores)),
                "cv_std_score": float(np.std(scores)),
                "cv_min_score": float(np.min(scores)),
                "cv_max_score": float(np.max(scores)),
                "cv_score_variance": float(np.var(scores)),
                "n_folds": len(fold_results),
                "n_valid_scores": len(scores),
            }

            # Coefficient of variation
            cv_stats["cv_coefficient_of_variation"] = coefficient_of_variation(scores)

            return cv_stats

        except Exception as e:
            self.logger.error(f"Failed to calculate CV statistics: {e}")
            return {"error": f"CV statistics calculation failed: {e}"}

    def _perform_statistical_testing(
        self, results: list[ConfigMap]
    ) -> ConfigMap:
        """Perform statistical significance testing between configurations."""
        try:
            if len(results) < 2:
                return {
                    "error": "Need at least 2 configurations for statistical testing"
                }

            # Simple t-test between best and second best
            best_scores = results[0].get("scores", [])
            second_best_scores = results[1].get("scores", []) if len(results) > 1 else []

            if (
                not isinstance(best_scores, list)
                or not isinstance(second_best_scores, list)
                or not best_scores
                or not second_best_scores
            ):
                return {"error": "Insufficient score data for statistical testing"}

            from scipy import stats

            # Perform t-test
            best_arr = [safe_to_float(v, 0.0) for v in best_scores]
            second_arr = [safe_to_float(v, 0.0) for v in second_best_scores]
            t_stat, p_value = stats.ttest_ind(best_arr, second_arr)
            t_stat_f = safe_to_float(t_stat, 0.0)
            p_value_f = safe_to_float(p_value, 1.0)

            statistical_results = {
                "t_statistic": t_stat_f,
                "p_value": p_value_f,
                "significant_difference": p_value_f < 0.05,
                "best_vs_second_best": True,
                "test_type": "t-test",
            }

            self.logger.debug(f"Statistical test: t={t_stat_f:.3f}, p={p_value_f:.3f}")

            return statistical_results

        except ImportError:
            self.logger.warning("scipy not available for statistical testing")
            return {"error": "scipy not available"}
        except Exception as e:
            self.logger.error(f"Statistical testing failed: {e}")
            return {"error": f"Statistical testing failed: {e}"}

    def _get_timestamp(self) -> str:
        """Get current timestamp."""
        from datetime import datetime

        return datetime.now().isoformat()

    @staticmethod
    def _extract_primary_score(result: ConfigMap) -> float:
        # Prefer explicit score; otherwise fallback to profit-based score.
        if "score" in result:
            return safe_to_float(result.get("score", 0.0), 0.0)
        return safe_to_float(result.get("profit", 0.0), 0.0)

    def get_evaluation_history(self) -> list[EvaluationRecord]:
        """Get evaluation history."""
        return list(self.evaluation_history)

    def clear_history(self) -> None:
        """Clear evaluation history."""
        self.evaluation_history.clear()
        self.logger.info("Evaluation history cleared")
