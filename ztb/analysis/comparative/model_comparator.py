#!/usr/bin/env python3
"""
Model Comparator - Statistical model comparison functionality

This module provides comprehensive statistical comparison between different
trading models using various metrics and statistical tests.
"""

import json
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
from scipy import stats

from ztb.utils.logging_utils import get_logger


class ModelComparator:
    """Statistical comparison of trading models."""

    def __init__(self):
        """Initialize model comparator."""
        self.logger = get_logger(__name__)

    def load_backtest_results(self, file_path: str) -> Dict[str, Any]:
        """Load backtest results from JSON file.

        Args:
            file_path: Path to the backtest results file

        Returns:
            Loaded backtest data
        """
        with open(file_path, "r") as f:
            data = json.load(f)
        return data

    def extract_trade_returns(self, trades: List[Dict[str, Any]]) -> List[float]:
        """Extract individual trade returns from trades data.

        Args:
            trades: List of trade dictionaries

        Returns:
            List of trade returns as percentages
        """
        returns = []
        for trade in trades:
            pnl = float(trade["pnl"])
            # Calculate return percentage based on position size and entry price
            position = abs(float(trade["position"]))
            entry_price = float(trade["entry_price"])
            trade_return = (pnl / (position * entry_price)) * 100
            returns.append(trade_return)
        return returns

    def perform_statistical_tests(
        self, model_results: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Perform t-tests and other statistical comparisons.

        Args:
            model_results: Dictionary of model results

        Returns:
            Statistical test results
        """
        results = {}

        # Extract trade returns for each model
        trade_returns = {}
        for model_name, data in model_results.items():
            trade_returns[model_name] = self.extract_trade_returns(data["trades"])

        # Perform pairwise t-tests
        models = list(model_results.keys())
        for i in range(len(models)):
            for j in range(i + 1, len(models)):
                model1, model2 = models[i], models[j]
                returns1 = trade_returns[model1]
                returns2 = trade_returns[model2]

                # Perform t-test
                t_stat, p_value = stats.ttest_ind(returns1, returns2, equal_var=False)

                results[f"{model1}_vs_{model2}"] = {
                    "t_statistic": float(t_stat),
                    "p_value": float(p_value),
                    "significant": bool(p_value < 0.05),
                    "model1_mean": float(np.mean(returns1)),
                    "model2_mean": float(np.mean(returns2)),
                    "model1_std": float(np.std(returns1)),
                    "model2_std": float(np.std(returns2)),
                }

        return results

    def p_mean_method(self, model_results: Dict[str, Any]) -> Dict[str, Any]:
        """Implement p-mean method for model comparison.

        Args:
            model_results: Dictionary of model results

        Returns:
            P-mean comparison results
        """
        p_mean_results = {}

        for model_name, data in model_results.items():
            metrics = data["metrics"]
            total_return = float(metrics["total_return"])
            sharpe_ratio = float(metrics["sharpe_ratio"])
            win_rate = float(metrics["win_rate"]) / 100  # Convert to decimal
            max_drawdown = abs(float(metrics["max_drawdown"]))

            # Calculate p-mean score (higher is better)
            # Weight: Return (40%), Sharpe (30%), Win Rate (20%), Drawdown penalty (10%)
            p_mean_score = (
                0.4 * total_return
                + 0.3 * sharpe_ratio
                + 0.2 * win_rate * 100  # Scale win rate back
                + 0.1 * (1 / (1 + max_drawdown)) * 100  # Penalize drawdown
            )

            p_mean_results[model_name] = {
                "p_mean_score": p_mean_score,
                "total_return": total_return,
                "sharpe_ratio": sharpe_ratio,
                "win_rate": win_rate * 100,
                "max_drawdown": max_drawdown,
            }

        return p_mean_results

    def compare_models(
        self,
        model_results_files: Dict[str, str],
        output_path: str = None,
    ) -> Dict[str, Any]:
        """Complete model comparison pipeline.

        Args:
            model_results_files: Dictionary mapping model names to result file paths
            output_path: Optional path to save comparison results

        Returns:
            Complete comparison results
        """
        # Load model results
        model_results = {}
        for model_name, file_path in model_results_files.items():
            if Path(file_path).exists():
                model_results[model_name] = self.load_backtest_results(file_path)
                self.logger.info(f"Loaded {model_name} results from {file_path}")
            else:
                self.logger.warning(f"Results file not found: {file_path}")

        if len(model_results) < 2:
            raise ValueError("Need at least 2 model results for comparison")

        # Perform statistical tests
        self.logger.info("Performing statistical tests...")
        stat_results = self.perform_statistical_tests(model_results)

        # Perform p-mean method comparison
        self.logger.info("Performing p-mean method comparison...")
        p_mean_results = self.p_mean_method(model_results)

        # Determine best model
        best_model = max(p_mean_results.items(), key=lambda x: x[1]["p_mean_score"])[0]

        # Prepare final results
        comparison_results = {
            "statistical_tests": stat_results,
            "p_mean_comparison": p_mean_results,
            "best_model": best_model,
            "summary": {
                "models_compared": list(model_results.keys()),
                "total_models": len(model_results),
                "best_performing_model": best_model,
            },
        }

        # Save results if output path provided
        if output_path:
            with open(output_path, "w") as f:
                json.dump(comparison_results, f, indent=2)
            self.logger.info(f"Comparison results saved to {output_path}")

        return comparison_results

    def print_comparison_summary(self, results: Dict[str, Any]) -> None:
        """Print formatted comparison summary.

        Args:
            results: Comparison results dictionary
        """
        print("\n=== MODEL COMPARISON SUMMARY ===")
        print(f"Models compared: {', '.join(results['summary']['models_compared'])}")
        print(f"Best model: {results['best_model']}")

        print("\nP-Mean Scores:")
        for model, model_results in results["p_mean_comparison"].items():
            print(".2f")

        print("\nStatistical Significance (t-tests):")
        for test_name, test_results in results["statistical_tests"].items():
            sig = "SIGNIFICANT" if test_results["significant"] else "NOT SIGNIFICANT"
            print(f"{test_name}: {sig}")
