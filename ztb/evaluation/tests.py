"""
Enhanced testing utilities for evaluation success rate validation.

This module provides comprehensive testing functionality including
success rate validation, edge case testing, and performance regression detection.
"""

from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import numpy as np
import pandas as pd

from ztb.evaluation.logging import EvaluationLogger


def validate_evaluation_success_rate(
    logger: EvaluationLogger, min_success_rate: float = 0.8, time_window_days: int = 30
) -> Dict[str, Any]:
    """
    Validate evaluation success rate over time window

    Args:
        logger: Evaluation logger instance
        min_success_rate: Minimum acceptable success rate
        time_window_days: Time window for validation

    Returns:
        Validation results
    """
    # Get evaluation history
    history = logger.get_evaluation_history(days=time_window_days)

    if not history:
        return {
            "status": "no_data",
            "message": f"No evaluation data found in last {time_window_days} days",
        }

    # Calculate success rate
    total_evaluations = len(history)
    successful_evaluations = len([r for r in history if r["status"] == "success"])
    success_rate = (
        successful_evaluations / total_evaluations if total_evaluations > 0 else 0
    )

    # Check against minimum threshold
    success_threshold_met = success_rate >= min_success_rate

    # Calculate recent trend (last 7 days vs previous 23 days)
    recent_cutoff = datetime.now() - timedelta(days=7)
    recent_evaluations = [
        r for r in history if datetime.fromisoformat(r["timestamp"]) >= recent_cutoff
    ]
    older_evaluations = [
        r for r in history if datetime.fromisoformat(r["timestamp"]) < recent_cutoff
    ]

    recent_success_rate = (
        len([r for r in recent_evaluations if r["status"] == "success"])
        / len(recent_evaluations)
        if recent_evaluations
        else 0
    )
    older_success_rate = (
        len([r for r in older_evaluations if r["status"] == "success"])
        / len(older_evaluations)
        if older_evaluations
        else 0
    )

    trend = (
        "improving"
        if recent_success_rate > older_success_rate
        else "declining" if recent_success_rate < older_success_rate else "stable"
    )

    return {
        "status": "success" if success_threshold_met else "failure",
        "success_rate": success_rate,
        "successful_evaluations": successful_evaluations,
        "total_evaluations": total_evaluations,
        "threshold_met": success_threshold_met,
        "min_success_rate": min_success_rate,
        "time_window_days": time_window_days,
        "trend": trend,
        "recent_success_rate": recent_success_rate,
        "older_success_rate": older_success_rate,
        "recent_evaluations": len(recent_evaluations),
        "older_evaluations": len(older_evaluations),
    }


def test_feature_computation_stability(
    feature_func: Callable[[pd.DataFrame], Any],
    ohlc_data: pd.DataFrame,
    n_runs: int = 10,
) -> Dict[str, Any]:
    """
    Test feature computation stability across multiple runs

    Args:
        feature_func: Feature computation function
        ohlc_data: OHLC data for testing
        n_runs: Number of test runs

    Returns:
        Stability test results
    """
    results = []
    errors = []

    for _ in range(n_runs):
        try:
            result = feature_func(ohlc_data)
            results.append(result)
        except Exception as e:
            errors.append(str(e))

    if not results:
        return {
            "status": "failure",
            "message": f"All {n_runs} runs failed",
            "errors": errors,
        }

    # Check result consistency
    first_result = results[0]
    consistent = True
    max_diff = 0

    for result in results[1:]:
        if isinstance(first_result, pd.DataFrame):
            diff = (first_result - result).abs().max().max()
        elif isinstance(first_result, pd.Series):
            diff = (first_result - result).abs().max()
        else:
            diff = abs(first_result - result) if np.isscalar(first_result) else 0

        max_diff = max(max_diff, diff)
        if diff > 1e-10:  # Allow for small numerical differences
            consistent = False

    success_rate = len(results) / n_runs

    return {
        "status": (
            "success"
            if consistent and success_rate >= 0.9
            else "warning" if success_rate >= 0.7 else "failure"
        ),
        "success_rate": success_rate,
        "consistent": consistent,
        "max_difference": max_diff,
        "n_runs": n_runs,
        "errors": errors,
    }


def test_evaluation_pipeline_robustness(
    evaluation_func: Callable[..., Any], test_cases: List[Dict[str, Any]]
) -> Dict[str, Any]:
    """
    Test evaluation pipeline robustness with various edge cases

    Args:
        evaluation_func: Evaluation function to test
        test_cases: List of test case dictionaries

    Returns:
        Robustness test results
    """
    results = []

    for i, test_case in enumerate(test_cases):
        try:
            result = evaluation_func(**test_case["args"])

            # Validate result structure
            expected_keys = test_case.get("expected_keys", [])
            has_expected_keys = all(key in result for key in expected_keys)

            # Check for errors
            has_error = "error" in result and result["error"]

            test_result = {
                "test_case": i,
                "status": (
                    "success" if has_expected_keys and not has_error else "failure"
                ),
                "has_expected_keys": has_expected_keys,
                "has_error": has_error,
                "result_keys": (
                    list(result.keys()) if isinstance(result, dict) else None
                ),
                "error": result.get("error"),
            }

        except Exception as e:
            test_result = {"test_case": i, "status": "exception", "exception": str(e)}

        results.append(test_result)

    # Summarize results
    successful_tests = len([r for r in results if r["status"] == "success"])
    total_tests = len(results)
    success_rate = successful_tests / total_tests if total_tests > 0 else 0

    return {
        "status": "success" if success_rate >= 0.8 else "failure",
        "success_rate": success_rate,
        "successful_tests": successful_tests,
        "total_tests": total_tests,
        "test_results": results,
    }


def detect_performance_regressions(
    logger: EvaluationLogger,
    baseline_period_days: int = 90,
    recent_period_days: int = 7,
    regression_threshold: float = 0.1,
) -> Dict[str, Any]:
    """
    Detect performance regressions in evaluation metrics

    Args:
        logger: Evaluation logger instance
        baseline_period_days: Baseline period for comparison
        recent_period_days: Recent period to check for regressions
        regression_threshold: Threshold for regression detection (10%)

    Returns:
        Regression analysis results
    """
    # Get baseline and recent data
    baseline_data = logger.get_evaluation_history(days=baseline_period_days)
    recent_data = logger.get_evaluation_history(days=recent_period_days)

    if not baseline_data or not recent_data:
        return {
            "status": "insufficient_data",
            "message": "Not enough data for regression analysis",
        }

    # Calculate baseline metrics
    baseline_successful = [r for r in baseline_data if r["status"] == "success"]
    baseline_metrics = {
        "computation_time": (
            np.mean([r["computation_time_ms"] for r in baseline_successful])
            if baseline_successful
            else 0
        ),
        "nan_rate": (
            np.mean([r["nan_rate"] for r in baseline_successful])
            if baseline_successful
            else 0
        ),
        "success_rate": (
            len(baseline_successful) / len(baseline_data) if baseline_data else 0
        ),
    }

    # Calculate recent metrics
    recent_successful = [r for r in recent_data if r["status"] == "success"]
    recent_metrics = {
        "computation_time": (
            np.mean([r["computation_time_ms"] for r in recent_successful])
            if recent_successful
            else 0
        ),
        "nan_rate": (
            np.mean([r["nan_rate"] for r in recent_successful])
            if recent_successful
            else 0
        ),
        "success_rate": len(recent_successful) / len(recent_data) if recent_data else 0,
    }

    # Detect regressions
    regressions = {}

    for metric, baseline_value in baseline_metrics.items():
        recent_value = recent_metrics[metric]

        if baseline_value > 0:
            change = (recent_value - baseline_value) / baseline_value

            # For some metrics, increase is bad (time, nan_rate), for others decrease is bad (success_rate)
            if metric in ["computation_time", "nan_rate"]:
                is_regression = change > regression_threshold
            else:  # success_rate
                is_regression = change < -regression_threshold

            if is_regression:
                regressions[metric] = {
                    "baseline": baseline_value,
                    "recent": recent_value,
                    "change": change,
                    "threshold": regression_threshold,
                }

    return {
        "status": "success",
        "has_regressions": len(regressions) > 0,
        "regressions": regressions,
        "baseline_metrics": baseline_metrics,
        "recent_metrics": recent_metrics,
        "baseline_period_days": baseline_period_days,
        "recent_period_days": recent_period_days,
    }


def generate_test_report(
    test_results: Dict[str, Any], output_path: Optional[str] = None
) -> str:
    """
    Generate comprehensive test report

    Args:
        test_results: Test results dictionary
        output_path: Optional path to save report

    Returns:
        Formatted test report
    """
    report_lines = [
        "# Evaluation Testing Report",
        f"Generated: {datetime.now().isoformat()}",
        "",
        "## Summary",
    ]

    # Overall status
    status = test_results.get("status", "unknown")
    status_emoji = {"success": "✅", "failure": "❌", "warning": "⚠️"}.get(status, "❓")

    report_lines.extend(
        [
            f"- **Status**: {status_emoji} {status.upper()}",
            f"- **Success Rate**: {test_results.get('success_rate', 0):.1%}",
            f"- **Total Tests**: {test_results.get('total_tests', 0)}",
            "",
        ]
    )

    # Detailed results
    if "test_results" in test_results:
        report_lines.extend(
            [
                "## Detailed Test Results",
                "",
                "| Test Case | Status | Details |",
                "|-----------|--------|---------|",
            ]
        )

        for result in test_results["test_results"]:
            status = result.get("status", "unknown")
            status_emoji = {"success": "✅", "failure": "❌", "exception": "💥"}.get(
                status, "❓"
            )

            details = []
            if result.get("has_error"):
                details.append("Has error")
            if not result.get("has_expected_keys", True):
                details.append("Missing keys")
            if result.get("exception"):
                details.append(f"Exception: {result['exception'][:50]}...")

            details_str = "; ".join(details) if details else "OK"

            report_lines.append(
                f"| {result.get('test_case', 'N/A')} | {status_emoji} {status} | {details_str} |"
            )

    # Performance regressions
    if "regressions" in test_results and test_results.get("has_regressions"):
        report_lines.extend(
            [
                "",
                "## Performance Regressions Detected ⚠️",
                "",
                "| Metric | Baseline | Recent | Change |",
                "|--------|----------|--------|--------|",
            ]
        )

        for metric, data in test_results["regressions"].items():
            change_pct = data["change"] * 100
            report_lines.append(
                f"| {metric} | {data['baseline']:.3f} | {data['recent']:.3f} | {change_pct:+.1f}% |"
            )

    report = "\n".join(report_lines)

    # Save to file if requested
    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(report)

    return report


def run_comprehensive_evaluation_tests(
    logger: EvaluationLogger,
    evaluation_func: Callable[..., Any],
    ohlc_data: pd.DataFrame,
) -> Dict[str, Any]:
    """
    Run comprehensive evaluation testing suite

    Args:
        logger: Evaluation logger instance
        evaluation_func: Evaluation function to test
        ohlc_data: OHLC data for testing

    Returns:
        Comprehensive test results
    """
    results = {}

    # 1. Success rate validation
    results["success_rate"] = validate_evaluation_success_rate(logger)

    # 2. Performance regression detection
    results["regressions"] = detect_performance_regressions(logger)

    # 3. Edge case testing
    edge_cases = [
        {
            "args": {
                "feature_class": lambda: None,
                "ohlc_data": ohlc_data.head(10),
                "feature_name": "test",
            },
            "expected_keys": ["status"],
        },
        {
            "args": {
                "feature_class": lambda: None,
                "ohlc_data": pd.DataFrame(),
                "feature_name": "empty",
            },
            "expected_keys": ["status"],
        },
    ]

    results["robustness"] = test_evaluation_pipeline_robustness(
        evaluation_func, edge_cases
    )

    # Overall assessment
    all_passed = all(
        result.get("status") in ["success", "no_data", "insufficient_data"]
        for result in results.values()
        if isinstance(result, dict) and "status" in result
    )

    results["overall_status"] = "success" if all_passed else "failure"  # type: ignore[assignment]

    return results
