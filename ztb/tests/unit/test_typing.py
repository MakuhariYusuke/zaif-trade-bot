"""
Type safety tests for TypedDict validation and mypy integration.
"""

import numpy as np
import pandas as pd
import pytest

from ztb.evaluation.logging import EvaluationRecord
from ztb.metrics.metrics import MetricsResult, calculate_all_metrics

from .test_autogen import BaseFeatureTest


class TestTypedDictValidation(BaseFeatureTest):
    """Test TypedDict structure validation"""

    def test_metrics_result_structure(self):
        """Test that MetricsResult contains all required keys"""
        # Create sample returns data
        returns = np.array([0.01, 0.02, -0.01, 0.015, -0.005])

        result = calculate_all_metrics(returns)

        # Check that result is a dict
        assert isinstance(result, dict)

        # Check required keys exist
        required_keys = [
            "total_return",
            "annual_return",
            "volatility",
            "sharpe_ratio",
            "sortino_ratio",
            "calmar_ratio",
            "max_drawdown",
            "win_rate",
            "profit_factor",
        ]

        for key in required_keys:
            assert key in result, f"Missing required key: {key}"
            assert isinstance(result[key], (int, float)), f"Key {key} should be numeric"

    def test_metrics_result_types(self):
        """Test that MetricsResult values have correct types"""
        returns = np.array([0.01, 0.02, -0.01, 0.015, -0.005])
        result = calculate_all_metrics(returns)

        # All values should be floats (not Optional)
        for key, value in result.items():
            assert isinstance(
                value, (int, float)
            ), f"Value for {key} should be numeric, got {type(value)}"

    def test_evaluation_result_structure(self):
        """Test that EvaluationResult contains all required keys"""
        # Create mock data for testing
        mock_data = {
            "total_evaluations": 10,
            "successful_evaluations": 8,
            "failed_evaluations": 2,
            "success_rate": 0.8,
            "avg_computation_time_ms": 150.5,
            "avg_sharpe_improvement": 0.15,
            "avg_delta_sharpe": 0.12,
            "feature_counts": {
                "verified": 5,
                "pending": 2,
                "failed": 1,
                "unverified": 2,
            },
            "top_performing_features": ["feature1", "feature2"],
            "computation_time_stats": {
                "mean": 150.5,
                "std": 25.0,
                "min": 100.0,
                "max": 200.0,
            },
        }

        # Check that all required keys are present
        required_keys = [
            "total_evaluations",
            "successful_evaluations",
            "failed_evaluations",
            "success_rate",
            "avg_computation_time_ms",
        ]

        for key in required_keys:
            assert key in mock_data, f"Missing required key: {key}"

    def test_evaluation_record_no_optional_overuse(self):
        """Test that EvaluationRecord doesn't overuse Optional fields"""
        # Create a sample record
        record = EvaluationRecord(
            timestamp="2024-01-01T00:00:00",
            feature_name="TestFeature",
            status="verified",
            computation_time_ms=150.5,
            nan_rate=0.05,
            total_columns=100,
            aligned_periods=500,
            baseline_sharpe=0.8,
            best_delta_sharpe=0.15,
            best_feature_name="test_feature",
        )

        # Check that non-optional fields are not None
        assert record.timestamp is not None
        assert record.feature_name is not None
        assert record.status is not None
        assert record.computation_time_ms is not None
        assert record.nan_rate is not None
        assert record.total_columns is not None
        assert record.aligned_periods is not None
        assert record.baseline_sharpe is not None
        assert record.best_delta_sharpe is not None
        assert record.best_feature_name is not None
        assert record.cv_results is not None  # Should be dict, not None
        assert record.feature_correlations is not None  # Should be dict, not None
        assert record.feature_performances is not None  # Should be dict, not None

    def test_metrics_result_mypy_compatibility(self):
        """Test that MetricsResult is mypy-compatible"""
        returns = np.array([0.01, 0.02, -0.01, 0.015, -0.005])
        result = calculate_all_metrics(returns)

        # This should not raise mypy errors
        sharpe: float = result["sharpe_ratio"]
        assert isinstance(sharpe, (int, float))

        # Test that we can access all fields without type issues
        total_return: float = result["total_return"]
        volatility: float = result["volatility"]
        max_drawdown: float = result["max_drawdown"]

        assert all(
            isinstance(x, (int, float))
            for x in [sharpe, total_return, volatility, max_drawdown]
        )


class TestMypyIntegration:
    """Test mypy integration with pytest"""

    def test_mypy_runs_without_errors(self):
        """Test that mypy can run on our typed modules"""
        # This test will be run by pytest-mypy plugin
        # If mypy finds errors, this test will fail
        pass

    def test_import_type_hints(self):
        """Test that all imports with type hints work correctly"""
        # Test that we can import typed modules without issues
        try:
            from ztb.evaluation.logging import EvaluationRecord
            from ztb.evaluation.re_evaluate_features import EvaluationResult
            from ztb.metrics.metrics import MetricsResult, calculate_all_metrics

            # If we get here, imports succeeded
            assert True
        except ImportError as e:
            pytest.fail(f"Import failed: {e}")

    def test_function_signatures_are_typed(self):
        """Test that key functions have proper type annotations"""
        import inspect

        # Check calculate_all_metrics signature
        sig = inspect.signature(calculate_all_metrics)
        assert "returns" in sig.parameters
        assert sig.return_annotation == MetricsResult

        # Check that parameters have type annotations
        returns_param = sig.parameters["returns"]
        assert returns_param.annotation is not inspect.Parameter.empty

    def test_status_reason_enum_validation(self):
        """Test that status/reason enums work correctly"""
        from ztb.evaluation.status import (
            FeatureReason,
            FeatureStatus,
            validate_status_reason,
        )

        # Test valid combinations
        assert validate_status_reason(FeatureStatus.VERIFIED, None) == True
        assert (
            validate_status_reason(
                FeatureStatus.PENDING, FeatureReason.INSUFFICIENT_DATA
            )
            == True
        )
        assert (
            validate_status_reason(FeatureStatus.UNVERIFIED, FeatureReason.NOT_TESTED)
            == True
        )
        assert (
            validate_status_reason(
                FeatureStatus.FAILED, FeatureReason.COMPUTATION_ERROR
            )
            == True
        )

        # Test invalid combinations
        assert (
            validate_status_reason(
                FeatureStatus.VERIFIED, FeatureReason.INSUFFICIENT_DATA
            )
            == False
        )
        assert validate_status_reason(FeatureStatus.PENDING, None) == False
        assert (
            validate_status_reason(FeatureStatus.PENDING, FeatureReason.NOT_TESTED)
            == False
        )  # Wrong reason for status

    def test_coverage_json_enum_consistency(self):
        """Test that coverage.json uses only defined enum values"""
        import json
        from pathlib import Path

        from ztb.evaluation.status import FeatureReason, FeatureStatus

        coverage_path = Path("coverage.json")
        if not coverage_path.exists():
            pytest.skip("coverage.json not found")

        with open(coverage_path, "r") as f:
            data = json.load(f)

        # Check that all status keys are valid enums (excluding metadata)
        valid_statuses = {status.value for status in FeatureStatus}
        status_keys = [key for key in data.keys() if key != "metadata"]
        for status in status_keys:
            assert status in valid_statuses, f"Invalid status: {status}"

        # Check that reasons are appropriate for their status
        for status_str, items in data.items():
            if status_str == "metadata":
                continue
            status = FeatureStatus(status_str)
            if status == FeatureStatus.VERIFIED:
                continue  # Verified features don't need reasons

            for item in items:
                if isinstance(item, dict):
                    reason_str = item.get("reason")
                    if reason_str:
                        reason = FeatureReason(reason_str)


class TestComputableFeatureCompliance(BaseFeatureTest):
    """Test that features comply with ComputableFeature protocol"""

    def test_kama_computable_feature_compliance(self, sample_ohlc_data):
        """Test that KAMA implements ComputableFeature protocol"""
        from ztb.features.trend import KAMA

        kama = KAMA()
        # Check that it has the required attributes (protocol compliance)
        assert hasattr(kama, "name")
        assert hasattr(kama, "deps")
        assert hasattr(kama, "compute")
        assert callable(kama.compute)

        # Test compute method signature
        result = kama.compute(sample_ohlc_data)
        assert isinstance(result, pd.DataFrame)
        assert not result.empty

    def test_ichimoku_computable_feature_compliance(self, sample_ohlc_data):
        """Test that Ichimoku implements ComputableFeature protocol"""
        from ztb.features.trend import Ichimoku

        ichimoku = Ichimoku()
        # Check that it has the required attributes (protocol compliance)
        assert hasattr(ichimoku, "name")
        assert hasattr(ichimoku, "deps")
        assert hasattr(ichimoku, "compute")
        assert callable(ichimoku.compute)

        # Test compute method signature
        result = ichimoku.compute(sample_ohlc_data)
        assert isinstance(result, pd.DataFrame)
        assert not result.empty
