#!/usr/bin/env python3
"""
Unit tests for schema validation of trading results.
"""

import json
from datetime import datetime
from pathlib import Path

import jsonschema
import pytest


class TestSchemaValidation:
    """Test schema validation for trading results."""

    @pytest.fixture
    def results_schema(self):
        """Load the results schema."""
        schema_path = Path(__file__).parent.parent / "config" / "results_schema.json"
        with open(schema_path, "r") as f:
            return json.load(f)

    @pytest.fixture
    def minimal_synthetic_results(self):
        """Create minimal synthetic results for testing."""
        return {
            "strategy": "buy_hold",
            "dataset": "btc_usd_1m",
            "slippage_bps": 5.0,
            "initial_capital": 10000.0,
            "total_pnl": 1250.0,
            "sharpe_ratio": 1.25,
            "deflated_sharpe_ratio": 1.15,
            "pvalue_bootstrap": 0.032,
            "max_drawdown": -500.0,
            "win_rate": 0.55,
            "total_trades": 10,
            "trades_per_day": 2.5,
            "duration_minutes": 120.0,
            "budget_analysis": {
                "annual_pnl": 9125.0,
                "trading_costs": 250.0,
                "net_profit": 8875.0,
                "break_even_trades": 400,
                "payback_period_months": 6.7,
                "roi_percentage": 177.5,
                "total_investment": 5000.0,
            },
            "equity_curve": [
                {"timestamp": "2020-01-01T00:00:00", "equity": 10000.0},
                {"timestamp": "2020-01-02T00:00:00", "equity": 11250.0},
            ],
            "orders": [
                {
                    "timestamp": "2020-01-01T00:00:00",
                    "action": "buy",
                    "price": 10000.0,
                    "shares": 1.0,
                    "notional": 10000.0,
                    "position_before": 0,
                    "position_after": 1,
                    "sizing_reason": "All-in position sizing",
                    "pnl": 0.0,
                }
            ],
            "run_metadata": {
                "python_version": "3.11.9",
                "os": "Linux",
                "cpu_model": "Intel(R) Core(TM) i7-9750H CPU @ 2.60GHz",
                "package_hashes": {"pandas": "abc123", "numpy": "def456"},
                "git_sha": "a1b2c3d4",
                "random_seed": 42,
                "timestamp": datetime.now().isoformat(),
            },
        }

    def test_minimal_results_valid(self, results_schema, minimal_synthetic_results):
        """Test that minimal synthetic results pass schema validation."""
        try:
            jsonschema.validate(minimal_synthetic_results, results_schema)
        except jsonschema.ValidationError as e:
            pytest.fail(f"Schema validation failed: {e.message}")

    def test_missing_required_field_fails(
        self, results_schema, minimal_synthetic_results
    ):
        """Test that missing required fields cause validation failure."""
        invalid_results = minimal_synthetic_results.copy()
        del invalid_results["strategy"]

        with pytest.raises(jsonschema.ValidationError):
            jsonschema.validate(invalid_results, results_schema)

    def test_invalid_enum_value_fails(self, results_schema, minimal_synthetic_results):
        """Test that invalid enum values cause validation failure."""
        invalid_results = minimal_synthetic_results.copy()
        invalid_results["strategy"] = "invalid_strategy"

        with pytest.raises(jsonschema.ValidationError):
            jsonschema.validate(invalid_results, results_schema)

    def test_null_statistical_fields_allowed(
        self, results_schema, minimal_synthetic_results
    ):
        """Test that null statistical fields are allowed."""
        results_with_nulls = minimal_synthetic_results.copy()
        results_with_nulls["deflated_sharpe_ratio"] = None
        results_with_nulls["pvalue_bootstrap"] = None

        try:
            jsonschema.validate(results_with_nulls, results_schema)
        except jsonschema.ValidationError as e:
            pytest.fail(f"Schema validation failed for null fields: {e.message}")

    def test_budget_analysis_required_fields(
        self, results_schema, minimal_synthetic_results
    ):
        """Test that budget analysis has all required fields."""
        results_missing_budget_field = minimal_synthetic_results.copy()
        del results_missing_budget_field["budget_analysis"]["annual_pnl"]

        with pytest.raises(jsonschema.ValidationError):
            jsonschema.validate(results_missing_budget_field, results_schema)

    def test_run_metadata_required_fields(
        self, results_schema, minimal_synthetic_results
    ):
        """Test that run metadata has all required fields."""
        results_missing_metadata_field = minimal_synthetic_results.copy()
        del results_missing_metadata_field["run_metadata"]["python_version"]

        with pytest.raises(jsonschema.ValidationError):
            jsonschema.validate(results_missing_metadata_field, results_schema)


if __name__ == "__main__":
    pytest.main([__file__])
