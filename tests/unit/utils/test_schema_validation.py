#!/usr/bin/env python3
"""
Unit tests for schema validation of trading results.
"""

import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict

import jsonschema
import pytest


class TestConfig:
    """Test configuration management."""

    @staticmethod
    def get_test_config() -> Dict[str, Any]:
        """Get test configuration with environment overrides."""
        base_config = {
            "test_data_dir": Path(__file__).parent / "test_data",
            "mock_api_responses": True,
            "test_timeout": 30,
            "parallel_tests": os.getenv("PARALLEL_TESTS", "false").lower() == "true",
        }

        # Environment-specific overrides
        if os.getenv("CI"):
            base_config.update(
                {
                    "test_timeout": 60,
                    "mock_api_responses": True,
                }
            )

        return base_config

    @staticmethod
    def create_test_data_dir() -> Path:
        """Create and return test data directory."""
        test_dir = TestConfig.get_test_config()["test_data_dir"]
        test_dir.mkdir(exist_ok=True)
        return test_dir


class TestSchemaValidation:
    """Test schema validation for trading results."""

    @pytest.fixture
    def results_schema(self):
        """Load the results schema."""
        repo_root = Path(__file__).resolve().parents[3]
        candidates = [
            repo_root / "schema" / "results_schema.json",
            repo_root / "configs" / "schema" / "results_schema.json",
            repo_root / "configs" / "results_schema.json",
        ]
        schema_path = next((path for path in candidates if path.exists()), None)
        if schema_path is None:
            pytest.fail("results_schema.json not found in expected locations")
        with open(schema_path, "r", encoding="utf-8") as f:
            return json.load(f)

    @pytest.fixture
    def minimal_synthetic_results(self):
        """Create minimal synthetic results for testing."""
        return {
            "metadata": {
                "version": "1.0.0",
                "timestamp": datetime.now().isoformat(),
                "run_id": "test-run-001",
                "type": "backtest",
                "config": {"symbol": "BTC_JPY"},
            },
            "performance": {
                "total_return": 0.125,
                "sharpe_ratio": 1.25,
                "max_drawdown": 0.05,
                "win_rate": 0.55,
                "total_trades": 10,
                "profit_factor": 1.6,
            },
            "risk": {
                "value_at_risk": -0.02,
                "expected_shortfall": -0.03,
                "beta": 1.05,
                "volatility": 0.12,
            },
            "trades": [
                {
                    "id": "trade-001",
                    "timestamp": "2020-01-01T00:00:00",
                    "symbol": "BTC_JPY",
                    "side": "buy",
                    "quantity": 1.0,
                    "price": 10000.0,
                    "fee": 10.0,
                    "pnl": 0.0,
                }
            ],
            "metrics": {
                "deflated_sharpe_ratio": 1.15,
                "pvalue_bootstrap": 0.032,
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
        del invalid_results["metadata"]

        with pytest.raises(jsonschema.ValidationError):
            jsonschema.validate(invalid_results, results_schema)

    def test_invalid_enum_value_fails(self, results_schema, minimal_synthetic_results):
        """Test that invalid enum values cause validation failure."""
        invalid_results = minimal_synthetic_results.copy()
        invalid_results["metadata"] = dict(invalid_results["metadata"])
        invalid_results["metadata"]["type"] = "invalid_type"

        with pytest.raises(jsonschema.ValidationError):
            jsonschema.validate(invalid_results, results_schema)

    def test_null_metrics_values_fail(
        self, results_schema, minimal_synthetic_results
    ):
        """Current schema does not allow null values in metrics payloads."""
        results_with_nulls = minimal_synthetic_results.copy()
        results_with_nulls["metrics"] = dict(results_with_nulls["metrics"])
        results_with_nulls["metrics"]["deflated_sharpe_ratio"] = None

        with pytest.raises(jsonschema.ValidationError):
            jsonschema.validate(results_with_nulls, results_schema)

    def test_trade_required_fields(
        self, results_schema, minimal_synthetic_results
    ):
        """Trades must include required core fields."""
        results_missing_trade_field = minimal_synthetic_results.copy()
        results_missing_trade_field["trades"] = [dict(minimal_synthetic_results["trades"][0])]
        del results_missing_trade_field["trades"][0]["id"]

        with pytest.raises(jsonschema.ValidationError):
            jsonschema.validate(results_missing_trade_field, results_schema)

    def test_metadata_required_fields(
        self, results_schema, minimal_synthetic_results
    ):
        """Metadata has the required core fields."""
        results_missing_metadata_field = minimal_synthetic_results.copy()
        results_missing_metadata_field["metadata"] = dict(
            minimal_synthetic_results["metadata"]
        )
        del results_missing_metadata_field["metadata"]["run_id"]

        with pytest.raises(jsonschema.ValidationError):
            jsonschema.validate(results_missing_metadata_field, results_schema)


if __name__ == "__main__":
    pytest.main([__file__])
