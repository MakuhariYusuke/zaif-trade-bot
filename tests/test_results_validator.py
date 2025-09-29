"""
Tests for unified results schema validator.
"""

import json
from pathlib import Path

import pytest

from schema.results_validator import ResultsValidator


class TestResultsValidator:
    """Test cases for ResultsValidator."""

    @pytest.fixture
    def validator(self):
        """Create validator instance."""
        return ResultsValidator()

    @pytest.fixture
    def valid_results(self):
        """Sample valid results data."""
        return {
            "metadata": {
                "version": "1.0.0",
                "timestamp": "2024-01-01T00:00:00Z",
                "run_id": "test-run-123",
                "type": "backtest",
                "config": {"param": "value"},
            },
            "performance": {
                "total_return": 0.15,
                "sharpe_ratio": 1.5,
                "max_drawdown": 0.1,
                "win_rate": 0.6,
                "total_trades": 100,
                "profit_factor": 1.2,
            },
        }

    def test_valid_results(self, validator, valid_results):
        """Test validation of valid results."""
        assert validator.validate(valid_results) is True

    def test_invalid_version_format(self, validator, valid_results):
        """Test validation fails with invalid version format."""
        invalid_results = valid_results.copy()
        invalid_results["metadata"]["version"] = "1.0"  # Missing patch version
        assert validator.validate(invalid_results) is False

    def test_missing_required_field(self, validator, valid_results):
        """Test validation fails with missing required field."""
        invalid_results = valid_results.copy()
        del invalid_results["metadata"]["run_id"]
        assert validator.validate(invalid_results) is False

    def test_invalid_enum_value(self, validator, valid_results):
        """Test validation fails with invalid enum value."""
        invalid_results = valid_results.copy()
        invalid_results["metadata"]["type"] = "invalid_type"
        assert validator.validate(invalid_results) is False

    def test_invalid_number_range(self, validator, valid_results):
        """Test validation fails with number outside allowed range."""
        invalid_results = valid_results.copy()
        invalid_results["performance"]["win_rate"] = 1.5  # Should be <= 1.0
        assert validator.validate(invalid_results) is False

    def test_validation_errors_detail(self, validator):
        """Test getting detailed validation errors."""
        invalid_results = {"invalid": "data"}
        errors = validator.get_validation_errors(invalid_results)
        assert len(errors) > 0
        assert "message" in errors[0]
        assert "path" in errors[0]

    def test_validate_file_valid(self, validator, valid_results, tmp_path):
        """Test validating valid results from file."""
        results_file = tmp_path / "results.json"
        with open(results_file, "w") as f:
            json.dump(valid_results, f)

        assert validator.validate_file(results_file) is True

    def test_validate_file_invalid_json(self, validator, tmp_path):
        """Test validating invalid JSON file."""
        results_file = tmp_path / "invalid.json"
        with open(results_file, "w") as f:
            f.write("invalid json")

        assert validator.validate_file(results_file) is False

    def test_validate_file_not_found(self, validator):
        """Test validating non-existent file."""
        assert validator.validate_file(Path("nonexistent.json")) is False
