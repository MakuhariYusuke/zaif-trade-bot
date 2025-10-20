#!/usr/bin/env python
"""
Unit tests for preflight_schema_scaler_check.py script.

Tests the validation logic for feature schema, normalization stats,
and config fingerprint files.
"""

import sys
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest

# Add scripts directory to path
REPO_ROOT = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from preflight_schema_scaler_check import (  # noqa: E402
    check_config_fingerprint,
    check_feature_schema,
    check_normalization_stats,
    compare_with_training,
)


@pytest.fixture
def temp_model_dir(tmp_path):
    """Create a temporary model directory."""
    model_dir = tmp_path / "models" / "test_run"
    model_dir.mkdir(parents=True)
    return model_dir


@pytest.fixture
def mock_feature_schema(temp_model_dir):
    """Create a mock feature schema file."""
    import pandas as pd

    from ztb.utils.feature_schema import FeaturesSchema

    # Create a mock DataFrame
    df = pd.DataFrame(
        {
            "feature1": [0.5] * 10,
            "feature2": [1.0] * 10,
            "feature3": [-0.5] * 10,
            "ts": range(10),  # Meta column
        }
    )

    # Use from_dataframe factory method
    schema = FeaturesSchema.from_dataframe(
        df, feature_columns=["feature1", "feature2", "feature3"]
    )

    # Save to file, not directory
    schema.save(temp_model_dir / "features_schema.json")
    return schema


@pytest.fixture
def mock_normalization_stats(temp_model_dir):
    """Create a mock normalization stats file."""
    from ztb.utils.normalization import NormalizationStats, save_scaler

    stats = NormalizationStats(
        feature_names=["feature1", "feature2", "feature3"],
        mean=np.array([0.5, 1.0, -0.5]),
        std=np.array([0.1, 0.2, 0.3]),
        n_samples=1000,
        version="1.0.0",
    )

    save_scaler(temp_model_dir, stats)
    return stats


@pytest.fixture
def mock_config_fingerprint(temp_model_dir):
    """Create a mock config fingerprint file."""
    from ztb.utils.config_fingerprint import ConfigFingerprint

    fingerprint = ConfigFingerprint(
        initial_portfolio_value=100000.0,
        transaction_cost=0.001,
        max_position_size=1.0,
        risk_free_rate=0.0,
        timeframe="1h",
        feature_set="extended",
        curriculum_stage="main",
        reward_scaling=1.0,
        reward_settings={},
    )

    # Save to file, not directory
    fingerprint.save(temp_model_dir / "config_fingerprint.json")
    return fingerprint


class TestCheckFeatureSchema:
    """Test feature schema validation."""

    def test_schema_exists_and_valid(self, temp_model_dir, mock_feature_schema):
        """Test validation with valid schema file."""
        success, message = check_feature_schema(temp_model_dir, strict=True)

        assert success is True
        assert "✅" in message
        assert "Feature schema valid" in message
        assert "hash:" in message

    def test_schema_missing(self, temp_model_dir):
        """Test validation with missing schema file."""
        success, message = check_feature_schema(temp_model_dir, strict=True)

        assert success is False
        assert "❌" in message
        assert "not found" in message

    def test_schema_invalid_content(self, temp_model_dir):
        """Test validation with invalid schema content."""
        # Create invalid JSON file
        schema_path = temp_model_dir / "features_schema.json"
        schema_path.write_text("invalid json content")

        success, message = check_feature_schema(temp_model_dir, strict=True)

        assert success is False
        assert "❌" in message
        assert "validation failed" in message


class TestCheckNormalizationStats:
    """Test normalization stats validation."""

    def test_stats_exists_and_valid(self, temp_model_dir, mock_normalization_stats):
        """Test validation with valid stats file."""
        success, message = check_normalization_stats(temp_model_dir, strict=True)

        assert success is True
        assert "✅" in message
        assert "Normalization stats valid" in message
        assert "hash:" in message
        assert "features: 3" in message
        assert "samples: 1000" in message

    def test_stats_missing(self, temp_model_dir):
        """Test validation with missing stats file."""
        success, message = check_normalization_stats(temp_model_dir, strict=True)

        assert success is False
        assert "❌" in message
        assert "not found" in message

    def test_stats_corrupted(self, temp_model_dir):
        """Test validation with corrupted stats file."""
        # Create corrupted npz file
        scaler_path = temp_model_dir / "scaler.npz"
        scaler_path.write_bytes(b"corrupted data")

        success, message = check_normalization_stats(temp_model_dir, strict=True)

        assert success is False
        assert "❌" in message
        assert "validation failed" in message


class TestCheckConfigFingerprint:
    """Test config fingerprint validation."""

    def test_fingerprint_exists_and_valid(
        self, temp_model_dir, mock_config_fingerprint
    ):
        """Test validation with valid fingerprint file."""
        success, message = check_config_fingerprint(temp_model_dir, strict=True)

        assert success is True
        assert "✅" in message
        assert "Config fingerprint valid" in message
        assert "hash:" in message
        assert "feature_set:" in message

    def test_fingerprint_missing_strict(self, temp_model_dir):
        """Test validation with missing fingerprint (strict mode)."""
        success, message = check_config_fingerprint(temp_model_dir, strict=True)

        assert success is False
        assert "❌" in message
        assert "not found" in message

    def test_fingerprint_missing_non_strict(self, temp_model_dir):
        """Test validation with missing fingerprint (non-strict mode)."""
        success, message = check_config_fingerprint(temp_model_dir, strict=False)

        # Should succeed with warning in non-strict mode
        assert success is True
        assert "⚠️" in message
        assert "not found" in message

    def test_fingerprint_invalid_content(self, temp_model_dir):
        """Test validation with invalid fingerprint content."""
        # Create invalid JSON file
        fp_path = temp_model_dir / "config_fingerprint.json"
        fp_path.write_text("invalid json")

        success, message = check_config_fingerprint(temp_model_dir, strict=True)

        assert success is False
        assert "❌" in message
        assert "validation failed" in message


class TestCompareWithTraining:
    """Test train/test data comparison."""

    def test_no_test_data(self, temp_model_dir, mock_normalization_stats):
        """Test comparison when no test data provided."""
        success, message = compare_with_training(temp_model_dir, test_data_path=None)

        assert success is True
        assert "ℹ️" in message
        assert "not provided" in message

    def test_test_data_missing(self, temp_model_dir, mock_normalization_stats):
        """Test comparison when test data file missing."""
        test_path = temp_model_dir / "nonexistent.csv"

        success, message = compare_with_training(
            temp_model_dir, test_data_path=test_path
        )

        assert success is True
        assert "ℹ️" in message
        assert "not provided" in message

    def test_comparison_success(
        self, temp_model_dir, mock_normalization_stats, tmp_path
    ):
        """Test successful train/test comparison."""
        import pandas as pd

        # Create test data with similar stats
        test_data = pd.DataFrame(
            {
                "feature1": np.random.normal(0.5, 0.1, 100),
                "feature2": np.random.normal(1.0, 0.2, 100),
                "feature3": np.random.normal(-0.5, 0.3, 100),
                "ts": range(100),  # Meta column
            }
        )

        test_path = tmp_path / "test_data.csv"
        test_data.to_csv(test_path, index=False)

        success, message = compare_with_training(
            temp_model_dir, test_data_path=test_path
        )

        assert success is True
        assert "✅" in message or "⚠️" in message
        assert "mean Δ=" in message
        assert "std Δ=" in message

    def test_comparison_large_difference(
        self, temp_model_dir, mock_normalization_stats, tmp_path
    ):
        """Test comparison with large train/test difference."""
        import pandas as pd

        # Create test data with very different stats
        test_data = pd.DataFrame(
            {
                "feature1": np.random.normal(5.0, 1.0, 100),  # Very different
                "feature2": np.random.normal(10.0, 2.0, 100),  # Very different
                "feature3": np.random.normal(-5.0, 3.0, 100),  # Very different
                "ts": range(100),
            }
        )

        test_path = tmp_path / "test_data.csv"
        test_data.to_csv(test_path, index=False)

        success, message = compare_with_training(
            temp_model_dir, test_data_path=test_path
        )

        assert success is True  # Still succeeds but with warning
        assert "⚠️" in message
        assert "Large train/test difference" in message

    def test_feature_mismatch(self, temp_model_dir, mock_normalization_stats, tmp_path):
        """Test comparison with feature mismatch."""
        import pandas as pd

        # Create test data with different features
        test_data = pd.DataFrame(
            {
                "feature1": np.random.normal(0.5, 0.1, 100),
                "feature4": np.random.normal(1.0, 0.2, 100),  # Wrong feature name
                "ts": range(100),
            }
        )

        test_path = tmp_path / "test_data.csv"
        test_data.to_csv(test_path, index=False)

        success, message = compare_with_training(
            temp_model_dir, test_data_path=test_path
        )

        assert success is False
        assert "❌" in message
        assert "Feature mismatch" in message


class TestMainFunction:
    """Test main CLI function."""

    def test_all_checks_pass(
        self,
        temp_model_dir,
        mock_feature_schema,
        mock_normalization_stats,
        mock_config_fingerprint,
    ):
        """Test main with all checks passing."""
        from preflight_schema_scaler_check import main

        with patch("sys.argv", ["script", "--model-dir", str(temp_model_dir)]):
            exit_code = main()

        assert exit_code == 0

    def test_missing_model_dir(self, tmp_path):
        """Test main with missing model directory."""
        from preflight_schema_scaler_check import main

        nonexistent = tmp_path / "nonexistent"

        with patch("sys.argv", ["script", "--model-dir", str(nonexistent)]):
            exit_code = main()

        assert exit_code == 2

    def test_schema_check_fails(self, temp_model_dir, mock_normalization_stats):
        """Test main when schema check fails."""
        from preflight_schema_scaler_check import main

        # Only stats exist, schema missing
        with patch("sys.argv", ["script", "--model-dir", str(temp_model_dir)]):
            exit_code = main()

        assert exit_code == 1

    def test_non_strict_mode(
        self, temp_model_dir, mock_feature_schema, mock_normalization_stats
    ):
        """Test main in non-strict mode."""
        from preflight_schema_scaler_check import main

        # Missing config fingerprint, but non-strict should pass
        with patch(
            "sys.argv", ["script", "--model-dir", str(temp_model_dir), "--no-strict"]
        ):
            exit_code = main()

        assert exit_code == 0
