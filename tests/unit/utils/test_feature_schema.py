#!/usr/bin/env python3
"""Tests for feature schema validation."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from ztb.utils.feature_schema import (
    FeaturesSchema,
    create_and_save_schema,
    load_and_validate_schema,
)


class TestFeaturesSchema:
    """Tests for FeaturesSchema class."""

    @pytest.fixture
    def sample_df(self) -> pd.DataFrame:
        """Create sample DataFrame for testing."""
        return pd.DataFrame(
            {
                "ts": [1000, 2000, 3000],
                "price": [100.0, 101.0, 102.0],
                "volume": [1000.0, 1100.0, 1200.0],
                "rsi": [50.0, 55.0, 60.0],
                "macd": [0.1, 0.2, 0.3],
                "pair": ["BTC/JPY", "BTC/JPY", "BTC/JPY"],
            }
        )

    @pytest.fixture
    def feature_columns(self) -> list[str]:
        """Feature column names."""
        return ["price", "volume", "rsi", "macd"]

    def test_from_dataframe(self, sample_df: pd.DataFrame, feature_columns: list[str]) -> None:
        """Test schema creation from DataFrame."""
        schema = FeaturesSchema.from_dataframe(sample_df, feature_columns)

        assert schema.columns == feature_columns
        assert len(schema.dtypes) == 4
        assert schema.order_hash
        assert len(schema.statistics) == 4
        assert "price" in schema.statistics
        assert "mean" in schema.statistics["price"]

    def test_auto_detect_features(self, sample_df: pd.DataFrame) -> None:
        """Test automatic feature column detection."""
        schema = FeaturesSchema.from_dataframe(sample_df, feature_columns=None)

        # Should detect numeric columns, excluding 'ts' and 'pair'
        assert "price" in schema.columns
        assert "volume" in schema.columns
        assert "rsi" in schema.columns
        assert "macd" in schema.columns
        assert "ts" not in schema.columns  # excluded
        assert "pair" not in schema.columns  # non-numeric

    def test_compute_hash(self, sample_df: pd.DataFrame, feature_columns: list[str]) -> None:
        """Test schema hash computation."""
        schema1 = FeaturesSchema.from_dataframe(sample_df, feature_columns)
        schema2 = FeaturesSchema.from_dataframe(sample_df, feature_columns)

        # Same data should produce same hash
        assert schema1.compute_hash() == schema2.compute_hash()

        # Different column order should produce different hash
        schema3 = FeaturesSchema.from_dataframe(
            sample_df, ["rsi", "macd", "price", "volume"]
        )
        assert schema1.compute_hash() != schema3.compute_hash()

    def test_validate_dataframe_success(
        self, sample_df: pd.DataFrame, feature_columns: list[str]
    ) -> None:
        """Test successful DataFrame validation."""
        schema = FeaturesSchema.from_dataframe(sample_df, feature_columns)

        # Should validate successfully
        is_valid, errors = schema.validate_dataframe(sample_df, feature_columns, strict=False)
        assert is_valid
        assert len(errors) == 0

    def test_validate_dataframe_missing_columns(
        self, sample_df: pd.DataFrame, feature_columns: list[str]
    ) -> None:
        """Test validation with missing columns."""
        schema = FeaturesSchema.from_dataframe(sample_df, feature_columns)

        # Drop a column
        df_missing = sample_df.drop(columns=["rsi"])

        is_valid, errors = schema.validate_dataframe(df_missing, feature_columns, strict=False)
        assert not is_valid
        assert any("Missing columns" in err for err in errors)

    def test_validate_dataframe_dtype_mismatch(
        self, sample_df: pd.DataFrame, feature_columns: list[str]
    ) -> None:
        """Test validation with dtype mismatch."""
        schema = FeaturesSchema.from_dataframe(sample_df, feature_columns)

        # Change dtype
        df_wrong_dtype = sample_df.copy()
        df_wrong_dtype["rsi"] = df_wrong_dtype["rsi"].astype(int)

        is_valid, errors = schema.validate_dataframe(
            df_wrong_dtype, feature_columns, strict=False
        )
        # Should still be valid (int and float are compatible)
        assert is_valid or any("Dtype mismatch" in err for err in errors)

    def test_validate_dataframe_strict_mode(
        self, sample_df: pd.DataFrame, feature_columns: list[str]
    ) -> None:
        """Test strict validation mode (raises on failure)."""
        schema = FeaturesSchema.from_dataframe(sample_df, feature_columns)

        # Drop a column
        df_missing = sample_df.drop(columns=["rsi"])

        with pytest.raises(ValueError, match="Feature schema validation failed"):
            schema.validate_dataframe(df_missing, feature_columns, strict=True)

    def test_diff(self, sample_df: pd.DataFrame) -> None:
        """Test schema diff computation."""
        schema1 = FeaturesSchema.from_dataframe(sample_df, ["price", "volume", "rsi"])
        schema2 = FeaturesSchema.from_dataframe(sample_df, ["price", "volume", "macd"])

        diff = schema1.diff(schema2)

        assert "rsi" in diff["columns_removed"]
        assert "macd" in diff["columns_added"]

    def test_save_and_load(
        self, sample_df: pd.DataFrame, feature_columns: list[str], tmp_path: Path
    ) -> None:
        """Test saving and loading schema."""
        schema = FeaturesSchema.from_dataframe(sample_df, feature_columns)

        # Save
        schema_path = tmp_path / "features_schema.json"
        schema.save(schema_path)

        assert schema_path.exists()

        # Load
        loaded_schema = FeaturesSchema.load(schema_path)

        assert loaded_schema.columns == schema.columns
        assert loaded_schema.dtypes == schema.dtypes
        assert loaded_schema.order_hash == schema.order_hash
        assert loaded_schema.compute_hash() == schema.compute_hash()

    def test_load_missing_file(self, tmp_path: Path) -> None:
        """Test loading from non-existent file."""
        with pytest.raises(FileNotFoundError):
            FeaturesSchema.load(tmp_path / "nonexistent.json")

    def test_load_and_validate_success(
        self, sample_df: pd.DataFrame, feature_columns: list[str], tmp_path: Path
    ) -> None:
        """Test load_and_validate with matching data."""
        schema = FeaturesSchema.from_dataframe(sample_df, feature_columns)
        schema_path = tmp_path / "features_schema.json"
        schema.save(schema_path)

        # Should succeed
        loaded = FeaturesSchema.load_and_validate(
            schema_path, sample_df, feature_columns, strict=True
        )
        assert loaded.compute_hash() == schema.compute_hash()

    def test_load_and_validate_failure(
        self, sample_df: pd.DataFrame, feature_columns: list[str], tmp_path: Path
    ) -> None:
        """Test load_and_validate with mismatched data."""
        schema = FeaturesSchema.from_dataframe(sample_df, feature_columns)
        schema_path = tmp_path / "features_schema.json"
        schema.save(schema_path)

        # Drop a column
        df_wrong = sample_df.drop(columns=["rsi"])

        with pytest.raises(ValueError, match="Feature schema validation failed"):
            FeaturesSchema.load_and_validate(
                schema_path, df_wrong, feature_columns, strict=True
            )

    def test_convenience_functions(
        self, sample_df: pd.DataFrame, feature_columns: list[str], tmp_path: Path
    ) -> None:
        """Test convenience functions."""
        # Create and save
        schema = create_and_save_schema(sample_df, tmp_path, feature_columns)
        assert (tmp_path / "features_schema.json").exists()

        # Load and validate
        loaded = load_and_validate_schema(tmp_path, sample_df, feature_columns, strict=True)
        assert loaded.compute_hash() == schema.compute_hash()

    def test_column_order_mismatch_detection(
        self, sample_df: pd.DataFrame, tmp_path: Path
    ) -> None:
        """Test detection of column order changes."""
        # Create schema with specific order
        schema1 = FeaturesSchema.from_dataframe(sample_df, ["price", "volume", "rsi", "macd"])
        schema_path = tmp_path / "features_schema.json"
        schema1.save(schema_path)

        # Load schema
        loaded_schema = FeaturesSchema.load(schema_path)

        # Pass different column order in feature_columns argument
        # (The order in feature_columns parameter is what matters for validation)
        is_valid, errors = loaded_schema.validate_dataframe(
            sample_df, ["rsi", "price", "macd", "volume"], strict=False
        )

        # Order mismatch should be detected
        assert not is_valid
        assert any("Column order mismatch" in err for err in errors)

    def test_statistics_computation(
        self, sample_df: pd.DataFrame, feature_columns: list[str]
    ) -> None:
        """Test statistics computation."""
        schema = FeaturesSchema.from_dataframe(sample_df, feature_columns, compute_stats=True)

        # Check statistics exist
        assert len(schema.statistics) == 4
        assert "price" in schema.statistics
        
        # Check statistics values
        price_stats = schema.statistics["price"]
        assert "mean" in price_stats
        assert "std" in price_stats
        assert "min" in price_stats
        assert "max" in price_stats

        # Check reasonable values
        assert price_stats["mean"] == pytest.approx(101.0, rel=1e-2)
        assert price_stats["min"] == 100.0
        assert price_stats["max"] == 102.0
