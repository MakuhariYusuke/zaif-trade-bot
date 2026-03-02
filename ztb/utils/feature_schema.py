#!/usr/bin/env python3
"""
Feature Schema Validation and Persistence.

Ensures that training and evaluation use identical feature schemas to prevent
silent data corruption and model input mismatches.

Critical Requirements:
1. Save schema during training: column names, dtypes, order, basic statistics
2. Load and validate during evaluation: FAIL immediately on any mismatch
3. SHA256 hash for quick comparison
4. Detailed diff reporting when mismatches occur

Usage:
    # During training
    schema = FeaturesSchema.from_dataframe(training_df, feature_columns)
    schema.save(model_dir / "features_schema.json")

    # During evaluation
    schema = FeaturesSchema.load(model_dir / "features_schema.json")
    schema.validate_dataframe(eval_df, feature_columns, strict=True)
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd

from ztb.utils.file_utils import safe_json_dump
from ztb.utils.logging_utils import get_logger

logger = get_logger(__name__)

@dataclass
class FeaturesSchema:
    """Feature schema with validation capabilities."""

    columns: list[str]
    dtypes: dict[str, str]
    order_hash: str  # SHA256 of column order
    statistics: dict[str, dict[str, float]] = field(default_factory=dict)
    created_at: str | None = None
    version: str = "1.0"

    @classmethod
    def from_dataframe(
        cls,
        df: pd.DataFrame,
        feature_columns: list[str] | None = None,
        compute_stats: bool = True,
    ) -> FeaturesSchema:
        """
        Create schema from DataFrame.

        Args:
            df: Input DataFrame
            feature_columns: list of feature column names (if None, use all numeric columns)
            compute_stats: Whether to compute basic statistics (mean, std, min, max)

        Returns:
            FeaturesSchema instance
        """
        if feature_columns is None:
            # Auto-detect numeric feature columns
            exclude = {
                "ts",
                "timestamp",
                "exchange",
                "pair",
                "episode_id",
                "side",
                "source",
            }
            feature_columns = [
                col
                for col in df.columns
                if col not in exclude and pd.api.types.is_numeric_dtype(df[col])
            ]

        # Ensure columns exist
        missing = [col for col in feature_columns if col not in df.columns]
        if missing:
            raise ValueError(f"Columns missing from DataFrame: {missing}")

        # Extract dtypes
        dtypes = {col: str(df[col].dtype) for col in feature_columns}

        # Compute order hash
        order_str = ",".join(feature_columns)
        order_hash = hashlib.sha256(order_str.encode()).hexdigest()

        # Compute statistics
        statistics: dict[str, dict[str, float]] = {}
        if compute_stats:
            for col in feature_columns:
                series = df[col].dropna()
                if len(series) == 0:
                    statistics[col] = {"mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0}
                else:
                    statistics[col] = {
                        "mean": float(series.mean()),
                        "std": float(series.std()),
                        "min": float(series.min()),
                        "max": float(series.max()),
                    }

        return cls(
            columns=feature_columns,
            dtypes=dtypes,
            order_hash=order_hash,
            statistics=statistics,
            created_at=pd.Timestamp.now().isoformat(),
        )

    def compute_hash(self) -> str:
        """
        Compute overall schema hash (SHA256).

        Includes: columns, dtypes, order
        Excludes: statistics (to allow minor numerical differences)

        Returns:
            SHA256 hex digest
        """
        normalized = {
            "columns": self.columns,
            "dtypes": self.dtypes,
            "order_hash": self.order_hash,
        }
        schema_str = json.dumps(normalized, sort_keys=True)
        return hashlib.sha256(schema_str.encode()).hexdigest()

    def validate_dataframe(
        self,
        df: pd.DataFrame,
        feature_columns: list[str] | None = None,
        strict: bool = True,
        tolerance: float = 0.1,
    ) -> tuple[bool, list[str]]:
        """
        Validate DataFrame against schema.

        Args:
            df: DataFrame to validate
            feature_columns: Expected feature columns (if None, use schema columns)
            strict: If True, raise ValueError on mismatch; if False, return (False, errors)
            tolerance: Tolerance for statistical comparisons (fraction, e.g., 0.1 = 10%)

        Returns:
            (is_valid, error_messages)

        Raises:
            ValueError: If strict=True and validation fails
        """
        if feature_columns is None:
            feature_columns = self.columns

        errors: list[str] = []

        # Check column presence
        missing = [col for col in feature_columns if col not in df.columns]
        if missing:
            errors.append(f"Missing columns: {missing}")

        extra = [
            col
            for col in df.columns
            if col in feature_columns and col not in self.columns
        ]
        if extra:
            errors.append(f"Extra columns (not in schema): {extra}")

        # Check column order (only if all columns are present)
        present_cols = [col for col in feature_columns if col in df.columns]
        if len(present_cols) == len(self.columns) and set(present_cols) == set(
            self.columns
        ):
            # All columns present, check order
            if present_cols != self.columns:
                errors.append(
                    f"Column order mismatch. Expected: {self.columns[:5]}..., "
                    f"Got: {present_cols[:5]}..."
                )

        # Check dtypes
        for col in present_cols:
            expected_dtype = self.dtypes.get(col)
            actual_dtype = str(df[col].dtype)
            if expected_dtype and actual_dtype != expected_dtype:
                # Allow compatible dtypes (e.g., float32 vs float64)
                if not self._dtypes_compatible(expected_dtype, actual_dtype):
                    errors.append(
                        f"Dtype mismatch for '{col}': expected {expected_dtype}, got {actual_dtype}"
                    )

        # Check statistics (warning only, not strict error)
        if self.statistics:
            for col in present_cols:
                if col not in self.statistics:
                    continue

                series = df[col].dropna()
                if len(series) == 0:
                    continue

                expected = self.statistics[col]
                actual_mean = float(series.mean())
                actual_std = float(series.std())

                # Check if statistics are wildly different (>tolerance)
                if expected["std"] > 0:
                    mean_diff = abs(actual_mean - expected["mean"]) / expected["std"]
                    if mean_diff > tolerance:
                        errors.append(
                            f"Statistics mismatch for '{col}': mean differs by {mean_diff:.2f} std "
                            f"(expected {expected['mean']:.4f}, got {actual_mean:.4f})"
                        )

        is_valid = len(errors) == 0

        if strict and not is_valid:
            error_msg = "Feature schema validation failed:\n" + "\n".join(
                f"  - {err}" for err in errors
            )
            raise ValueError(error_msg)

        return is_valid, errors

    def _dtypes_compatible(self, dtype1: str, dtype2: str) -> bool:
        """Check if two dtypes are compatible (e.g., float32 vs float64)."""
        # Normalize dtype names
        d1 = dtype1.lower().replace("numpy.", "").replace("np.", "")
        d2 = dtype2.lower().replace("numpy.", "").replace("np.", "")

        # Float types are compatible
        if "float" in d1 and "float" in d2:
            return True

        # Int types are compatible
        if "int" in d1 and "int" in d2:
            return True

        # Exact match
        return d1 == d2

    def diff(self, other: FeaturesSchema) -> dict[str, Any]:
        """
        Generate detailed diff between this schema and another.

        Args:
            other: Another FeaturesSchema to compare against

        Returns:
            Dictionary with diff details
        """
        diff: dict[str, Any] = {
            "columns_added": list(set(other.columns) - set(self.columns)),
            "columns_removed": list(set(self.columns) - set(other.columns)),
            "dtype_changes": {},
            "order_changed": self.order_hash != other.order_hash,
        }

        # Find dtype changes
        common_cols = set(self.columns) & set(other.columns)
        for col in common_cols:
            if self.dtypes.get(col) != other.dtypes.get(col):
                diff["dtype_changes"][col] = {
                    "old": self.dtypes.get(col),
                    "new": other.dtypes.get(col),
                }

        return diff

    def save(self, path: Path) -> None:
        """
        Save schema to JSON file.

        Args:
            path: Output file path (e.g., model_dir/features_schema.json)
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        schema_dict = {
            "columns": self.columns,
            "dtypes": self.dtypes,
            "order_hash": self.order_hash,
            "statistics": self.statistics,
            "created_at": self.created_at,
            "version": self.version,
            "schema_hash": self.compute_hash(),
        }

        safe_json_dump(schema_dict, path, indent=2, ensure_ascii=False)

        logger.info(
            "Feature schema saved to %s (hash: %s...)", path, self.compute_hash()[:16]
        )

    @classmethod
    def load(cls, path: Path) -> FeaturesSchema:
        """
        Load schema from JSON file.

        Args:
            path: Input file path

        Returns:
            FeaturesSchema instance

        Raises:
            FileNotFoundError: If schema file does not exist
            ValueError: If schema format is invalid
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Feature schema not found: {path}")

        with open(path, "r", encoding="utf-8") as f:
            schema_dict = json.load(f)

        # Validate version
        version = schema_dict.get("version", "1.0")
        if version != "1.0":
            raise ValueError(f"Unsupported schema version: {version}")

        # Verify hash integrity
        schema = cls(
            columns=schema_dict["columns"],
            dtypes=schema_dict["dtypes"],
            order_hash=schema_dict["order_hash"],
            statistics=schema_dict.get("statistics", {}),
            created_at=schema_dict.get("created_at"),
            version=version,
        )

        saved_hash = schema_dict.get("schema_hash")
        computed_hash = schema.compute_hash()
        if saved_hash and saved_hash != computed_hash:
            raise ValueError(
                f"Schema hash mismatch: saved {saved_hash[:16]}..., "
                f"computed {computed_hash[:16]}..."
            )

        logger.info(
            "Feature schema loaded from %s (hash: %s...)", path, computed_hash[:16]
        )
        return schema

    @classmethod
    def load_and_validate(
        cls,
        path: Path,
        df: pd.DataFrame,
        feature_columns: list[str] | None = None,
        strict: bool = True,
    ) -> FeaturesSchema:
        """
        Load schema and validate DataFrame in one step.

        Args:
            path: Schema file path
            df: DataFrame to validate
            feature_columns: Expected feature columns
            strict: If True, raise on validation failure

        Returns:
            Loaded FeaturesSchema instance

        Raises:
            FileNotFoundError: If schema file missing
            ValueError: If validation fails and strict=True
        """
        schema = cls.load(path)
        is_valid, errors = schema.validate_dataframe(df, feature_columns, strict=False)

        if not is_valid:
            error_msg = "Feature schema validation failed:\n" + "\n".join(
                f"  - {err}" for err in errors
            )
            if strict:
                raise ValueError(error_msg)
            else:
                logger.warning("WARNING: %s", error_msg)

        return schema

def create_and_save_schema(
    df: pd.DataFrame,
    model_dir: Path,
    feature_columns: list[str] | None = None,
) -> FeaturesSchema:
    """
    Convenience function: Create schema from DataFrame and save to model directory.

    Args:
        df: Training DataFrame
        model_dir: Model directory path
        feature_columns: Feature column names (auto-detect if None)

    Returns:
        Created FeaturesSchema instance
    """
    schema = FeaturesSchema.from_dataframe(df, feature_columns, compute_stats=True)
    schema_path = model_dir / "features_schema.json"
    schema.save(schema_path)
    return schema

def load_and_validate_schema(
    model_dir: Path,
    df: pd.DataFrame,
    feature_columns: list[str] | None = None,
    strict: bool = True,
) -> FeaturesSchema:
    """
    Convenience function: Load schema and validate DataFrame.

    Args:
        model_dir: Model directory path
        df: DataFrame to validate
        feature_columns: Feature column names
        strict: If True, raise on validation failure

    Returns:
        Loaded FeaturesSchema instance

    Raises:
        FileNotFoundError: If schema file missing
        ValueError: If validation fails and strict=True
    """
    schema_path = model_dir / "features_schema.json"
    return FeaturesSchema.load_and_validate(schema_path, df, feature_columns, strict)
