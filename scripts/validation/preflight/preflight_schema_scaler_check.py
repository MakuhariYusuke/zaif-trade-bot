#!/usr/bin/env python
"""
Preflight validation script for CI/CD pipelines.

Validates the integrity of training artifacts before deployment:
1. Feature schema consistency (features_schema.json)
2. Normalization statistics consistency (scaler.npz)
3. Configuration fingerprint (config_fingerprint.json)

Exit codes:
    0: All checks passed
    1: One or more checks failed
    2: Missing required files

Usage:
    python scripts/preflight_schema_scaler_check.py --model-dir models/ppo_run_20250106
    python scripts/preflight_schema_scaler_check.py --model-dir models/latest --strict
"""

import argparse
import sys
from pathlib import Path
from typing import Optional, Tuple


def check_feature_schema(model_dir: Path, strict: bool = True) -> Tuple[bool, str]:
    """
    Validate feature schema file exists and is valid.

    Args:
        model_dir: Directory containing the model artifacts
        strict: Whether to enforce strict validation

    Returns:
        (success, message): Tuple of validation result and message
    """
    schema_path = model_dir / "features_schema.json"

    if not schema_path.exists():
        return False, f"❌ Feature schema not found: {schema_path}"

    try:
        from ztb.utils.feature_schema import FeaturesSchema

        schema = FeaturesSchema.load(schema_path)  # Load from file, not directory
        schema_hash = schema.compute_hash()

        return True, f"✅ Feature schema valid (hash: {schema_hash[:16]}...)"
    except Exception as e:
        return False, f"❌ Feature schema validation failed: {e}"


def check_normalization_stats(model_dir: Path, strict: bool = True) -> Tuple[bool, str]:
    """
    Validate normalization statistics file exists and is valid.

    Args:
        model_dir: Directory containing the model artifacts
        strict: Whether to enforce strict validation

    Returns:
        (success, message): Tuple of validation result and message
    """
    scaler_path = model_dir / "scaler.npz"

    if not scaler_path.exists():
        return False, f"❌ Normalization stats not found: {scaler_path}"

    try:
        from ztb.utils.normalization import load_scaler

        stats = load_scaler(model_dir, strict=strict)
        stats_hash = stats.compute_hash()

        return (
            True,
            f"✅ Normalization stats valid "
            f"(hash: {stats_hash[:16]}..., "
            f"features: {len(stats.feature_names)}, "
            f"samples: {stats.n_samples})",
        )
    except Exception as e:
        return False, f"❌ Normalization stats validation failed: {e}"


def check_config_fingerprint(model_dir: Path, strict: bool = True) -> Tuple[bool, str]:
    """
    Validate configuration fingerprint file exists and is valid.

    Args:
        model_dir: Directory containing the model artifacts
        strict: Whether to enforce strict validation

    Returns:
        (success, message): Tuple of validation result and message
    """
    fingerprint_path = model_dir / "config_fingerprint.json"

    if not fingerprint_path.exists():
        msg = f"⚠️  Config fingerprint not found: {fingerprint_path}"
        if strict:
            return False, f"❌ {msg}"
        return True, msg

    try:
        from ztb.utils.config_fingerprint import ConfigFingerprint

        fingerprint = ConfigFingerprint.load(fingerprint_path)  # Load from file
        fp_hash = fingerprint.compute_hash()

        return (
            True,
            f"✅ Config fingerprint valid "
            f"(hash: {fp_hash[:16]}..., "
            f"feature_set: {fingerprint.feature_set}, "
            f"stage: {fingerprint.curriculum_stage})",
        )
    except Exception as e:
        return False, f"❌ Config fingerprint validation failed: {e}"


def compare_with_training(
    model_dir: Path, test_data_path: Optional[Path] = None
) -> Tuple[bool, str]:
    """
    Compare normalization stats between training and test data (if available).

    Args:
        model_dir: Directory containing the model artifacts
        test_data_path: Path to test data CSV (optional)

    Returns:
        (success, message): Tuple of validation result and message
    """
    if test_data_path is None or not test_data_path.exists():
        return True, "ℹ️  Test data not provided, skipping comparison"

    try:
        import numpy as np
        import pandas as pd

        from ztb.utils.normalization import load_scaler

        # Load training stats
        train_stats = load_scaler(model_dir, strict=True)

        # Load test data
        test_df = pd.read_csv(test_data_path)

        # Auto-detect feature columns (exclude meta columns)
        exclude_cols = {
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
            for col in test_df.columns
            if col not in exclude_cols and pd.api.types.is_numeric_dtype(test_df[col])
        ]

        # Ensure feature order matches
        if set(feature_columns) != set(train_stats.feature_names):
            missing = set(train_stats.feature_names) - set(feature_columns)
            extra = set(feature_columns) - set(train_stats.feature_names)
            return (
                False,
                f"❌ Feature mismatch: missing={missing}, extra={extra}",
            )

        # Reorder to match training
        feature_columns = train_stats.feature_names

        # Compute test stats
        feature_data = test_df[feature_columns].values
        test_mean = np.mean(feature_data, axis=0)
        test_std = np.std(feature_data, axis=0)

        # Compute differences
        mean_diff = np.max(np.abs(train_stats.mean - test_mean))
        std_diff = np.max(np.abs(train_stats.std - test_std))

        # Warn if large differences
        if mean_diff > 1.0 or std_diff > 1.0:
            return (
                True,
                f"⚠️  Large train/test difference detected: "
                f"mean Δ={mean_diff:.6f}, std Δ={std_diff:.6f}",
            )

        return (
            True,
            f"✅ Train/test stats similar: "
            f"mean Δ={mean_diff:.6f}, std Δ={std_diff:.6f}",
        )

    except Exception as e:
        return True, f"⚠️  Could not compare with test data: {e}"


def main() -> int:
    """
    Main entry point for preflight validation.

    Returns:
        Exit code (0 = success, 1 = failure, 2 = missing files)
    """
    parser = argparse.ArgumentParser(
        description="Preflight validation for model artifacts",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Validate latest model
  python scripts/preflight_schema_scaler_check.py --model-dir models/latest

  # Validate with test data comparison
  python scripts/preflight_schema_scaler_check.py \\
      --model-dir models/ppo_run_20250106 \\
      --test-data ml-dataset-enhanced.csv

  # Non-strict mode (warnings instead of errors)
  python scripts/preflight_schema_scaler_check.py \\
      --model-dir models/latest \\
      --no-strict
        """,
    )

    parser.add_argument(
        "--model-dir",
        type=Path,
        required=True,
        help="Directory containing model artifacts (e.g., models/ppo_run_20250106)",
    )

    parser.add_argument(
        "--test-data",
        type=Path,
        default=None,
        help="Optional: Test data CSV for train/test comparison",
    )

    parser.add_argument(
        "--strict",
        dest="strict",
        action="store_true",
        default=True,
        help="Strict mode: fail on any validation error (default)",
    )

    parser.add_argument(
        "--no-strict",
        dest="strict",
        action="store_false",
        help="Non-strict mode: warnings instead of errors",
    )

    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Verbose output with detailed validation info",
    )

    args = parser.parse_args()

    # Resolve model directory
    model_dir = args.model_dir.resolve()

    if not model_dir.exists():
        print(f"❌ Model directory not found: {model_dir}")
        return 2

    if not model_dir.is_dir():
        print(f"❌ Not a directory: {model_dir}")
        return 2

    print("=" * 70)
    print(f"Preflight Validation: {model_dir.name}")
    print("=" * 70)
    print()

    # Run all checks
    checks = [
        ("Feature Schema", check_feature_schema(model_dir, args.strict)),
        ("Normalization Stats", check_normalization_stats(model_dir, args.strict)),
        ("Config Fingerprint", check_config_fingerprint(model_dir, args.strict)),
    ]

    # Add optional comparison check
    if args.test_data:
        checks.append(
            ("Train/Test Comparison", compare_with_training(model_dir, args.test_data))
        )

    # Print results
    all_passed = True
    for check_name, (success, message) in checks:
        print(f"{check_name:25} {message}")
        if not success:
            all_passed = False

    print()
    print("=" * 70)

    if all_passed:
        print("✅ All preflight checks passed!")
        print("=" * 70)
        return 0
    else:
        print("❌ Preflight validation FAILED")
        print("=" * 70)
        print()
        print("Action required:")
        print("  1. Fix validation errors above")
        print("  2. Retrain model with consistent configuration")
        print("  3. Verify feature schema and normalization stats are saved")
        print()
        return 1


if __name__ == "__main__":
    sys.exit(main())
