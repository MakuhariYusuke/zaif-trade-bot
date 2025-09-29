#!/usr/bin/env python3
"""
CI script to test only verified features from coverage.json
Comprehensive testing including basic execution, parameter sensitivity, edge cases, and performance regression.
"""

import json
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import yaml

# Add ztb to path
sys.path.insert(0, str(Path(__file__).parent))

from typing import TYPE_CHECKING

from ztb.features import FeatureRegistry
from ztb.features.base import CommonPreprocessor

if TYPE_CHECKING:
    from typing import Type

    FeatureRegistryType = Type[FeatureRegistry]
else:
    FeatureRegistryType = FeatureRegistry


def get_feature_category(feature_name: str) -> str:
    """Determine feature category from feature name"""
    name_lower = feature_name.lower()

    # Trend indicators
    if any(
        keyword in name_lower
        for keyword in [
            "ema",
            "sma",
            "wma",
            "kama",
            "tema",
            "dema",
            "ichimoku",
            "trend",
        ]
    ):
        return "trend"

    # Oscillators
    if any(
        keyword in name_lower
        for keyword in ["rsi", "stoch", "macd", "cci", "williams", "oscillator"]
    ):
        return "oscillator"

    # Volume indicators
    if any(keyword in name_lower for keyword in ["volume", "obv", "vwap", "vpt"]):
        return "volume"

    # Channel indicators
    if any(
        keyword in name_lower
        for keyword in ["bollinger", "donchian", "channel", "envelope"]
    ):
        return "channel"

    return "other"


def check_category_requirements(
    verified_features: List[str], config_path: str = "config/feature_params.yaml"
) -> bool:
    """Check if category requirements are met"""
    print("Checking category requirements...")

    # Load config
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    category_reqs = config.get("ci", {}).get("category_requirements", {})
    if not category_reqs:
        print("No category requirements defined")
        return True

    # Count verified features by category
    category_counts = {}
    for feature in verified_features:
        category = get_feature_category(feature)
        category_counts[category] = category_counts.get(category, 0) + 1

    print(f"Verified features by category: {category_counts}")

    # Check requirements
    all_met = True
    for category, required_count in category_reqs.items():
        actual_count = category_counts.get(category, 0)
        if actual_count < required_count:
            print(
                f"❌ Category '{category}': {actual_count}/{required_count} (insufficient)"
            )
            all_met = False
        else:
            print(f"✓ Category '{category}': {actual_count}/{required_count}")

    return all_met


def load_performance_history(history_path: str) -> Dict[str, float]:
    """Load previous performance benchmarks"""
    if not Path(history_path).exists():
        return {}

    try:
        with open(history_path, "r") as f:
            return json.load(f)
    except Exception as e:
        print(f"Warning: Could not load performance history: {e}")
        return {}


def save_performance_history(history_path: str, benchmarks: Dict[str, float]):
    """Save current performance benchmarks"""
    Path(history_path).parent.mkdir(parents=True, exist_ok=True)

    try:
        with open(history_path, "w") as f:
            json.dump(benchmarks, f, indent=2)
    except Exception as e:
        print(f"Warning: Could not save performance history: {e}")


def check_performance_regression(
    verified_features: List[str],
    current_benchmarks: Dict[str, float],
    config_path: str = "config/feature_params.yaml",
) -> List[str]:
    """Check for performance regression against historical benchmarks"""
    print("Checking performance regression...")

    # Load config
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    tolerance_ratio = config.get("ci", {}).get("perf_regression_tolerance_ratio", 0.5)
    history_path = config.get("ci", {}).get(
        "bench_history_path", "reports/performance/feature_bench_history.json"
    )

    # Load historical benchmarks
    historical_benchmarks = load_performance_history(history_path)

    regressed_features = []

    for feature_name in verified_features:
        if feature_name not in current_benchmarks:
            continue

        current_time = current_benchmarks[feature_name]
        historical_time = historical_benchmarks.get(feature_name)

        if historical_time is None:
            print(f"✓ {feature_name}: {current_time:.3f}s (new benchmark)")
            continue

        # Calculate regression ratio
        ratio = current_time / historical_time

        # Check if within tolerance (allowing both faster and slower within tolerance)
        if abs(ratio - 1.0) > tolerance_ratio:
            direction = "slower" if ratio > 1.0 else "faster"
            change_pct = abs(ratio - 1.0) * 100
            print(
                f"❌ {feature_name}: {change_pct:.1f}% {direction} ({current_time:.3f}s vs {historical_time:.3f}s)"
            )
            regressed_features.append(feature_name)
        else:
            change_pct = (ratio - 1.0) * 100
            print(f"✓ {feature_name}: {change_pct:+.1f}% ({current_time:.3f}s)")

    return regressed_features


def test_basic_execution(
    verified_features: List[str],
    sample_data: pd.DataFrame,
    registry: FeatureRegistryType,
) -> List[str]:
    """Test that all verified features can compute successfully"""
    print(f"Testing {len(verified_features)} verified features (basic execution)...")

    failed_features = []

    for feature_name in verified_features:
        try:
            if feature_name not in registry.list():
                print(f"Feature {feature_name} not found in registry")
                failed_features.append(feature_name)
                continue

            feature = registry.get(feature_name)
            result = feature(sample_data)

            if not isinstance(result, pd.DataFrame):
                print(f"Feature {feature_name} did not return DataFrame")
                failed_features.append(feature_name)
                continue

            if result.empty:
                print(f"Feature {feature_name} returned empty DataFrame")
                failed_features.append(feature_name)
                continue

            print(f"✓ {feature_name}")

        except Exception as e:
            print(f"✗ {feature_name}: {e}")
            failed_features.append(feature_name)

    return failed_features


def test_parameter_sensitivity(
    verified_features: List[str],
    sample_data: pd.DataFrame,
    registry: FeatureRegistryType,
) -> List[str]:
    """Test parameter sensitivity - features should behave consistently with different parameters"""
    print("Testing parameter sensitivity...")

    failed_features = []

    # Test with different data scales
    scales = [0.1, 1.0, 10.0]

    for scale in scales:
        scaled_data = sample_data.copy()
        scaled_data[["open", "high", "low", "close"]] *= scale

        for feature_name in verified_features:
            try:
                if feature_name not in registry.list():
                    continue

                feature = registry.get(feature_name)
                result1 = feature(sample_data)
                result2 = feature(scaled_data)

                # Check if results have same structure
                if not result1.columns.equals(result2.columns):
                    print(
                        f"✗ {feature_name}: Inconsistent column structure across scales"
                    )
                    failed_features.append(feature_name)
                    continue

                # For price-based features, results should scale appropriately
                if any(
                    col in str(feature_name).lower()
                    for col in ["price", "close", "high", "low"]
                ):
                    # Price-based features should scale
                    if not np.allclose(
                        result1.values / scale,
                        result2.values,
                        rtol=1e-10,
                        atol=1e-10,
                        equal_nan=True,
                    ):
                        print(
                            f"✗ {feature_name}: Does not scale properly with price changes"
                        )
                        failed_features.append(feature_name)

            except Exception as e:
                print(f"✗ {feature_name} (parameter sensitivity): {e}")
                failed_features.append(feature_name)

    return list(set(failed_features))  # Remove duplicates


def test_edge_cases(
    verified_features: List[str], registry: FeatureRegistryType
) -> List[str]:
    """Test edge cases: NaN data, extreme values, short periods"""
    print("Testing edge cases...")

    failed_features = []

    # Test case 1: Data with NaN values
    dates = pd.date_range("2020-01-01", periods=50, freq="D")
    nan_data = pd.DataFrame(
        {
            "open": [
                np.nan if i % 5 == 0 else 100 + np.random.randn() for i in range(50)
            ],
            "high": [
                np.nan if i % 5 == 0 else 105 + np.random.randn() for i in range(50)
            ],
            "low": [
                np.nan if i % 5 == 0 else 95 + np.random.randn() for i in range(50)
            ],
            "close": [
                np.nan if i % 5 == 0 else 100 + np.random.randn() for i in range(50)
            ],
            "volume": [
                np.nan if i % 5 == 0 else np.random.randint(1000, 10000)
                for i in range(50)
            ],
        },
        index=dates,
    )
    nan_data = CommonPreprocessor.preprocess(nan_data)

    # Test case 2: Extreme values
    extreme_data = pd.DataFrame(
        {
            "open": [1e10 if i % 10 == 0 else 100 for i in range(50)],
            "high": [1e10 if i % 10 == 0 else 105 for i in range(50)],
            "low": [1e-10 if i % 10 == 0 else 95 for i in range(50)],
            "close": [1e10 if i % 10 == 0 else 100 for i in range(50)],
            "volume": [1e15 if i % 10 == 0 else 5000 for i in range(50)],
        },
        index=dates,
    )
    extreme_data = CommonPreprocessor.preprocess(extreme_data)

    # Test case 3: Very short data
    short_data = pd.DataFrame(
        {
            "open": [100, 101, 102],
            "high": [105, 106, 107],
            "low": [95, 96, 97],
            "close": [100, 101, 102],
            "volume": [1000, 2000, 3000],
        },
        index=pd.date_range("2020-01-01", periods=3, freq="D"),
    )
    short_data = CommonPreprocessor.preprocess(short_data)

    test_cases = [
        ("NaN data", nan_data),
        ("Extreme values", extreme_data),
        ("Short data", short_data),
    ]

    for case_name, test_data in test_cases:
        print(f"  Testing {case_name}...")

        for feature_name in verified_features:
            try:
                if feature_name not in registry.list():
                    continue

                feature = registry.get(feature_name)
                result = feature(test_data)

                # Should not crash and should return DataFrame
                if not isinstance(result, pd.DataFrame):
                    print(f"✗ {feature_name} ({case_name}): Did not return DataFrame")
                    failed_features.append(feature_name)
                    continue

                # Should handle NaN gracefully (not all NaN)
                if result.isna().all().all():
                    print(f"✗ {feature_name} ({case_name}): All results are NaN")
                    failed_features.append(feature_name)

            except Exception as e:
                print(f"✗ {feature_name} ({case_name}): {e}")
                failed_features.append(feature_name)

    return list(set(failed_features))


def test_performance_regression(
    verified_features: List[str],
    sample_data: pd.DataFrame,
    registry: FeatureRegistryType,
    config_path: str = "config/feature_params.yaml",
) -> Tuple[List[str], Dict[str, float]]:
    """Test performance regression - ensure features don't run extremely slow"""
    print("Testing performance regression...")

    # Load config
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    max_time_per_feature = 5.0  # 5 seconds max per feature

    failed_features = []
    benchmarks = {}

    for feature_name in verified_features:
        try:
            if feature_name not in registry.list():
                continue

            feature = registry.get(feature_name)

            start_time = time.time()
            result = feature(sample_data)
            end_time = time.time()

            execution_time = end_time - start_time
            benchmarks[feature_name] = execution_time

            if execution_time > max_time_per_feature:
                print(
                    f"✗ {feature_name}: Too slow ({execution_time:.2f}s > {max_time_per_feature}s)"
                )
                failed_features.append(feature_name)
            else:
                print(f"✓ {feature_name}: {execution_time:.3f}s")

        except Exception as e:
            print(f"✗ {feature_name} (performance): {e}")
            failed_features.append(feature_name)

    return failed_features, benchmarks


def test_verified_features():
    """Comprehensive test suite for verified features"""

    # Load coverage.json
    coverage_path = Path("coverage.json")
    if not coverage_path.exists():
        print("coverage.json not found")
        return False

    with open(coverage_path, "r") as f:
        coverage = json.load(f)

    verified_features = coverage.get("verified", [])
    if not verified_features:
        print("No verified features to test")
        return True

    # Create sample OHLC data
    dates = pd.date_range("2020-01-01", periods=100, freq="D")
    np.random.seed(42)
    sample_data = pd.DataFrame(
        {
            "open": 100 + np.random.randn(100).cumsum(),
            "high": 105 + np.random.randn(100).cumsum(),
            "low": 95 + np.random.randn(100).cumsum(),
            "close": 100 + np.random.randn(100).cumsum(),
            "volume": np.random.randint(1000, 10000, 100),
        },
        index=dates,
    )

    # Preprocess data
    sample_data = CommonPreprocessor.preprocess(sample_data)

    # Get feature registry (features auto-registered via imports)

    # Run all test suites
    all_failed_features = []

    # 1. Basic execution test
    failed_basic = test_basic_execution(verified_features, sample_data, FeatureRegistry)
    all_failed_features.extend(failed_basic)

    # 2. Parameter sensitivity test
    failed_sensitivity = test_parameter_sensitivity(
        verified_features, sample_data, FeatureRegistry
    )
    all_failed_features.extend(failed_sensitivity)

    # 3. Edge cases test
    failed_edge = test_edge_cases(verified_features, FeatureRegistry)
    all_failed_features.extend(failed_edge)

    # 4. Performance regression test
    failed_performance, benchmarks = test_performance_regression(
        verified_features, sample_data, FeatureRegistry
    )
    all_failed_features.extend(failed_performance)

    # 5. Category requirements check
    category_req_met = check_category_requirements(verified_features)
    if not category_req_met:
        print("❌ Category requirements not met")
        return False

    # 6. Performance regression check
    regressed_features = check_performance_regression(verified_features, benchmarks)
    if regressed_features:
        print(f"❌ Performance regression detected in: {regressed_features}")
        all_failed_features.extend(regressed_features)

    # Save benchmarks for future comparison
    with open("config/feature_params.yaml", "r") as f:
        config = yaml.safe_load(f)
    history_path = config.get("ci", {}).get(
        "bench_history_path", "reports/performance/feature_bench_history.json"
    )
    save_performance_history(history_path, benchmarks)

    # Remove duplicates
    all_failed_features = list(set(all_failed_features))

    if all_failed_features:
        print(f"\n❌ Failed features: {all_failed_features}")
        print(f"Total failures: {len(all_failed_features)}")
        return False

    print("\n✅ All verified features passed comprehensive testing!")
    return True


if __name__ == "__main__":
    success = test_verified_features()
    sys.exit(0 if success else 1)
