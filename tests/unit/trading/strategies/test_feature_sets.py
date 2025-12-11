#!/usr/bin/env python3
"""
Test script for configurable feature sets.

Tests the new feature set configuration system.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from ztb.features.models.sac.sac_v427_feature_engineering import SACv427FeatureEngineer


def create_test_data(n_points=100):
    """Create test OHLCV data."""
    np.random.seed(42)
    dates = pd.date_range("2023-01-01", periods=n_points, freq="D")

    # Generate realistic OHLCV data
    close = 100 + np.cumsum(np.random.randn(n_points) * 2)
    high = close + np.abs(np.random.randn(n_points))
    low = close - np.abs(np.random.randn(n_points))
    open_price = close + np.random.randn(n_points) * 0.5
    volume = np.random.randint(1000, 10000, n_points)

    df = pd.DataFrame(
        {
            "open": open_price,
            "high": high,
            "low": low,
            "close": close,
            "volume": volume,
            "dividends": 0.0,  # Always zero
            "stock splits": 0.0,  # Always zero
        },
        index=dates,
    )

    return df


def test_feature_sets():
    """Test different feature set configurations."""
    print("=== Testing Configurable Feature Sets ===\n")

    # Create test data
    test_df = create_test_data(50)
    print(f"Created test dataset with {len(test_df)} points")
    print(f"Original features: {list(test_df.columns)}\n")

    # Test different feature sets
    feature_sets = [
        "minimal",
        "no_harmful",
        "full",
    ]  # Skip high_quality for now due to column exclusion

    for set_name in feature_sets:
        print(f"--- Testing feature set: {set_name} ---")

        try:
            # Create feature engineer with specific set
            feature_engineer = SACv427FeatureEngineer(market_system=None)

            # Generate features with specific set
            features_df = feature_engineer.generate_v427_features(
                test_df.copy(), feature_set=set_name
            )

            print(f"Total features generated: {len(features_df.columns)}")
            print(
                f"Excluded features: {feature_engineer.feature_config.get_excluded_features()}"
            )

            # Check if harmful features are excluded
            harmful_features = ["dividends", "stock splits"]
            excluded_count = sum(
                1 for col in harmful_features if col not in features_df.columns
            )
            print(
                f"Harmful features excluded: {excluded_count}/{len(harmful_features)}"
            )

            print()

        except Exception as e:
            print(f"Error testing {set_name}: {e}\n")


def test_custom_config():
    """Test custom configuration."""
    print("=== Testing Custom Configuration ===\n")

    # Create feature engineer
    feature_engineer = SACv427FeatureEngineer(market_system=None)

    # Get config and modify
    config = feature_engineer.feature_config
    print(f"Current excluded features: {config.get_excluded_features()}")

    # Add custom exclusion
    config.add_excluded_feature("volume")
    print(f"After adding 'volume': {config.get_excluded_features()}")

    # Test generation
    test_df = create_test_data(30)
    features_df = feature_engineer.generate_v427_features(test_df.copy())

    print(f"Features after custom config: {len(features_df.columns)}")
    print(f"'volume' excluded: {'volume' not in features_df.columns}")


if __name__ == "__main__":
    test_feature_sets()
    test_custom_config()
    print("=== Test completed ===")
