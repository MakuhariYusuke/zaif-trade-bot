#!/usr/bin/env python3
"""
Test script for Multi-Timeframe Feature Engineering System

Tests the complete multi-timeframe feature engineering pipeline
with sample data and validation.
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

import numpy as np
import pandas as pd
from ztb.features.multi_timeframe import MultiTimeframeFeatureSystem
from ztb.features.timeframe import Timeframe
from ztb.features.feature_set_config import get_feature_config


def create_sample_data():
    """Create sample data for testing."""
    # Create sample 5-minute data
    dates = pd.date_range('2024-01-01', periods=1000, freq='5min')

    # Generate synthetic OHLCV data
    np.random.seed(42)
    base_price = 50000

    data = []
    for i, timestamp in enumerate(dates):
        # Random walk with trend
        price_change = np.random.normal(0, 100)
        base_price += price_change

        # Generate OHLC
        high = base_price + abs(np.random.normal(0, 50))
        low = base_price - abs(np.random.normal(0, 50))
        open_price = base_price + np.random.normal(0, 20)
        close = base_price + np.random.normal(0, 20)

        # Ensure OHLC relationships
        high = max(high, open_price, close)
        low = min(low, open_price, close)

        volume = np.random.uniform(0.1, 10.0)

        data.append({
            'timestamp': timestamp,
            'open': round(open_price, 2),
            'high': round(high, 2),
            'low': round(low, 2),
            'close': round(close, 2),
            'volume': round(volume, 4),
        })

    df = pd.DataFrame(data)
    return df


def test_multi_timeframe_system():
    """Test the multi-timeframe feature engineering system."""
    print("Testing Multi-Timeframe Feature Engineering System")
    print("=" * 60)

    # Test 1: Test with multi-timeframe features enabled (default)
    print("\nTest 1: Multi-timeframe features ENABLED")
    print("-" * 40)

    # Get global feature config and ensure multi-timeframe is enabled
    feature_config = get_feature_config()
    original_config = feature_config.get_feature_flags().copy()
    feature_config.current_config["include_multi_timeframe_features"] = True

    success_enabled = test_with_config(feature_config, "enabled")

    # Test 2: Test with multi-timeframe features disabled
    print("\nTest 2: Multi-timeframe features DISABLED")
    print("-" * 40)

    feature_config.current_config["include_multi_timeframe_features"] = False

    success_disabled = test_with_config(feature_config, "disabled")

    # Restore original config
    feature_config.current_config.update(original_config)

    return success_enabled and success_disabled


def test_with_config(feature_config, config_type):
    """Test system with specific configuration."""
    # Create sample data
    print("Creating sample data...")
    sample_data = create_sample_data()
    print(f"Created {len(sample_data)} rows of sample data")

    # Save sample data to temporary files
    data_dir = Path("test_data")
    data_dir.mkdir(exist_ok=True)

    sample_files = {}
    for timeframe in [Timeframe.M1, Timeframe.M5, Timeframe.M15, Timeframe.H1]:
        filename = f"btc_jpy_{timeframe.value}_sample.csv"
        filepath = data_dir / filename

        # For simplicity, use the same data for all timeframes
        # In practice, you'd have different data for each timeframe
        sample_data.to_csv(filepath, index=False)
        sample_files[timeframe] = str(filepath)

    print(f"Saved sample data files: {list(sample_files.keys())}")

    try:
        # Initialize the system
        print(f"\nInitializing MultiTimeframeFeatureSystem ({config_type})...")
        system = MultiTimeframeFeatureSystem()

        # Get system info
        info = system.get_system_info()
        print(f"System Info: {info}")

        if config_type == "disabled":
            # When disabled, system should return empty DataFrame
            features_df = system.process_multi_timeframe_data(
                data_files=sample_files,
                feature_set="minimal",
            )
            if features_df.empty:
                print("✓ Multi-timeframe features correctly disabled - returned empty DataFrame")
                return True
            else:
                print("✗ Multi-timeframe features not properly disabled")
                return False

        # Process multi-timeframe features (only for enabled case)
        print("\nProcessing multi-timeframe features...")
        features_df = system.process_multi_timeframe_data(
            data_files=sample_files,
            feature_set="minimal",  # Use minimal for faster testing
        )

        print("Feature generation completed!")
        print(f"Output shape: {features_df.shape}")
        print(f"Feature columns: {len(features_df.columns)}")

        # Show some feature names
        print("\nSample feature names:")
        for i, col in enumerate(features_df.columns[:10]):
            print(f"  {i+1}. {col}")

        # Get data quality report
        print("\nData Quality Report:")
        quality_report = system.get_data_quality_report()
        if 'timeframes' in quality_report:
            for tf, report in quality_report['timeframes'].items():
                print(f"  {tf}: {report['row_count']} rows, {report['data_quality']['valid_ohlc']:.1%} valid OHLC")
        else:
            print(f"  Status: {quality_report.get('status', 'unknown')}")

        # Get feature counts
        print("\nFeature Counts per Timeframe:")
        feature_counts = system.get_feature_counts()
        for tf, count in feature_counts.items():
            print(f"  {tf}: {count} features")

        # Validate system
        print("\nSystem Validation:")
        issues = system.validate_system()
        if issues:
            print("Issues found:")
            for issue in issues:
                print(f"  - {issue}")
        else:
            print("✓ No validation issues found")

        print(f"\n✓ Multi-timeframe feature engineering system test ({config_type}) completed successfully!")
        return True

    except Exception as e:
        print(f"\n✗ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False

    finally:
        # Clean up
        import shutil
        if data_dir.exists():
            shutil.rmtree(data_dir)
            print(f"\nCleaned up test data directory: {data_dir}")


if __name__ == "__main__":
    success = test_multi_timeframe_system()
    sys.exit(0 if success else 1)