"""
Test script for SAC v437 Feature Engineering

Tests quality filtering, bull/bear balance, and feature generation.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import sys

# Add the project root to Python path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from ztb.features.v437.engine.sac_v437_feature_engineering import SACv437FeatureEngineer

def create_test_data():
    """Create synthetic test data."""
    np.random.seed(42)
    n_samples = 100

    # Generate synthetic OHLCV data
    dates = pd.date_range('2023-01-01', periods=n_samples, freq='D')

    # Base price series with trend
    base_price = 100 + np.cumsum(np.random.randn(n_samples) * 0.5)
    base_price = np.maximum(base_price, 50)  # Prevent negative prices

    # Generate OHLCV
    high = base_price * (1 + np.abs(np.random.randn(n_samples)) * 0.02)
    low = base_price * (1 - np.abs(np.random.randn(n_samples)) * 0.02)
    open_price = base_price + np.random.randn(n_samples) * 0.5
    close = base_price + np.random.randn(n_samples) * 0.5
    volume = np.random.randint(1000, 10000, n_samples)

    # Create DataFrame
    df = pd.DataFrame({
        'timestamp': dates,
        'open': open_price,
        'high': high,
        'low': low,
        'close': close,
        'volume': volume
    })

    df.set_index('timestamp', inplace=True)
    return df

def test_quality_filtering():
    """Test quality filtering functionality."""
    print("Testing quality filtering...")

    # Create test data with some problematic features
    df = create_test_data()

    # Add some problematic features
    df['constant_feature'] = 1.0  # Zero variance
    df['mostly_zero'] = np.where(np.random.rand(len(df)) > 0.9, 1.0, 0.0)  # 90% zeros
    df['high_nan'] = df['close'].copy()
    df.iloc[:10, df.columns.get_loc('high_nan')] = np.nan  # 10% NaN

    # Initialize engineer
    engineer = SACv437FeatureEngineer()

    # Generate features
    features = engineer.generate_v437_features(df, feature_set='minimal')

    # Check that problematic features were removed
    removed_features = ['constant_feature', 'mostly_zero', 'high_nan']
    for feature in removed_features:
        if feature in features.columns:
            print(f"WARNING: Problematic feature {feature} was not removed!")
        else:
            print(f"✓ Problematic feature {feature} was correctly removed")

    print(f"Generated {len(features.columns)} features after quality filtering")
    return features

def test_bull_bear_balance():
    """Test bull/bear market feature balance."""
    print("\nTesting bull/bear market balance...")

    df = create_test_data()
    engineer = SACv437FeatureEngineer()

    features = engineer.generate_v437_features(df, feature_set='high_quality')

    # Count bull and bear features
    bull_features = [col for col in features.columns if col.startswith('bull_')]
    bear_features = [col for col in features.columns if col.startswith('bear_')]

    print(f"Bull market features: {len(bull_features)}")
    print(f"Bear market features: {len(bear_features)}")

    # Check balance
    balance_ratio = len(bull_features) / max(len(bear_features), 1)
    print(".2f")

    if 0.8 <= balance_ratio <= 1.2:
        print("✓ Bull/bear features are well balanced")
    else:
        print("⚠ Bull/bear features may be imbalanced")

    return features

def test_feature_sets():
    """Test different feature set configurations."""
    print("\nTesting feature set configurations...")

    df = create_test_data()
    engineer = SACv437FeatureEngineer()

    sets = ['minimal', 'balanced', 'high_quality']
    results = {}

    for feature_set in sets:
        features = engineer.generate_v437_features(df, feature_set=feature_set)
        results[feature_set] = len(features.columns)
        print(f"{feature_set}: {len(features.columns)} features")

    # Check that sets are properly ordered
    if results['minimal'] <= results['balanced'] <= results['high_quality']:
        print("✓ Feature sets are properly ordered by size")
    else:
        print("⚠ Feature sets may not be properly ordered")

    return results

def test_correlation_control():
    """Test correlation control functionality."""
    print("\nTesting correlation control...")

    df = create_test_data()
    engineer = SACv437FeatureEngineer()

    # Add highly correlated features
    df['close_dup'] = df['close'] + np.random.randn(len(df)) * 0.01  # 99% correlated
    df['open_dup'] = df['open'] + np.random.randn(len(df)) * 0.01   # 99% correlated

    features = engineer.generate_v437_features(df, feature_set='balanced')

    # Check correlation matrix
    if len(features.columns) > 1:
        corr_matrix = features.corr().abs()
        max_corr = corr_matrix.where(np.triu(np.ones_like(corr_matrix), k=1).astype(bool)).max().max()

        print(".3f")
        if max_corr < 0.95:
            print("✓ Correlation control is working")
        else:
            print("⚠ High correlation still present")

    return features

def main():
    """Run all tests."""
    print("SAC v437 Feature Engineering Test Suite")
    print("=" * 50)

    try:
        # Run tests
        test_quality_filtering()
        test_bull_bear_balance()
        test_feature_sets()
        test_correlation_control()

        print("\n" + "=" * 50)
        print("✓ All tests completed successfully!")

    except Exception as e:
        print(f"\n✗ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return 1

    return 0

if __name__ == "__main__":
    exit(main())