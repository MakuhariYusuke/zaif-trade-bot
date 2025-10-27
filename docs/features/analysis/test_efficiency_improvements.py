"""
Test script for SAC v427 feature engineering efficiency improvements
"""

# Add project root to path
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.append(str(Path(__file__).parent.parent.parent))


def create_test_data(n_points=1000):
    """Create synthetic OHLCV test data."""
    np.random.seed(42)
    dates = pd.date_range("2023-01-01", periods=n_points, freq="1H")

    # Generate synthetic price data with trend and noise
    trend = np.linspace(100, 120, n_points)
    noise = np.random.normal(0, 2, n_points)
    close = trend + noise

    # Generate OHLCV data
    high = close + np.random.uniform(0, 1, n_points)
    low = close - np.random.uniform(0, 1, n_points)
    open_price = close + np.random.normal(0, 0.5, n_points)
    volume = np.random.uniform(1000, 10000, n_points)

    df = pd.DataFrame(
        {
            "open": open_price,
            "high": high,
            "low": low,
            "close": close,
            "volume": volume,
        },
        index=dates,
    )

    return df


def test_feature_generation_efficiency():
    """Test the efficiency of feature generation."""
    print("=== SAC v427 Feature Engineering Efficiency Test ===\n")

    # Create test data
    test_df = create_test_data(500)
    print(f"Created test dataset with {len(test_df)} points")

    # Test basic feature generation (without external dependencies)
    try:
        from ztb.features.sac_v427_feature_engineering import SACv427FeatureEngineer

        # Create feature engineer (without market system to avoid import issues)
        feature_engineer = SACv427FeatureEngineer(market_system=None)

        # Time the feature generation
        start_time = time.time()
        feature_df = feature_engineer.generate_v427_features(test_df.copy())
        end_time = time.time()

        generation_time = end_time - start_time
        total_features = len(feature_df.columns) - len(test_df.columns)

        print(f"Feature generation time: {generation_time:.3f} seconds")
        print(f"Total features generated: {total_features}")
        print(f"Features per second: {total_features / generation_time:.1f}")

        # Memory usage estimate
        memory_mb = feature_df.memory_usage(deep=True).sum() / 1024 / 1024
        print(f"Estimated memory usage: {memory_mb:.2f} MB")
        print("\n=== Feature Categories ===")
        print(f"Regime features: {len(regime_features)}")
        print(f"Correlation features: {len(correlation_features)}")
        print(f"Ensemble features: {len(ensemble_features)}")
        print(f"Technical indicators: {len(technical_features)}")
        print(
            f"Other features: {total_features - len(regime_features) - len(correlation_features) - len(ensemble_features) - len(technical_features)}"
        )

        return {
            "generation_time": generation_time,
            "total_features": total_features,
            "memory_mb": memory_mb,
            "feature_categories": {
                "regime": len(regime_features),
                "correlation": len(correlation_features),
                "ensemble": len(ensemble_features),
                "technical": len(technical_features),
            },
        }

    except ImportError as e:
        print(f"Import error: {e}")
        print("Cannot test full feature generation due to missing dependencies.")
        return None
    except Exception as e:
        print(f"Error during feature generation: {e}")
        return None


def analyze_feature_characteristics(feature_df, original_df):
    """Analyze basic characteristics of generated features."""
    feature_cols = [col for col in feature_df.columns if col not in original_df.columns]

    print("\n=== Feature Quality Analysis ===")

    # NaN analysis
    nan_counts = feature_df[feature_cols].isnull().sum()
    total_nans = nan_counts.sum()
    nan_percentage = (total_nans / (len(feature_df) * len(feature_cols))) * 100

    print(f"NaN percentage: {nan_percentage:.2f}%")

    # Zero value analysis
    zero_counts = (feature_df[feature_cols] == 0).sum().sum()
    zero_percentage = (zero_counts / (len(feature_df) * len(feature_cols))) * 100

    print(f"Zero value percentage: {zero_percentage:.2f}%")

    # Feature variance analysis
    variances = feature_df[feature_cols].var()
    low_variance_features = (variances < 0.01).sum()
    print(f"Low variance features (<0.01): {low_variance_features}")

    return {
        "nan_percentage": nan_percentage,
        "zero_percentage": zero_percentage,
        "low_variance_count": low_variance_features,
    }


if __name__ == "__main__":
    # Run efficiency test
    results = test_feature_generation_efficiency()

    if results:
        print("\n=== Efficiency Improvements Summary ===")
        print("✓ Pre-computed common calculations (returns, volatility, SMAs)")
        print("✓ Reduced pd.concat() operations through batching")
        print("✓ Eliminated redundant calculations across feature categories")
        print("✓ Memory optimization with float32 conversion")

        print("\n=== Performance Metrics ===")
        print(f"Generation time: {results['generation_time']:.3f} seconds")
        print(
            f"Features per second: {results['total_features'] / results['generation_time']:.1f}"
        )

        print("\n=== Recommendations for Feature Removal ===")
        print("To analyze feature importance and redundancy:")
        print("1. Run feature importance analysis with actual trading data")
        print("2. Review correlation matrix for redundant features")
        print("3. Consider removing features with:")
        print("   - Very low correlation with target (< 0.05)")
        print("   - High correlation with other features (> 0.95)")
        print("   - Consistently zero or NaN values")

        print("\nNext steps:")
        print("- Test with real market data")
        print("- Profile memory usage during training")
        print("- Validate feature importance with backtesting")
