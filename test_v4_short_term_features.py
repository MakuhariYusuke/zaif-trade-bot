#!/usr/bin/env python3
"""
Test script for V4FeatureExtractor short-term enhanced features

V4FeatureExtractorの短期間拡張特徴量テストスクリプト
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta

# Add project root to path
import sys
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ztb.features.unified_feature import V4FeatureExtractor


def create_sample_market_data(n_periods: int = 100) -> pd.DataFrame:
    """Create sample market data for testing"""
    np.random.seed(42)

    # Generate timestamps
    start_time = datetime(2024, 1, 1, 9, 0, 0)
    timestamps = [start_time + timedelta(minutes=i) for i in range(n_periods)]

    # Generate OHLCV data with some trends and volatility
    base_price = 50000.0
    prices = [base_price]

    for i in range(1, n_periods):
        # Add some trend and random walk
        trend = 0.0001 * np.sin(i / 10)  # Slight trend
        noise = np.random.normal(0, 0.005)  # Random noise
        new_price = prices[-1] * (1 + trend + noise)
        prices.append(max(new_price, 1000))  # Floor price

    # Create OHLCV from close prices
    df_data = []
    for i, close in enumerate(prices):
        # Generate OHLC around close price
        volatility = abs(np.random.normal(0, 0.002))
        high = close * (1 + volatility)
        low = close * (1 - volatility)
        open_price = prices[i-1] if i > 0 else close * (1 + np.random.normal(0, 0.001))
        volume = np.random.randint(100, 1000)

        df_data.append({
            'timestamp': timestamps[i],
            'open': open_price,
            'high': high,
            'low': low,
            'close': close,
            'volume': volume
        })

    return pd.DataFrame(df_data)


def test_v4_feature_extractor():
    """Test V4FeatureExtractor with short-term enhanced features"""
    print("🧪 Testing V4FeatureExtractor with short-term enhanced features")
    print("=" * 60)

    # Create sample data
    print("📊 Creating sample market data...")
    df = create_sample_market_data(200)
    print(f"Created {len(df)} periods of market data")
    print(f"Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")
    print(f"Price range: ${df['close'].min():.2f} - ${df['close'].max():.2f}")
    # Initialize V4FeatureExtractor
    print("\n🔧 Initializing V4FeatureExtractor...")
    extractor = V4FeatureExtractor()

    # Test feature extraction
    print("⚙️  Extracting features...")
    try:
        features_df = extractor.extract_features(df)
        print("✅ Feature extraction successful!")
        print(f"Original columns: {len(df.columns)}")
        print(f"Features added: {len(features_df.columns) - len(df.columns)}")
        print(f"Total columns: {len(features_df.columns)}")

        # Check for new short-term features
        short_term_features = [
            'realized_volatility',
            'tick_volume_ratio',
            'order_flow_imbalance'
        ]

        print("\n📈 Short-term features check:")
        for feature in short_term_features:
            if feature in features_df.columns:
                values = features_df[feature].dropna()
                if len(values) > 0:
                    print(f"✅ {feature}: {len(values)} values, "
                          f"range: [{values.min():.6f}, {values.max():.6f}]")
                else:
                    print(f"⚠️  {feature}: No valid values")
            else:
                print(f"❌ {feature}: Not found")

        # Show sample of features
        print("\n🔍 Sample features (last 5 rows):")
        display_cols = ['close', 'volume', 'realized_volatility', 'tick_volume_ratio', 'order_flow_imbalance']
        available_cols = [col for col in display_cols if col in features_df.columns]
        if available_cols:
            print(features_df[available_cols].tail())

        return True

    except Exception as e:
        print(f"❌ Feature extraction failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_news_sentiment_integration():
    """Test news sentiment integration"""
    print("\n📰 Testing news sentiment integration")
    print("=" * 40)

    # Create sample data
    df = create_sample_market_data(50)

    # Create sample news data
    sample_news = [
        "Bitcoin surges to new all-time high as institutional adoption increases",
        "Market volatility rises amid economic uncertainty",
        "Cryptocurrency regulations may impact trading volumes",
        "Positive earnings reports boost market confidence"
    ]

    print(f"📊 Sample news items: {len(sample_news)}")

    # Initialize extractor
    extractor = V4FeatureExtractor()

    try:
        # Test with news data
        features_df = extractor.extract_features(df, news_data=sample_news)

        news_features = ['news_sentiment_score', 'news_sentiment_intensity']
        print("📰 News sentiment features check:")
        for feature in news_features:
            if feature in features_df.columns:
                values = features_df[feature].dropna()
                if len(values) > 0:
                    print(f"✅ {feature}: {len(values)} values, "
                          f"range: [{float(values.min()):.6f}, {float(values.max()):.6f}]")
                else:
                    print(f"⚠️  {feature}: No valid values")
            else:
                print(f"❌ {feature}: Not found")

        return True

    except Exception as e:
        print(f"❌ News sentiment integration failed: {e}")
        return False


if __name__ == "__main__":
    print("🚀 V4FeatureExtractor Short-term Features Test")
    print("=" * 50)

    # Test basic functionality
    success1 = test_v4_feature_extractor()

    # Test news sentiment integration
    success2 = test_news_sentiment_integration()

    # Summary
    print("\n" + "=" * 50)
    if success1 and success2:
        print("🎉 All tests passed! Short-term features are working correctly.")
    else:
        print("⚠️  Some tests failed. Please check the implementation.")

    print("=" * 50)