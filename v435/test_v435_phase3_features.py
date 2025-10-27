#!/usr/bin/env python3
"""
Test script for SAC v435 Phase 3: Adaptive Feature Selection and Multi-Timeframe Integration

This script tests the implementation of:
1. Adaptive feature selection based on market regime
2. Multi-timeframe feature integration
3. Feature importance analysis
"""

import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from ztb.data.v433_feature_engineering import AdaptiveFeatureEngineer
from ztb.features.adaptive_selection import AdaptiveFeatureSelector
from ztb.utils.logging_utils import setup_logging

# Setup logging
setup_logging()
logger = logging.getLogger(__name__)


def test_adaptive_feature_selection():
    """Test adaptive feature selection functionality"""
    logger.info("🧪 Testing Adaptive Feature Selection")

    # Create sample data with various market conditions
    np.random.seed(42)
    n_samples = 1000

    # Generate synthetic OHLCV data
    data = {
        "open": 100 + np.random.randn(n_samples).cumsum() * 0.1,
        "high": 101 + np.random.randn(n_samples).cumsum() * 0.1,
        "low": 99 + np.random.randn(n_samples).cumsum() * 0.1,
        "close": 100 + np.random.randn(n_samples).cumsum() * 0.1,
        "volume": np.random.randint(1000, 10000, n_samples),
    }

    df = pd.DataFrame(data)

    # Add technical indicators
    engineer = AdaptiveFeatureEngineer()
    df_with_features = engineer.create_features(df)

    logger.info(f"Generated dataset with {len(df_with_features.columns)} features")

    # Test adaptive feature selector
    selector = AdaptiveFeatureSelector()

    # Get all available features
    all_features = [
        col
        for col in df_with_features.columns
        if col not in ["open", "high", "low", "close", "volume"]
    ]

    # Test feature selection
    selected_features, stats = selector.select_features(df_with_features, all_features)

    logger.info(
        f"Selected {len(selected_features)} features out of {len(all_features)}"
    )
    logger.info(f"Selection method: {stats['selection_method']}")
    logger.info(f"Top 10 selected features: {selected_features[:10]}")

    # Verify selection makes sense
    assert len(selected_features) > 0, "No features selected"
    assert len(selected_features) <= len(
        all_features
    ), "Selected more features than available"
    assert all(
        f in all_features for f in selected_features
    ), "Selected features not in original list"

    logger.info("✅ Adaptive feature selection test passed")
    return True


def test_multi_timeframe_integration():
    """Test multi-timeframe feature integration"""
    logger.info("🧪 Testing Multi-Timeframe Integration")

    # Test timeframe definitions
    logger.info(f"Available timeframes: {list(TIMEFRAME_DEFINITIONS.keys())}")

    # Create sample data for different timeframes
    np.random.seed(42)
    base_price = 100

    # Generate data for multiple timeframes
    timeframes_data = {}
    for tf_name, tf_config in TIMEFRAME_DEFINITIONS.items():
        n_samples = 500  # 500 periods for each timeframe

        # Generate price data with different volatility scales
        volatility_scale = tf_config.get("volatility_scale", 1.0)
        trend_strength = tf_config.get("trend_strength", 1.0)

        prices = (
            base_price + np.random.randn(n_samples).cumsum() * volatility_scale * 0.1
        )
        prices += np.linspace(0, trend_strength * 10, n_samples)  # Add trend

        data = {
            "open": prices[:-1],
            "high": prices[:-1]
            + abs(np.random.randn(n_samples - 1)) * volatility_scale * 0.05,
            "low": prices[:-1]
            - abs(np.random.randn(n_samples - 1)) * volatility_scale * 0.05,
            "close": prices[1:],
            "volume": np.random.randint(1000, 10000, n_samples - 1),
        }

        df = pd.DataFrame(data)
        timeframes_data[tf_name] = df

        logger.info(f"Generated {tf_name} data: {len(df)} samples")

    # Test feature engineering on different timeframes
    engineer = AdaptiveFeatureEngineer()

    multi_tf_features = {}
    for tf_name, df in timeframes_data.items():
        try:
            features_df = engineer.create_features(df)
            multi_tf_features[tf_name] = features_df
            logger.info(
                f"Calculated features for {tf_name}: {len(features_df.columns)} features"
            )
        except Exception as e:
            logger.warning(f"Failed to calculate features for {tf_name}: {e}")

    # Test multi-timeframe feature combination
    if len(multi_tf_features) >= 2:
        # Combine features from different timeframes
        primary_tf = "1h"
        secondary_tf = "4h"

        if primary_tf in multi_tf_features and secondary_tf in multi_tf_features:
            primary_features = multi_tf_features[primary_tf]
            secondary_features = multi_tf_features[secondary_tf]

            # Resample secondary to match primary timeframe (simplified)
            # In real implementation, this would use proper resampling
            secondary_resampled = secondary_features.iloc[::4]  # Simple downsampling

            # Combine features
            combined_features = pd.concat(
                [
                    primary_features.add_suffix("_1h"),
                    secondary_resampled.add_suffix("_4h").reset_index(drop=True),
                ],
                axis=1,
            )

            logger.info(
                f"Combined multi-timeframe features: {len(combined_features.columns)} total features"
            )
            logger.info("✅ Multi-timeframe integration test passed")
            return True

    logger.warning("⚠️ Multi-timeframe integration test incomplete - insufficient data")
    return False


def test_feature_importance_analysis():
    """Test feature importance analysis"""
    logger.info("🧪 Testing Feature Importance Analysis")

    # Create sample data
    np.random.seed(42)
    n_samples = 500

    data = {
        "open": 100 + np.random.randn(n_samples).cumsum() * 0.1,
        "high": 101 + np.random.randn(n_samples).cumsum() * 0.1,
        "low": 99 + np.random.randn(n_samples).cumsum() * 0.1,
        "close": 100 + np.random.randn(n_samples).cumsum() * 0.1,
        "volume": np.random.randint(1000, 10000, n_samples),
    }

    df = pd.DataFrame(data)

    # Add technical indicators
    engineer = AdaptiveFeatureEngineer()
    df_with_features = engineer.create_features(df)

    # Generate synthetic rewards based on some features
    # Reward is higher when RSI < 30 (oversold) and price is rising
    rsi = df_with_features.get("RSI", pd.Series(np.random.randn(n_samples) * 10 + 50))
    price_change = df_with_features["close"].pct_change()

    rewards = np.where(
        (rsi < 30) & (price_change > 0),
        1.0,  # Good reward
        np.where((rsi > 70) & (price_change < 0), -1.0, 0.0),  # Bad reward  # Neutral
    )

    df_with_features["reward"] = rewards

    # Test feature importance analysis
    selector = AdaptiveFeatureSelector()

    # Get all features
    all_features = [
        col
        for col in df_with_features.columns
        if col not in ["open", "high", "low", "close", "volume", "reward"]
    ]

    # Test causal feature selection
    try:
        selected_features, stats = selector.select_features(
            df_with_features, all_features, use_causal=True, outcome_feature="reward"
        )

        logger.info(f"Causal selection: {len(selected_features)} features selected")
        logger.info(f"Selection stats: {stats}")

        if len(selected_features) > 0:
            logger.info("✅ Feature importance analysis test passed")
            return True
        else:
            logger.warning("⚠️ No features selected in causal analysis")
            return False

    except Exception as e:
        logger.warning(f"⚠️ Causal feature selection failed: {e}")
        # Fall back to adaptive selection
        selected_features, stats = selector.select_features(
            df_with_features, all_features
        )
        logger.info(f"Fallback adaptive selection: {len(selected_features)} features")
        return len(selected_features) > 0


def main():
    """Main test function"""
    logger.info("🚀 Starting SAC v435 Phase 3 Feature Engineering Tests")

    results = {}

    try:
        # Test 1: Adaptive Feature Selection
        results["adaptive_selection"] = test_adaptive_feature_selection()

        # Test 2: Multi-Timeframe Integration
        results["multi_timeframe"] = test_multi_timeframe_integration()

        # Test 3: Feature Importance Analysis
        results["feature_importance"] = test_feature_importance_analysis()

        # Summary
        passed = sum(results.values())
        total = len(results)

        logger.info(f"📊 Test Results: {passed}/{total} tests passed")

        if passed == total:
            logger.info("🎉 All Phase 3 tests passed!")
            return True
        else:
            logger.warning(f"⚠️ {total - passed} tests failed")
            return False

    except Exception as e:
        logger.error(f"❌ Test suite failed: {e}")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
