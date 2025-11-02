#!/usr/bin/env python3
"""
Quick test script for V444 regime adaptation improvements
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np

# Add project root to path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from ztb.analysis.v444_regime_classifier import V444RegimeClassifier

def create_sample_data():
    """Create sample OHLCV data for testing"""
    np.random.seed(42)
    dates = pd.date_range('2023-01-01', periods=200, freq='1H')

    # Generate sample price data with some trends
    base_price = 5000000  # 5M JPY
    price_changes = np.random.normal(0, 0.01, 200).cumsum()
    close = pd.Series(base_price * (1 + price_changes), index=dates)

    # Generate OHLCV
    high = close * (1 + np.abs(np.random.normal(0, 0.005, 200)))
    low = close * (1 - np.abs(np.random.normal(0, 0.005, 200)))
    open_price = close.shift(1).fillna(close.iloc[0])
    volume = pd.Series(np.random.uniform(1000, 10000, 200), index=dates)

    df = pd.DataFrame({
        'open': open_price,
        'high': high,
        'low': low,
        'close': close,
        'volume': volume
    }, index=dates)

    return df

def test_regime_classification():
    """Test regime classification with dynamic thresholds"""
    print("Testing V444 Regime Classification...")

    # Create classifier
    config = {
        "dynamic_thresholds": True,
        "adaptation_window": 50
    }
    classifier = V444RegimeClassifier(config)

    # Create sample data
    data = create_sample_data()

    # Test classification at different points
    test_indices = [50, 100, 150]

    for idx in test_indices:
        result = classifier.detect_regime(data, idx)
        print(f"Index {idx}: {result.primary_regime.value} (confidence: {result.confidence:.3f})")

        # Test adaptive feature weights
        feature_names = ['rsi', 'adx', 'atr', 'macd', 'bollinger', 'momentum', 'trend', 'volume']
        weights = classifier.get_adaptive_feature_weights(result.primary_regime, feature_names)
        print(f"  Feature weights: {dict(list(weights.items())[:3])}...")

    print("Regime classification test completed.\n")

def test_dynamic_thresholds():
    """Test dynamic threshold adaptation"""
    print("Testing Dynamic Threshold Adaptation...")

    classifier = V444RegimeClassifier({"dynamic_thresholds": True, "adaptation_window": 20})

    # Create volatile data
    dates = pd.date_range('2023-01-01', periods=100, freq='1h')
    volatile_changes = np.random.normal(0, 0.03, 100).cumsum()  # High volatility
    close = pd.Series(5000000 * (1 + volatile_changes), index=dates)

    high = close * (1 + np.abs(np.random.normal(0, 0.01, 100)))
    low = close * (1 - np.abs(np.random.normal(0, 0.01, 100)))
    open_price = close.shift(1).fillna(close.iloc[0])
    volume = pd.Series(np.random.uniform(1000, 10000, 100), index=dates)

    volatile_data = pd.DataFrame({
        'open': open_price,
        'high': high,
        'low': low,
        'close': close,
        'volume': volume
    }, index=dates)

    # Test adaptation
    print("Initial thresholds:")
    print(f"  Strong trend: {classifier.thresholds['strong_trend_threshold']}")
    print(f"  High volatility: {classifier.thresholds['high_volatility_threshold']}")

    # Run classification multiple times to trigger adaptation
    for i in range(10):
        classifier.detect_regime(volatile_data, 50 + i)

    print("After adaptation:")
    print(f"  Strong trend: {classifier.thresholds['strong_trend_threshold']}")
    print(f"  High volatility: {classifier.thresholds['high_volatility_threshold']}")

    print("Dynamic threshold adaptation test completed.\n")

def test_multi_timeframe():
    """Test multi-timeframe regime detection"""
    print("Testing Multi-Timeframe Regime Detection...")

    classifier = V444RegimeClassifier({"dynamic_thresholds": True})

    # Create sample data
    data = create_sample_data()

    # Test multi-timeframe detection
    try:
        mtf_result = classifier.detect_multi_timeframe_regime(data, 100)
        print(f"Multi-timeframe integrated regime: {mtf_result.integrated_regime.value}")
        print(f"Integration confidence: {mtf_result.integration_confidence:.3f}")
        print(f"Timeframe weights: {mtf_result.timeframe_weights}")

        # Compare with single timeframe
        single_result = classifier.detect_regime(data, 100)
        print(f"Single timeframe regime: {single_result.primary_regime.value}")
        print(f"Single confidence: {single_result.confidence:.3f}")

    except Exception as e:
        print(f"Multi-timeframe test failed: {e}")
        import traceback
        traceback.print_exc()

    print("Multi-timeframe test completed.\n")

if __name__ == "__main__":
    try:
        test_regime_classification()
        test_dynamic_thresholds()
        test_multi_timeframe()
        print("All tests completed successfully!")
    except Exception as e:
        print(f"Test failed with error: {e}")
        import traceback
        traceback.print_exc()