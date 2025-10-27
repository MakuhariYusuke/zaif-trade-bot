#!/usr/bin/env python3
"""
Test script for Ichimoku Cloud Pattern Recognizer.
一目均衡表パターン認識器のテストスクリプト
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..', '..'))

import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Import Ichimoku pattern recognizer
from ztb.trading.strategies.action_signal_guide.pattern_recognition.ichimoku import IchimokuPatternRecognizer


def generate_test_data(length: int = 200) -> pd.DataFrame:
    """Generate synthetic OHLCV data for testing Ichimoku patterns."""
    np.random.seed(42)

    # Generate base price series with trend changes
    base_price = 100
    prices = [base_price]

    for i in range(length - 1):
        # Create multiple trend phases for Ichimoku testing
        if i < length // 4:
            trend = 0.001  # Initial uptrend
        elif i < length // 2:
            trend = -0.001  # Downtrend
        elif i < 3 * length // 4:
            trend = 0.002  # Strong uptrend
        else:
            trend = -0.0005  # Weak downtrend

        volatility = 0.02 + 0.01 * np.sin(i * 0.1)  # Varying volatility
        noise = np.random.normal(0, volatility)
        new_price = prices[-1] * (1 + trend + noise)
        prices.append(max(new_price, 0.1))

    # Generate OHLCV from price series with realistic spreads
    data = []
    for i, close in enumerate(prices):
        volatility_factor = 0.005 + 0.01 * np.sin(i * 0.05)
        high = close * (1 + volatility_factor * np.random.uniform(0.5, 1.5))
        low = close * (1 - volatility_factor * np.random.uniform(0.5, 1.5))
        open_price = data[-1]['close'] if data else close * (1 + np.random.normal(0, 0.002))
        volume = np.random.uniform(10000, 100000)

        data.append({
            'timestamp': datetime.now() + timedelta(minutes=i),
            'open': open_price,
            'high': high,
            'low': low,
            'close': close,
            'volume': volume
        })

    return pd.DataFrame(data)


def test_ichimoku_recognizer():
    """Test Ichimoku Cloud Pattern Recognizer."""
    print("Testing Ichimoku Cloud Pattern Recognizer")
    print("=" * 50)

    # Generate test data
    test_data = generate_test_data(200)
    print(f"Generated test data with {len(test_data)} rows")

    # Initialize recognizer with custom config
    config = {
        'tenkan_kijun_threshold': 0.02,
        'cloud_expansion_threshold': 0.1,
        'wave_theory_threshold': 0.15,
        'time_theory_threshold': 0.2,
        'value_measurement_threshold': 0.25,
        'momentum_confirmation_threshold': 0.3,
        'sanyaku_kouten_threshold': 0.8
    }

    recognizer = IchimokuPatternRecognizer(config)

    # Test recognition at different indices
    test_indices = [50, 100, 150, -1]  # Different points in the data

    for idx in test_indices:
        print(f"\n--- Testing at index {idx} ---")

        try:
            result = recognizer.recognize(test_data, idx)
            if result:
                print("✓ Signal detected:")
                print(f"  Type: {result.signal_type}")
                print(f"  Direction: {result.direction}")
                print(f"  Strength: {result.strength:.3f}")
                print(f"  Confidence: {result.confidence:.3f}")
                print(f"  Risk Level: {result.risk_level}")
                print(f"  Validity Period: {result.validity_period}")
                print(f"  Description: {result.description}")
            else:
                print("- No signal detected at this index")

        except Exception as e:
            print(f"✗ Error at index {idx}: {str(e)}")

    print("\n" + "=" * 50)
    print("Ichimoku Cloud Pattern Recognizer test completed!")


def test_ichimoku_components():
    """Test individual Ichimoku components."""
    print("\nTesting Individual Ichimoku Components")
    print("=" * 40)

    test_data = generate_test_data(100)

    try:
        from ztb.features.trend.ichimoku.ichimoku import compute_ichimoku_diff_norm
        diff_norm = compute_ichimoku_diff_norm(test_data)
        print(f"✓ Ichimoku Diff Norm: {diff_norm.iloc[-1]:.4f}")

        from ztb.features.trend.ichimoku.ichimoku_cloud_expansion import compute_ichimoku_cloud_expansion
        cloud_exp = compute_ichimoku_cloud_expansion(test_data)
        print(f"✓ Cloud Expansion: {cloud_exp.iloc[-1]:.4f}")

        from ztb.features.trend.ichimoku.ichimoku_wave_theory import compute_ichimoku_wave_theory
        wave_theory = compute_ichimoku_wave_theory(test_data)
        print(f"✓ Wave Theory: {wave_theory.iloc[-1]:.4f}")

        print("All Ichimoku components calculated successfully!")

    except Exception as e:
        print(f"✗ Error testing components: {str(e)}")


if __name__ == "__main__":
    test_ichimoku_recognizer()
    test_ichimoku_components()