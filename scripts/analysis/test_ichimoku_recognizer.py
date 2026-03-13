#!/usr/bin/env python3
"""
Test script for Ichimoku Pattern Recognizer
一目均衡表パターン認識器のテストスクリプト
"""

import sys
import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Add the project root to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from ztb.trading.strategies.action_signal_guide.pattern_recognition.ichimoku import IchimokuPatternRecognizer

def create_test_data():
    """Create test OHLCV data for testing"""
    np.random.seed(42)
    n_points = 200

    # Generate base price data
    base_price = 5000000.0  # JPY-based price
    prices = []
    for i in range(n_points):
        change = np.random.normal(0, 0.02)
        base_price *= (1 + change)
        prices.append(base_price)

    # Create OHLCV data
    data = []
    for i, price in enumerate(prices):
        high = price * (1 + abs(np.random.normal(0, 0.01)))
        low = price * (1 - abs(np.random.normal(0, 0.01)))
        open_price = prices[i-1] if i > 0 else price
        volume = np.random.uniform(1000, 10000)

        data.append({
            'timestamp': datetime.now() - timedelta(hours=n_points-i),
            'open': open_price,
            'high': high,
            'low': low,
            'close': price,
            'volume': volume
        })

    df = pd.DataFrame(data)
    df.set_index('timestamp', inplace=True)
    return df

def test_ichimoku_recognizer():
    """Test the Ichimoku pattern recognizer"""
    print("Testing Ichimoku Pattern Recognizer...")
    print("=" * 50)

    # Create test data
    data = create_test_data()
    print(f"Created test data with {len(data)} points")
    print(f"Data type: {type(data)}")
    print(f"Data columns: {list(data.columns)}")
    print(f"Data types:\n{data.dtypes}")
    print(f"Sample data:\n{data.head()}")
    print()

    # Initialize recognizer
    try:
        recognizer = IchimokuPatternRecognizer()
        print("✓ IchimokuPatternRecognizer initialized successfully")
    except Exception as e:
        print(f"✗ Failed to initialize recognizer: {e}")
        return False

    # Test pattern recognition
    try:
        print("Testing pattern recognition...")
        signal = recognizer.recognize(data)

        if signal:
            print("✓ Recognition completed. Found signal:")
            print(f"    Type: {signal.signal_type}")
            print(f"    Strength: {signal.strength}")
            print(f"    Direction: {signal.direction}")
            print(f"    Confidence: {signal.confidence}")
            print(f"    Risk Level: {signal.risk_level}")
            print(f"    Valid Until: {signal.validity_period}")
        else:
            print("✓ Recognition completed. No signal found")

        # Test caching
        print("Testing caching functionality...")
        signal_cached = recognizer.recognize_with_cache(data)
        if signal_cached:
            print("✓ Cached recognition completed. Found signal")
        else:
            print("✓ Cached recognition completed. No signal found")

        return True

    except Exception as e:
        print(f"✗ Pattern recognition failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_ichimoku_recognizer()
    sys.exit(0 if success else 1)