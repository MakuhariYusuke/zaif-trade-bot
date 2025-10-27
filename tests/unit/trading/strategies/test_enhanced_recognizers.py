#!/usr/bin/env python3
"""
Test script for enhanced pattern recognizers using existing features.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..'))

import numpy as np
import pandas as pd
from ztb.trading.strategies.action_signal_guide.pattern_recognition.heikin_ashi import HeikinAshiRecognizer
from ztb.trading.strategies.action_signal_guide.pattern_recognition.granville_law import GranvilleLawRecognizer
from ztb.trading.strategies.action_signal_guide.pattern_recognition.dow_theory import DowTheoryRecognizer

def test_enhanced_recognizers():
    """Test enhanced pattern recognizers with existing features."""

    # Create larger test data for better analysis
    dates = pd.date_range('2023-01-01', periods=100, freq='D')
    np.random.seed(42)

    # Create more realistic price data with trends
    base_price = 100.0
    prices = []
    volumes = []

    for i in range(100):
        # Add some trend and noise
        trend = i * 0.1  # Upward trend
        noise = np.random.normal(0, 2)
        price = base_price + trend + noise
        prices.append(max(price, 1.0))  # Ensure positive prices

        # Volume with some correlation to price changes
        vol_noise = np.random.normal(1000, 200)
        volume = max(500 + vol_noise, 100)
        volumes.append(volume)

    data = pd.DataFrame({
        'open': prices,
        'high': [p + abs(np.random.normal(0, 1)) for p in prices],
        'low': [p - abs(np.random.normal(0, 1)) for p in prices],
        'close': prices,
        'volume': volumes
    }, index=dates)

    # Adjust high/low to be realistic
    data['high'] = np.maximum(data['high'], data[['open', 'close']].max(axis=1))
    data['low'] = np.minimum(data['low'], data[['open', 'close']].min(axis=1))

    try:
        # Test HeikinAshiRecognizer with existing feature
        ha_recognizer = HeikinAshiRecognizer()
        ha_signal = ha_recognizer.recognize(data)
        print(f"✓ HeikinAshiRecognizer: {'Signal found' if ha_signal else 'No signal'}")
        if ha_signal:
            print(f"  Direction: {ha_signal.direction}, Strength: {ha_signal.strength:.3f}")

        # Test GranvilleLawRecognizer with OBV
        gl_recognizer = GranvilleLawRecognizer({'use_obv': True})
        gl_signal = gl_recognizer.recognize(data)
        print(f"✓ GranvilleLawRecognizer (with OBV): {'Signal found' if gl_signal else 'No signal'}")
        if gl_signal:
            print(f"  Direction: {gl_signal.direction}, Strength: {gl_signal.strength:.3f}")

        # Test DowTheoryRecognizer with SuperTrend and Bollinger Bands
        dt_recognizer = DowTheoryRecognizer({
            'use_supertrend': True,
            'use_bollinger': True,
            'primary_trend_period': 30  # Shorter for test data
        })
        dt_signal = dt_recognizer.recognize(data)
        print(f"✓ DowTheoryRecognizer (enhanced): {'Signal found' if dt_signal else 'No signal'}")
        if dt_signal:
            print(f"  Direction: {dt_signal.direction}, Strength: {dt_signal.strength:.3f}")

        return True

    except Exception as e:
        print(f"✗ Enhanced recognizers test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_enhanced_recognizers()
    sys.exit(0 if success else 1)