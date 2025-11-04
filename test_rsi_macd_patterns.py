#!/usr/bin/env python3
"""
Simple test script for RSI and MACD pattern recognizers
"""

import sys
import pandas as pd
import numpy as np
from pathlib import Path

# Add project root to path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

def test_rsi_pattern():
    """Test RSI pattern recognizer."""
    try:
        from ztb.trading.strategies.action_signal_guide.pattern_recognition.rsi import RSIPatternRecognizer

        # Create mock config
        config = {
            'rsi_period': 14,
            'overbought_level': 70,
            'oversold_level': 30,
            'enable_multi_timeframe': True,
            'mtf_timeframes': ['1h', '4h', '1d']
        }

        # Create sample data
        dates = pd.date_range('2023-01-01', periods=100, freq='h')
        data = pd.DataFrame({
            'close': np.random.uniform(100, 200, 100),
            'high': np.random.uniform(105, 205, 100),
            'low': np.random.uniform(95, 195, 100),
            'open': np.random.uniform(100, 200, 100),
            'volume': np.random.uniform(1000, 10000, 100)
        }, index=dates)

        # Initialize recognizer
        recognizer = RSIPatternRecognizer(config)

        # Test pattern recognition
        signal = recognizer.recognize(data)

        print(f"RSI Pattern Test: Generated signal: {signal}")
        if signal:
            print(f"Signal details: type={signal.signal_type}, strength={signal.strength}, direction={signal.direction}")

        return True

    except Exception as e:
        print(f"RSI Pattern Test Failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_macd_pattern():
    """Test MACD pattern recognizer."""
    try:
        from ztb.trading.strategies.action_signal_guide.pattern_recognition.macd import MACDPatternRecognizer

        # Create mock config
        config = {
            'fast_period': 12,
            'slow_period': 26,
            'signal_period': 9,
            'enable_multi_timeframe': True,
            'mtf_timeframes': ['1h', '4h', '1d']
        }

        # Create sample data
        dates = pd.date_range('2023-01-01', periods=100, freq='h')
        data = pd.DataFrame({
            'close': np.random.uniform(100, 200, 100),
            'high': np.random.uniform(105, 205, 100),
            'low': np.random.uniform(95, 195, 100),
            'open': np.random.uniform(100, 200, 100),
            'volume': np.random.uniform(1000, 10000, 100)
        }, index=dates)

        # Initialize recognizer
        recognizer = MACDPatternRecognizer(config)

        # Test pattern recognition
        signal = recognizer.recognize(data)

        print(f"MACD Pattern Test: Generated signal: {signal}")
        if signal:
            print(f"Signal details: type={signal.signal_type}, strength={signal.strength}, direction={signal.direction}")

        return True

    except Exception as e:
        print(f"MACD Pattern Test Failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("Testing RSI and MACD Pattern Recognizers...")

    rsi_success = test_rsi_pattern()
    macd_success = test_macd_pattern()

    if rsi_success and macd_success:
        print("All tests passed!")
    else:
        print("Some tests failed!")
        sys.exit(1)