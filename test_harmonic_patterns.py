#!/usr/bin/env python3
"""
Test script for harmonic pattern recognizers with multi-timeframe support
"""

import sys
import pandas as pd
import numpy as np
from pathlib import Path

# Add project root to path
project_root = Path(__file__).resolve().parent
sys.path.insert(0, str(project_root))

def test_harmonic_patterns():
    """Test all harmonic pattern recognizers with multi-timeframe support."""
    patterns = []

    try:
        # Import harmonic pattern recognizers
        from ztb.trading.strategies.action_signal_guide.pattern_recognition.harmonic_patterns import (
            GartleyRecognizer,
            ButterflyRecognizer,
            BatRecognizer,
            CrabRecognizer
        )

        patterns = [
            ("Gartley", GartleyRecognizer),
            ("Butterfly", ButterflyRecognizer),
            ("Bat", BatRecognizer),
            ("Crab", CrabRecognizer)
        ]

    except ImportError as e:
        print(f"Import error: {e}")
        return False

    # Create sample data with more realistic price movements for harmonic patterns
    dates = pd.date_range('2023-01-01', periods=200, freq='h')

    # Create a trending price series that might form harmonic patterns
    base_price = 100
    trend = np.linspace(0, 50, 200)  # Upward trend
    noise = np.random.normal(0, 2, 200)  # Add some noise
    close_prices = base_price + trend + noise

    # Ensure prices don't go negative and add some volatility
    close_prices = np.maximum(close_prices, 50)

    # Create OHLC data
    data = pd.DataFrame({
        'close': close_prices,
        'high': close_prices + np.random.uniform(1, 5, 200),
        'low': close_prices - np.random.uniform(1, 5, 200),
        'open': close_prices + np.random.normal(0, 1, 200),
        'volume': np.random.uniform(1000, 10000, 200)
    }, index=dates)

    # Ensure high >= max(open, close) and low <= min(open, close)
    data['high'] = np.maximum(data[['open', 'close', 'high']].max(axis=1), data['high'])
    data['low'] = np.minimum(data[['open', 'close', 'low']].min(axis=1), data['low'])

    # Create mock multi-timeframe data
    multi_timeframe_data = {
        '4h': {
            'data': pd.DataFrame({
                'close': np.random.uniform(100, 200, 50),
                'high': np.random.uniform(105, 205, 50),
                'low': np.random.uniform(95, 195, 50),
                'open': np.random.uniform(100, 200, 50),
                'volume': np.random.uniform(1000, 10000, 50)
            }, index=pd.date_range('2023-01-01', periods=50, freq='4h'))
        },
        '1d': {
            'data': pd.DataFrame({
                'close': np.random.uniform(100, 200, 14),
                'high': np.random.uniform(105, 205, 14),
                'low': np.random.uniform(95, 195, 14),
                'open': np.random.uniform(100, 200, 14),
                'volume': np.random.uniform(1000, 10000, 14)
            }, index=pd.date_range('2023-01-01', periods=14, freq='D'))
        }
    }

    results = {}

    for pattern_name, pattern_class in patterns:
        try:
            # Create mock config
            config = {
                'lookback_period': 60,
                'tolerance': 0.05
            }

            # Initialize recognizer
            recognizer = pattern_class(config)

            # Test pattern recognition with multi-timeframe data
            signal = recognizer.recognize(data, index=-1, multi_timeframe_data=multi_timeframe_data)

            results[pattern_name] = {
                'success': True,
                'signal': signal,
                'signal_type': signal.signal_type if signal else None,
                'strength': signal.strength if signal else None,
                'direction': signal.direction if signal else None,
                'has_mtf': 'mtf' in (signal.metadata.get('mtf_confidence', 1.0) != 1.0) if signal else False
            }

            status = "✅"
            mtf_indicator = " (MTF)" if results[pattern_name]['has_mtf'] else ""
            print(f"{status} {pattern_name}{mtf_indicator}: Generated signal")
            if signal:
                mtf_conf = signal.metadata.get('mtf_confidence', 1.0)
                print(f"   Type: {signal.signal_type}, Strength: {signal.strength:.6f}, Direction: {signal.direction}, MTF: {mtf_conf:.3f}")

        except Exception as e:
            results[pattern_name] = {
                'success': False,
                'error': str(e)
            }
            print(f"❌ {pattern_name}: Failed - {e}")
            import traceback
            traceback.print_exc()

    # Summary
    successful = sum(1 for r in results.values() if r['success'])
    total = len(results)
    mtf_signals = sum(1 for r in results.values() if r.get('has_mtf', False))

    print(f"\n📊 Harmonic Pattern Test Results: {successful}/{total} patterns successful")
    print(f"🎯 Multi-timeframe signals: {mtf_signals}/{successful} successful patterns")

    if successful == total:
        print("🎉 All harmonic pattern recognizers implemented successfully!")
        if mtf_signals > 0:
            print(f"🚀 Multi-timeframe functionality working ({mtf_signals} patterns using MTF)")
        return True
    else:
        print("⚠️  Some harmonic patterns failed")
        return False

if __name__ == "__main__":
    print("Testing Harmonic Pattern Recognizers with Multi-Timeframe Support...")
    success = test_harmonic_patterns()
    sys.exit(0 if success else 1)