#!/usr/bin/env python3
"""
Test script for all oscillator pattern recognizers with multi-timeframe support
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Add project root to path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))


def test_oscillator_patterns():
    """Test all oscillator pattern recognizers."""
    patterns = []

    try:
        from ztb.trading.strategies.action_signal_guide.pattern_recognition.oscillator_patterns import (
            CCIRecognizer,
            MFIRecognizer,
            StochasticRecognizer,
            WilliamsRRecognizer,
        )

        patterns = [
            ("CCI", CCIRecognizer),
            ("Stochastic", StochasticRecognizer),
            ("Williams %R", WilliamsRRecognizer),
            ("MFI", MFIRecognizer),
        ]

    except ImportError as e:
        print(f"Import error: {e}")
        return False

    # Create sample data
    dates = pd.date_range("2023-01-01", periods=100, freq="h")
    data = pd.DataFrame(
        {
            "close": np.random.uniform(100, 200, 100),
            "high": np.random.uniform(105, 205, 100),
            "low": np.random.uniform(95, 195, 100),
            "open": np.random.uniform(100, 200, 100),
            "volume": np.random.uniform(1000, 10000, 100),
        },
        index=dates,
    )

    results = {}

    for pattern_name, pattern_class in patterns:
        try:
            # Create mock config with multi-timeframe enabled
            config = {
                "enable_multi_timeframe": True,
                "mtf_timeframes": ["1h", "4h", "1d"],
                "regime_aware": True,
            }

            # Initialize recognizer
            recognizer = pattern_class(config)

            # Test pattern recognition
            signal = recognizer.recognize(data)

            results[pattern_name] = {
                "success": True,
                "signal": signal,
                "signal_type": signal.signal_type if signal else None,
                "strength": signal.strength if signal else None,
                "direction": signal.direction if signal else None,
            }

            print(f"✅ {pattern_name}: Generated signal")
            if signal:
                print(
                    f"   Type: {signal.signal_type}, Strength: {signal.strength:.3f}, Direction: {signal.direction}"
                )

        except Exception as e:
            results[pattern_name] = {"success": False, "error": str(e)}
            print(f"❌ {pattern_name}: Failed - {e}")
            import traceback

            traceback.print_exc()

    # Summary
    successful = sum(1 for r in results.values() if r["success"])
    total = len(results)

    print(f"\n📊 Test Results: {successful}/{total} patterns successful")

    if successful == total:
        print("🎉 All oscillator patterns implemented successfully!")
        return True
    else:
        print("⚠️  Some patterns failed")
        return False


if __name__ == "__main__":
    print("Testing Oscillator Pattern Recognizers with Multi-Timeframe Support...")
    success = test_oscillator_patterns()
    sys.exit(0 if success else 1)
