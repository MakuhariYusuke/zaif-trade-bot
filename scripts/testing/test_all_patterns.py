#!/usr/bin/env python3
"""
Comprehensive test script for all pattern recognizers with multi-timeframe support
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Add project root to path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))


def test_all_patterns():
    """Test all pattern recognizers with multi-timeframe support."""
    patterns = []

    try:
        # Import all pattern recognizers
        from ztb.trading.strategies.action_signal_guide.pattern_recognition.macd import (
            MACDPatternRecognizer,
        )
        from ztb.trading.strategies.action_signal_guide.pattern_recognition.oscillator_patterns import (
            CCIRecognizer,
            MFIRecognizer,
            StochasticRecognizer,
            WilliamsRRecognizer,
        )
        from ztb.trading.strategies.action_signal_guide.pattern_recognition.rsi import (
            RSIPatternRecognizer,
        )

        patterns = [
            ("RSI", RSIPatternRecognizer),
            ("MACD", MACDPatternRecognizer),
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

    # Create mock multi-timeframe data
    multi_timeframe_data = {
        "4h": {
            "data": pd.DataFrame(
                {
                    "close": np.random.uniform(100, 200, 25),
                    "high": np.random.uniform(105, 205, 25),
                    "low": np.random.uniform(95, 195, 25),
                    "open": np.random.uniform(100, 200, 25),
                    "volume": np.random.uniform(1000, 10000, 25),
                },
                index=pd.date_range("2023-01-01", periods=25, freq="4h"),
            )
        },
        "1d": {
            "data": pd.DataFrame(
                {
                    "close": np.random.uniform(100, 200, 7),
                    "high": np.random.uniform(105, 205, 7),
                    "low": np.random.uniform(95, 195, 7),
                    "open": np.random.uniform(100, 200, 7),
                    "volume": np.random.uniform(1000, 10000, 7),
                },
                index=pd.date_range("2023-01-01", periods=7, freq="D"),
            )
        },
    }

    results = {}

    for pattern_name, pattern_class in patterns:
        try:
            # Create mock config with multi-timeframe enabled
            config = {
                "enable_multi_timeframe": True,
                "mtf_timeframes": ["4h", "1d"],
                "regime_aware": True,
            }

            # Initialize recognizer
            recognizer = pattern_class(config)

            # Test pattern recognition with multi-timeframe data
            signal = recognizer.recognize(
                data, multi_timeframe_data=multi_timeframe_data
            )

            results[pattern_name] = {
                "success": True,
                "signal": signal,
                "signal_type": signal.signal_type if signal else None,
                "strength": signal.strength if signal else None,
                "direction": signal.direction if signal else None,
                "has_mtf": "mtf" in (signal.signal_type or "") if signal else False,
            }

            status = "✅"
            mtf_indicator = " (MTF)" if results[pattern_name]["has_mtf"] else ""
            print(f"{status} {pattern_name}{mtf_indicator}: Generated signal")
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
    mtf_signals = sum(1 for r in results.values() if r.get("has_mtf", False))

    print(f"\n📊 Test Results: {successful}/{total} patterns successful")
    print(f"🎯 Multi-timeframe signals: {mtf_signals}/{successful} successful patterns")

    if successful == total:
        print("🎉 All pattern recognizers implemented successfully!")
        if mtf_signals > 0:
            print(
                f"🚀 Multi-timeframe functionality working ({mtf_signals} patterns using MTF)"
            )
        return True
    else:
        print("⚠️  Some patterns failed")
        return False


if __name__ == "__main__":
    print("Testing All Pattern Recognizers with Multi-Timeframe Support...")
    success = test_all_patterns()
    sys.exit(0 if success else 1)
