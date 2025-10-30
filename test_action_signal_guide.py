#!/usr/bin/env python3
"""
Quick test of ActionSignalGuide to see if it generates signals.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Add project root to path
project_root = Path(__file__).resolve().parent
sys.path.insert(0, str(project_root))

from ztb.trading.strategies.action_signal_guide.action_signal_guide import (
    ActionSignalGuide,
    ActionSignalGuideConfig,
    GuidanceLevel,
)


def create_test_data():
    """Create test data with specific candlestick patterns."""
    dates = pd.date_range("2023-01-01", periods=100, freq="h")
    np.random.seed(42)

    # Create OHLCV data with some hammer patterns
    data = []

    for i in range(len(dates)):
        if i == 25:  # Create a hammer pattern at index 25
            # Hammer: long lower wick, small body, little/no upper wick
            high = 50300  # Small upper wick
            low = 49000  # Long lower wick
            open_price = 50100
            close = 50200  # Small body near high
            volume = 1000
        elif i == 50:  # Create another hammer pattern at index 50
            # Hammer in downtrend
            high = 50100  # Small upper wick
            low = 48000  # Long lower wick
            open_price = 49900
            close = 50000  # Small body near high
            volume = 1200
        else:
            # Random data
            base_price = 50000 + np.sin(i * 0.1) * 1000
            high = base_price * (1 + abs(np.random.normal(0, 0.02)))
            low = base_price * (1 - abs(np.random.normal(0, 0.02)))
            open_price = base_price + np.random.normal(0, 200)
            close = base_price + np.random.normal(0, 200)
            volume = np.random.randint(100, 1000)

        data.append(
            {
                "timestamp": dates[i],
                "open": open_price,
                "high": high,
                "low": low,
                "close": close,
                "volume": volume,
            }
        )

    df = pd.DataFrame(data)
    df.set_index("timestamp", inplace=True)
    return df


def test_action_signal_guide():
    """Test ActionSignalGuide signal generation."""
    print("Testing ActionSignalGuide...")

    # Create test data
    data = create_test_data()
    print(f"Created test data with {len(data)} rows")
    print(f"Data columns: {list(data.columns)}")
    print(f"Sample data:\n{data.head()}")

    # Create ActionSignalGuide with debug config
    config = ActionSignalGuideConfig(
        debug_short_mode=False,  # Disable debug short mode to run all recognizers
        guidance_level=GuidanceLevel.WEAK,  # Use GuidanceLevel enum
        enable_candlestick_patterns=True,
        enable_fibonacci_patterns=False,
        enable_gann_patterns=False,
        enable_wave_patterns=False,
        enable_harmonic_patterns=False,
        enable_oscillator_patterns=False,
        enable_volume_patterns=False,
        enable_bollinger_patterns=False,
        enable_adx_patterns=False,
        enable_granville_patterns=False,
        enable_heikin_ashi_patterns=False,
        enable_dow_theory_patterns=False,
    )

    guide = ActionSignalGuide(config=config)
    print(f"Initialized with {len(guide.all_recognizers)} recognizers")
    print(f"SignalGenerator guidance_level: {guide.signal_generator.guidance_level}")
    print(f"Config guidance_level: {config.guidance_level}")
    print(f"GuidanceLevel.WEAK: {GuidanceLevel.WEAK}")
    print(
        f"Are they equal? {guide.signal_generator.guidance_level == GuidanceLevel.WEAK}"
    )

    # Print recognizer names
    print("Active recognizers:")
    for i, rec in enumerate(guide.all_recognizers):
        pattern_type = getattr(rec, "pattern_type", "unknown")
        print(f"  {i}: {rec.name} ({pattern_type})")

    # Test individual recognizers with debug
    print("\nTesting individual recognizers...")
    test_recognizer = guide.all_recognizers[3]  # HammerRecognizer
    print(f"Testing {test_recognizer.name}...")

    test_indices_for_recognizer = [25, 50]
    for idx in test_indices_for_recognizer:
        print(f"  Index {idx}: ", end="")
        try:
            # Check trend first
            is_downtrend = test_recognizer._is_downtrend(data, idx, lookback=5)
            print(f"Downtrend: {is_downtrend}, ", end="")

            # Check candle characteristics
            candle = data.iloc[idx]
            body_size = abs(candle["close"] - candle["open"])
            lower_shadow = test_recognizer.calculate_lower_shadow(data, idx)
            upper_shadow = test_recognizer.calculate_upper_shadow(data, idx)
            total_range = candle["high"] - candle["low"]

            print(
                f"Body: {body_size:.2f}, Lower: {lower_shadow:.2f}, Upper: {upper_shadow:.2f}, Range: {total_range:.2f}"
            )

            result = test_recognizer.recognize(data, index=idx)
            if result is not None:
                print(
                    f"    -> Signal found - direction: {result.direction}, strength: {result.strength}, confidence: {result.confidence}"
                )
            else:
                print("    -> No signal")
        except Exception as e:
            print(f"Error: {e}")

    # Test signal generation at different indices
    test_indices = [25, 50, 75, 99]

    for idx in test_indices:
        print(f"\nTesting index {idx}...")
        print(f"Data at index {idx}: {data.iloc[idx]}")

        try:
            signals = guide.generate_signals(data, idx)
            print(f"Generated {len(signals)} signals")

            if signals:
                for i, signal in enumerate(signals):
                    print(
                        f"  Signal {i}: direction={signal.direction:.3f}, confidence={signal.confidence:.3f}, type={signal.signal_type}"
                    )
            else:
                print("  No signals generated")
                # Debug: test signal generator directly
                print("  Debug: Testing signal generator directly...")
                try:
                    # Test individual recognizers to see what signals they produce
                    all_signals = []
                    for rec in guide.signal_generator.all_recognizers:
                        try:
                            result = rec.recognize(data, index=idx)
                            if result is not None:
                                print(
                                    f"    {rec.name}: {result.direction}, {result.strength}, {result.confidence}"
                                )
                                all_signals.append(result)
                            else:
                                print(f"    {rec.name}: No signal")
                        except Exception as e:
                            print(f"    {rec.name}: Error {e}")

                    print(f"    Total signals collected: {len(all_signals)}")

                    # Test filtering
                    print("    Testing guidance level filtering...")
                    try:
                        filtered = guide.signal_generator._filter_by_guidance_level(
                            all_signals
                        )
                        print(f"    Filtered signals: {len(filtered)}")
                        for sig in filtered:
                            print(
                                f"      {sig.signal_type}: {sig.direction}, {sig.strength}, {sig.confidence}"
                            )
                    except Exception as e:
                        print(f"    Filter error: {e}")
                        import traceback

                        traceback.print_exc()

                    sg_signal = guide.signal_generator.generate_signal(data, idx)
                    print(
                        f"    SignalGenerator returned: direction={sg_signal.direction}, strength={sg_signal.strength}, confidence={sg_signal.confidence}"
                    )
                except Exception as e:
                    print(f"    SignalGenerator error: {e}")
                    import traceback

                    traceback.print_exc()
        except Exception as e:
            print(f"  Error generating signals: {e}")
            import traceback

            traceback.print_exc()


if __name__ == "__main__":
    test_action_signal_guide()
