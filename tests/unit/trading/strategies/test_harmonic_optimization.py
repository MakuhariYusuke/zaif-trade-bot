#!/usr/bin/env python3
"""
Test script for optimized harmonic pattern recognition
"""
import time

import numpy as np
import pandas as pd

from ztb.trading.strategies.action_signal_guide.pattern_recognition.harmonic_patterns import (
    BatRecognizer,
    ButterflyRecognizer,
    CrabRecognizer,
    GartleyRecognizer,
)


def generate_test_data(length=1000):
    """Generate synthetic OHLCV data for testing"""
    np.random.seed(42)
    dates = pd.date_range("2023-01-01", periods=length, freq="1H")

    # Generate realistic price data with trends and volatility
    base_price = 50000
    prices = [base_price]

    for i in range(length - 1):
        # Add some trend and random walk
        trend = 0.0001 if i < length // 2 else -0.0001  # Uptrend then downtrend
        change = np.random.normal(trend, 0.01)
        new_price = prices[-1] * (1 + change)
        prices.append(max(new_price, 1000))  # Floor price

    df = pd.DataFrame(
        {
            "open": prices,
            "high": [p * (1 + abs(np.random.normal(0, 0.005))) for p in prices],
            "low": [p * (1 - abs(np.random.normal(0, 0.005))) for p in prices],
            "close": prices[1:] + [prices[-1]],
            "volume": np.random.randint(100, 10000, length),
        },
        index=dates,
    )

    # Ensure high >= max(open, close) and low <= min(open, close)
    for i in range(len(df)):
        df.loc[df.index[i], "high"] = max(
            df.loc[df.index[i], ["open", "close"]].max(), df.loc[df.index[i], "high"]
        )
        df.loc[df.index[i], "low"] = min(
            df.loc[df.index[i], ["open", "close"]].min(), df.loc[df.index[i], "low"]
        )

    return df


def test_recognizer_performance(recognizer_class, data, name):
    """Test performance of a single recognizer"""
    recognizer = recognizer_class(lookback_period=60, tolerance=0.05)

    start_time = time.time()
    signals = []

    # Test recognition at multiple points
    for i in range(100, len(data) - 10, 10):  # Test every 10th point
        signal = recognizer.recognize(data, i)
        if signal:
            signals.append(signal)

    end_time = time.time()
    duration = end_time - start_time

    print(f"{name}: Found {len(signals)} signals in {duration:.3f}s")
    return len(signals), duration


def main():
    print("Testing optimized harmonic pattern recognition...")

    # Generate test data
    data = generate_test_data(2000)
    print(f"Generated {len(data)} data points")

    # Test all recognizers
    recognizers = [
        (GartleyRecognizer, "Gartley"),
        (ButterflyRecognizer, "Butterfly"),
        (BatRecognizer, "Bat"),
        (CrabRecognizer, "Crab"),
    ]

    total_signals = 0
    total_time = 0

    for recognizer_class, name in recognizers:
        signals, duration = test_recognizer_performance(recognizer_class, data, name)
        total_signals += signals
        total_time += duration

    print(f"\nTotal: {total_signals} signals found in {total_time:.3f}s")
    print("Optimization test completed successfully!")


if __name__ == "__main__":
    main()
