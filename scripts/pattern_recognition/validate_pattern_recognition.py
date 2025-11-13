#!/usr/bin/env python3
"""
Pattern Recognition Validation Script

This script validates that all pattern recognizers have been updated to use
continuous direction values [-1, 1] instead of discrete ACTION_BUY/ACTION_SELL.
"""

import logging
import os
import sys
from datetime import datetime, timedelta
from typing import List

import numpy as np
import pandas as pd

# Setup logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Add project root to path
project_root = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)
sys.path.insert(0, project_root)

# Import all pattern recognizers
from ztb.trading.strategies.action_signal_guide.pattern_recognition import (
    adx_patterns,
    atr,
    bollinger_patterns,
    candlestick_patterns,
    dow_theory,
    fibonacci_patterns,
    gann_analysis,
    granville_law,
    ichimoku,
    macd,
    oscillator_patterns,
    rsi,
    volume_patterns,
)
from ztb.trading.strategies.action_signal_guide.pattern_recognition.base import (
    PatternRecognizer,
    SignalResult,
)


def create_sample_data(num_bars: int = 200) -> pd.DataFrame:
    """Create sample OHLCV data for testing."""
    np.random.seed(42)  # For reproducible results

    # Generate base price series with trend
    base_price = 5000000.0  # JPY-based price
    prices: List[float] = []

    for i in range(num_bars):
        # Add some trend and noise
        trend = 0.001 * i  # Slight upward trend
        noise = np.random.normal(0, 0.02)  # Random noise
        price = base_price * (1 + trend + noise)
        prices.append(price)

    # Create OHLCV data
    timestamps = [
        datetime.now() - timedelta(hours=num_bars - i) for i in range(num_bars)
    ]
    opens = [prices[0] * (1 + np.random.normal(0, 0.005))]

    data = []
    for i, close in enumerate(prices):
        high = close * (1 + abs(np.random.normal(0, 0.01)))
        low = close * (1 - abs(np.random.normal(0, 0.01)))
        volume = int(np.random.normal(1000, 200))

        if i < len(prices) - 1:
            opens.append(close * (1 + np.random.normal(0, 0.005)))

        data.append(
            {
                "timestamp": timestamps[i],
                "open": opens[i] if i < len(opens) else close,
                "high": high,
                "low": low,
                "close": close,
                "volume": volume,
            }
        )

    df = pd.DataFrame(data)
    df.set_index("timestamp", inplace=True)
    return df


def get_all_recognizers():
    """Get all pattern recognizers for testing."""
    recognizers = []

    # RSI
    recognizers.append(rsi.RSIPatternRecognizer())

    # MACD
    recognizers.append(macd.MACDPatternRecognizer())

    # Bollinger Bands
    recognizers.append(bollinger_patterns.BollingerBandsRecognizer())

    # Ichimoku
    recognizers.append(ichimoku.IchimokuPatternRecognizer())

    # Fibonacci
    recognizers.append(fibonacci_patterns.FibonacciRetracementRecognizer())
    recognizers.append(fibonacci_patterns.FibonacciExtensionRecognizer())
    recognizers.append(fibonacci_patterns.FibonacciProjectionRecognizer())

    # Gann Analysis
    recognizers.append(gann_analysis.GannAngleRecognizer())
    recognizers.append(gann_analysis.GannSquareRecognizer())
    recognizers.append(gann_analysis.GannTimeClusterRecognizer())

    # ADX
    recognizers.append(adx_patterns.ADXRecognizer())

    # ATR
    recognizers.append(atr.ATRPatternRecognizer())

    # Candlestick
    recognizers.append(candlestick_patterns.HammerRecognizer())
    recognizers.append(candlestick_patterns.HangingManRecognizer())
    recognizers.append(candlestick_patterns.MorningStarRecognizer())
    recognizers.append(candlestick_patterns.EveningStarRecognizer())

    # Dow Theory
    recognizers.append(dow_theory.DowTheoryRecognizer())

    # Granville Law
    recognizers.append(granville_law.GranvilleLawRecognizer())

    # Oscillator Patterns
    recognizers.append(oscillator_patterns.StochasticRecognizer())

    # Volume Patterns
    recognizers.append(volume_patterns.ChaikinADRecognizer())

    return recognizers


def validate_recognizer(recognizer: PatternRecognizer, data: pd.DataFrame) -> bool:
    """Validate a single recognizer."""
    recognizer_name = recognizer.__class__.__name__
    logger.info(f"Testing {recognizer_name}...")

    try:
        # Test recognition at different indices
        test_indices = [50, 100, 150, -1]

        for idx in test_indices:
            signal = recognizer.recognize(data, index=idx)

            if signal is not None:
                # Validate signal structure
                assert isinstance(
                    signal, SignalResult
                ), "Signal should be SignalResult instance"

                # Validate direction is continuous [-1, 1]
                assert (
                    -1.0 <= signal.direction <= 1.0
                ), f"Direction {signal.direction} should be in [-1, 1]"

                # Validate strength is [0, 1]
                assert (
                    0.0 <= signal.strength <= 1.0
                ), f"Strength {signal.strength} should be in [0, 1]"

                # Log successful signal
                logger.info(
                    f"  ✓ {recognizer_name} at index {idx}: direction={signal.direction:.3f}, strength={signal.strength:.3f}"
                )

                # Check metadata contains market adaptation info
                if hasattr(signal, "metadata") and signal.metadata:
                    if "volatility_ratio" in signal.metadata:
                        logger.info(
                            f"    Volatility ratio: {signal.metadata['volatility_ratio']:.3f}"
                        )
                    if "trend_strength" in signal.metadata:
                        logger.info(
                            f"    Trend strength: {signal.metadata['trend_strength']:.3f}"
                        )

        logger.info(f"✅ {recognizer_name} validation passed")
        return True

    except Exception as e:
        logger.error(f"❌ {recognizer_name} validation failed: {e}")
        import traceback

        traceback.print_exc()
        return False


def main():
    """Main validation function."""
    logger.info("Starting Pattern Recognition Validation")
    logger.info("=" * 50)

    # Create sample data
    logger.info("Creating sample data...")
    data = create_sample_data(200)
    logger.info(f"Created {len(data)} bars of sample data")

    # Get all recognizers
    recognizers = get_all_recognizers()
    logger.info(f"Testing {len(recognizers)} pattern recognizers")

    # Validate each recognizer
    passed = 0
    failed = 0

    for recognizer in recognizers:
        if validate_recognizer(recognizer, data):
            passed += 1
        else:
            failed += 1

    # Summary
    logger.info("=" * 50)
    logger.info("Validation Summary:")
    logger.info(f"  Passed: {passed}")
    logger.info(f"  Failed: {failed}")
    logger.info(f"  Total:  {passed + failed}")

    if failed == 0:
        logger.info("🎉 All pattern recognizers validated successfully!")
        return True
    else:
        logger.error(f"❌ {failed} pattern recognizers failed validation")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
