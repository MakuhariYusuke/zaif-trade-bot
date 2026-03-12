#!/usr/bin/env python3
"""
Test script for Action Signal Guide pattern validation functionality.
"""

import os
import sys

import numpy as np
import pandas as pd

# Add the project root to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", ".."))

from ztb.trading.strategies.action_signal_guide.action_signal_guide import (
    ActionSignalGuide,
    ActionSignalGuideConfig,
    RecognizerConfig,
)


def create_sample_data():
    """Create sample OHLCV data for testing."""
    dates = pd.date_range(start="2023-01-01", end="2023-12-31", freq="D")
    np.random.seed(42)

    # Generate realistic-looking price data
    n = len(dates)
    base_price = 100
    prices = [base_price]

    for i in range(1, n):
        change = np.random.normal(0, 0.02)  # 2% daily volatility
        new_price = prices[-1] * (1 + change)
        prices.append(max(new_price, 0.01))  # Prevent negative prices

    # Create OHLCV data
    data = []
    for i, price in enumerate(prices):
        high = price * (1 + abs(np.random.normal(0, 0.01)))
        low = price * (1 - abs(np.random.normal(0, 0.01)))
        open_price = prices[i - 1] if i > 0 else price
        volume = np.random.randint(1000, 10000)

        data.append(
            {
                "timestamp": dates[i],
                "open": open_price,
                "high": high,
                "low": low,
                "close": price,
                "volume": volume,
            }
        )

    return pd.DataFrame(data)


def test_pattern_validation():
    """Test the pattern validation functionality."""
    print("Testing Action Signal Guide Pattern Validation")
    print("=" * 50)

    # Create sample data
    data = create_sample_data()
    print(f"Created sample data with {len(data)} rows")

    # Create config with selective pattern enabling - only enable working patterns
    config = ActionSignalGuideConfig(
        enable_candlestick_patterns=True,
        enable_fibonacci_patterns=False,  # Disable problematic patterns
        enable_gann_patterns=False,
        enable_wave_patterns=False,
        enable_harmonic_patterns=False,
        enable_oscillator_patterns=True,  # Enable ATR patterns
        enable_volume_patterns=True,
        enable_bollinger_patterns=True,
        enable_adx_patterns=False,
        enable_granville_patterns=False,
        enable_heikin_ashi_patterns=False,
        enable_dow_theory_patterns=False,
        max_signals_per_bar=5,
        enable_caching=True,
        cache_size=1000,
        oscillator_patterns=[
            RecognizerConfig("cci", group="oscillator"),
            RecognizerConfig("stochastic", group="oscillator"),
            RecognizerConfig("williamsr", group="oscillator"),
            RecognizerConfig("mfi", group="oscillator"),
            RecognizerConfig("atr", group="oscillator"),  # Add ATR pattern
        ],
    )

    # Create Action Signal Guide
    guide = ActionSignalGuide(config=config)
    print("Created ActionSignalGuide with selective pattern enabling")

    # Generate signals for multiple bars
    signals_generated = 0
    for i in range(min(100, len(data))):  # Test first 100 bars
        signals = guide.generate_signals(data, i)
        signals_generated += len(signals)

    print(f"Generated {signals_generated} signals across {min(100, len(data))} bars")

    # Analyze pattern effectiveness
    analysis = guide.analyze_pattern_effectiveness()
    print("\nPattern Effectiveness Analysis:")
    print(f"Total signals: {analysis['total_signals']}")
    print(f"Enabled patterns: {analysis['enabled_patterns']}")
    print(f"Disabled patterns: {analysis['disabled_patterns']}")

    print("\nPattern Statistics:")
    for pattern, stats in analysis["pattern_stats"].items():
        enabled = "ENABLED" if pattern in analysis["enabled_patterns"] else "DISABLED"
        print(f"  {pattern} ({enabled}): {stats['signals_generated']} signals")

    # Generate validation report
    report = guide.generate_validation_report()
    print("\nValidation Report:")
    print(report)

    # Test with mock trading results
    mock_trading_results = [
        {
            "profit": 100.50,
            "win_rate": 0.65,
            "sharpe_ratio": 1.2,
            "max_drawdown": -0.15,
            "signals": [
                {"source_patterns": ["candlestick", "oscillator"]},
                {"source_patterns": ["fibonacci"]},
            ],
        },
        {
            "profit": -50.25,
            "win_rate": 0.45,
            "sharpe_ratio": -0.8,
            "max_drawdown": -0.25,
            "signals": [
                {"source_patterns": ["volume", "bollinger"]},
                {"source_patterns": ["oscillator", "volume"]},
            ],
        },
        {
            "profit": 75.75,
            "win_rate": 0.70,
            "sharpe_ratio": 1.5,
            "max_drawdown": -0.10,
            "signals": [
                {"source_patterns": ["candlestick"]},
                {"source_patterns": ["fibonacci", "oscillator"]},
            ],
        },
    ]

    analysis_with_correlations = guide.analyze_pattern_effectiveness(
        mock_trading_results
    )
    print("\nAnalysis with Trading Correlations:")
    if "correlation_analysis" in analysis_with_correlations:
        for pattern, correlations in analysis_with_correlations[
            "correlation_analysis"
        ].items():
            print(f"  {pattern}:")
            for metric, value in correlations.items():
                print(f"    {metric}: {value:.3f}")

    print("\nTest completed successfully!")


if __name__ == "__main__":
    test_pattern_validation()
