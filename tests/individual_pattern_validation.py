#!/usr/bin/env python3
"""
Individual Pattern Validation Script for Action Signal Guide.

Tests each pattern recognition system individually to validate functionality.
"""

import sys
import time
from pathlib import Path
from typing import Dict

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


def create_test_data(n_points: int = 5000) -> pd.DataFrame:
    """Create test data with specific patterns for validation."""
    dates = pd.date_range("2023-01-01", periods=n_points, freq="h")
    np.random.seed(42)

    data = []
    for i in range(len(dates)):
        # Create realistic OHLCV data with some trends and volatility
        base_price = 50000 + np.sin(i * 0.01) * 5000 + np.sin(i * 0.001) * 10000
        volatility = 0.02 + 0.01 * np.sin(i * 0.005)  # Varying volatility

        high = base_price * (1 + abs(np.random.normal(0, volatility)))
        low = base_price * (1 - abs(np.random.normal(0, volatility)))
        open_price = base_price + np.random.normal(0, base_price * volatility * 0.5)
        close = base_price + np.random.normal(0, base_price * volatility * 0.5)
        volume = np.random.randint(100, 10000)

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


# Pattern groups to test individually
PATTERN_RECOGNIZERS = [
    {
        "name": "CANDLESTICK",
        "config": {
            "enable_candlestick_patterns": True,
            "enable_fibonacci_patterns": False,
            "enable_gann_patterns": False,
            "enable_wave_patterns": False,
            "enable_harmonic_patterns": False,
            "enable_oscillator_patterns": False,
            "enable_volume_patterns": False,
            "enable_bollinger_patterns": False,
            "enable_adx_patterns": False,
            "enable_granville_patterns": False,
            "enable_heikin_ashi_patterns": False,
            "enable_dow_theory_patterns": False,
        },
    },
    {
        "name": "FIBONACCI",
        "config": {
            "enable_candlestick_patterns": False,
            "enable_fibonacci_patterns": True,
            "enable_gann_patterns": False,
            "enable_wave_patterns": False,
            "enable_harmonic_patterns": False,
            "enable_oscillator_patterns": False,
            "enable_volume_patterns": False,
            "enable_bollinger_patterns": False,
            "enable_adx_patterns": False,
            "enable_granville_patterns": False,
            "enable_heikin_ashi_patterns": False,
            "enable_dow_theory_patterns": False,
        },
    },
    {
        "name": "GANN",
        "config": {
            "enable_candlestick_patterns": False,
            "enable_fibonacci_patterns": False,
            "enable_gann_patterns": True,
            "enable_wave_patterns": False,
            "enable_harmonic_patterns": False,
            "enable_oscillator_patterns": False,
            "enable_volume_patterns": False,
            "enable_bollinger_patterns": False,
            "enable_adx_patterns": False,
            "enable_granville_patterns": False,
            "enable_heikin_ashi_patterns": False,
            "enable_dow_theory_patterns": False,
        },
    },
    {
        "name": "WAVE",
        "config": {
            "enable_candlestick_patterns": False,
            "enable_fibonacci_patterns": False,
            "enable_gann_patterns": False,
            "enable_wave_patterns": True,
            "enable_harmonic_patterns": False,
            "enable_oscillator_patterns": False,
            "enable_volume_patterns": False,
            "enable_bollinger_patterns": False,
            "enable_adx_patterns": False,
            "enable_granville_patterns": False,
            "enable_heikin_ashi_patterns": False,
            "enable_dow_theory_patterns": False,
        },
    },
    {
        "name": "HARMONIC",
        "config": {
            "enable_candlestick_patterns": False,
            "enable_fibonacci_patterns": False,
            "enable_gann_patterns": False,
            "enable_wave_patterns": False,
            "enable_harmonic_patterns": True,
            "enable_oscillator_patterns": False,
            "enable_volume_patterns": False,
            "enable_bollinger_patterns": False,
            "enable_adx_patterns": False,
            "enable_granville_patterns": False,
            "enable_heikin_ashi_patterns": False,
            "enable_dow_theory_patterns": False,
        },
    },
    {
        "name": "OSCILLATOR",
        "config": {
            "enable_candlestick_patterns": False,
            "enable_fibonacci_patterns": False,
            "enable_gann_patterns": False,
            "enable_wave_patterns": False,
            "enable_harmonic_patterns": False,
            "enable_oscillator_patterns": True,
            "enable_volume_patterns": False,
            "enable_bollinger_patterns": False,
            "enable_adx_patterns": False,
            "enable_granville_patterns": False,
            "enable_heikin_ashi_patterns": False,
            "enable_dow_theory_patterns": False,
        },
    },
    {
        "name": "VOLUME",
        "config": {
            "enable_candlestick_patterns": False,
            "enable_fibonacci_patterns": False,
            "enable_gann_patterns": False,
            "enable_wave_patterns": False,
            "enable_harmonic_patterns": False,
            "enable_oscillator_patterns": False,
            "enable_volume_patterns": True,
            "enable_bollinger_patterns": False,
            "enable_adx_patterns": False,
            "enable_granville_patterns": False,
            "enable_heikin_ashi_patterns": False,
            "enable_dow_theory_patterns": False,
        },
    },
    {
        "name": "BOLLINGER",
        "config": {
            "enable_candlestick_patterns": False,
            "enable_fibonacci_patterns": False,
            "enable_gann_patterns": False,
            "enable_wave_patterns": False,
            "enable_harmonic_patterns": False,
            "enable_oscillator_patterns": False,
            "enable_volume_patterns": False,
            "enable_bollinger_patterns": True,
            "enable_adx_patterns": False,
            "enable_granville_patterns": False,
            "enable_heikin_ashi_patterns": False,
            "enable_dow_theory_patterns": False,
        },
    },
    {
        "name": "ADX",
        "config": {
            "enable_candlestick_patterns": False,
            "enable_fibonacci_patterns": False,
            "enable_gann_patterns": False,
            "enable_wave_patterns": False,
            "enable_harmonic_patterns": False,
            "enable_oscillator_patterns": False,
            "enable_volume_patterns": False,
            "enable_bollinger_patterns": False,
            "enable_adx_patterns": True,
            "enable_granville_patterns": False,
            "enable_heikin_ashi_patterns": False,
            "enable_dow_theory_patterns": False,
        },
    },
    {
        "name": "GRANVILLE",
        "config": {
            "enable_candlestick_patterns": False,
            "enable_fibonacci_patterns": False,
            "enable_gann_patterns": False,
            "enable_wave_patterns": False,
            "enable_harmonic_patterns": False,
            "enable_oscillator_patterns": False,
            "enable_volume_patterns": False,
            "enable_bollinger_patterns": False,
            "enable_adx_patterns": False,
            "enable_granville_patterns": True,
            "enable_heikin_ashi_patterns": False,
            "enable_dow_theory_patterns": False,
        },
    },
    {
        "name": "HEIKIN_ASHI",
        "config": {
            "enable_candlestick_patterns": False,
            "enable_fibonacci_patterns": False,
            "enable_gann_patterns": False,
            "enable_wave_patterns": False,
            "enable_harmonic_patterns": False,
            "enable_oscillator_patterns": False,
            "enable_volume_patterns": False,
            "enable_bollinger_patterns": False,
            "enable_adx_patterns": False,
            "enable_granville_patterns": False,
            "enable_heikin_ashi_patterns": True,
            "enable_dow_theory_patterns": False,
        },
    },
    {
        "name": "DOW_THEORY",
        "config": {
            "enable_candlestick_patterns": False,
            "enable_fibonacci_patterns": False,
            "enable_gann_patterns": False,
            "enable_wave_patterns": False,
            "enable_harmonic_patterns": False,
            "enable_oscillator_patterns": False,
            "enable_volume_patterns": False,
            "enable_bollinger_patterns": False,
            "enable_adx_patterns": False,
            "enable_granville_patterns": False,
            "enable_heikin_ashi_patterns": False,
            "enable_dow_theory_patterns": True,
        },
    },
]


def run_individual_pattern_backtest(pattern_name: str, config_dict: Dict) -> Dict:
    """Run backtest for individual pattern group."""
    print(f"\n{'='*60}")
    print(f"Testing {pattern_name} patterns...")
    print(f"{'='*60}")

    # Create test data
    data = create_test_data(5000)
    print(f"Created test data with {len(data)} rows")

    # Create config with error suppression
    config = ActionSignalGuideConfig(
        debug_short_mode=False,
        guidance_level=GuidanceLevel.WEAK,
        error_suppression_threshold=0,  # Suppress all error logs
        **config_dict,
    )

    # Initialize ActionSignalGuide
    start_time = time.time()
    guide = ActionSignalGuide(config=config)
    init_time = time.time() - start_time
    print(
        f"Initialized {pattern_name} with {len(guide.all_recognizers)} recognizers in {init_time:.2f}s"
    )

    # Run backtest
    signals_generated = 0
    buy_signals = 0
    sell_signals = 0
    hold_signals = 0

    backtest_start = time.time()
    for i in range(100, len(data)):  # Start from index 100 to have enough history
        try:
            signals = guide.generate_signals(data, i)
            if signals:
                signals_generated += len(signals)
                # Count signal types (simplified)
                for signal in signals:
                    if signal.direction > 0.1:
                        buy_signals += 1
                    elif signal.direction < -0.1:
                        sell_signals += 1
                    else:
                        hold_signals += 1

            # Progress indicator
            if i % 500 == 0:
                elapsed = time.time() - backtest_start
                progress = (i - 100) / (len(data) - 100) * 100
                print(
                    f"Progress: {progress:.1f}% ({i}/{len(data)-1}) - Elapsed: {elapsed:.1f}s"
                )

        except Exception as e:
            print(f"Error at index {i}: {e}")
            continue

    backtest_time = time.time() - backtest_start

    results = {
        "pattern_name": pattern_name,
        "recognizers_count": len(guide.all_recognizers),
        "data_points": len(data) - 100,
        "signals_generated": signals_generated,
        "buy_signals": buy_signals,
        "sell_signals": sell_signals,
        "hold_signals": hold_signals,
        "init_time": init_time,
        "backtest_time": backtest_time,
        "signals_per_second": signals_generated / backtest_time
        if backtest_time > 0
        else 0,
    }

    print(f"\nResults for {pattern_name}:")
    print(f"  Recognizers: {results['recognizers_count']}")
    print(f"  Data points processed: {results['data_points']}")
    print(f"  Signals generated: {results['signals_generated']}")
    print(
        f"  Buy/Sell/Hold: {results['buy_signals']}/{results['sell_signals']}/{results['hold_signals']}"
    )
    print(f"  Backtest time: {results['backtest_time']:.2f}s")
    print(f"  Signals/second: {results['signals_per_second']:.2f}")

    return results


def main() -> None:
    """Run individual pattern validation for all pattern groups."""
    print("Starting Individual Pattern Validation...")

    all_results = []

    for pattern_config in PATTERN_RECOGNIZERS:
        try:
            result = run_individual_pattern_backtest(
                pattern_config["name"], pattern_config["config"]
            )
            all_results.append(result)
        except Exception as e:
            print(f"Failed to test {pattern_config['name']}: {e}")
            import traceback

            traceback.print_exc()
            continue

    # Summary
    print(f"\n{'='*80}")
    print("INDIVIDUAL PATTERN VALIDATION SUMMARY")
    print(f"{'='*80}")

    total_signals = sum(r["signals_generated"] for r in all_results)
    total_time = sum(r["backtest_time"] for r in all_results)

    print(f"Total patterns tested: {len(all_results)}")
    print(f"Total signals generated: {total_signals}")
    print(f"Total backtest time: {total_time:.2f}s")
    print(
        f"Average signals/second: {total_signals/total_time:.2f}"
        if total_time > 0
        else "N/A"
    )

    print("\nDetailed Results:")
    print(
        f"{'Pattern':<15} {'Recognizers':<12} {'Signals':<10} {'Time(s)':<10} {'Sig/s':<8}"
    )
    print("-" * 65)
    for result in all_results:
        print(
            f"{result['pattern_name']:<15} {result['recognizers_count']:<12} {result['signals_generated']:<10} {result['backtest_time']:<10.2f} {result['signals_per_second']:<8.2f}"
        )


if __name__ == "__main__":
    main()
