import time

import numpy as np
import pandas as pd

from ztb.trading.strategies.action_signal_guide.action_signal_guide import (
    ActionSignalGuide,
    ActionSignalGuideConfig,
)


def create_sample_data():
    """Create longer test data with clear bullish engulfing pattern"""
    # Create longer data series with clear bullish engulfing pattern
    data = []

    # Initial downtrend (20+ candles to satisfy lookback requirements)
    base_price = 100.0
    for i in range(25):  # 25 candles of downtrend
        trend = -0.001 * (25 - i)  # Gradual downtrend
        noise = np.random.normal(0, 0.002)
        close_price = base_price * (1 + trend + noise)
        high_price = close_price * (1 + abs(np.random.normal(0, 0.005)))
        low_price = close_price * (1 - abs(np.random.normal(0, 0.005)))
        open_price = base_price if i == 0 else data[-1]["close"]

        data.append(
            {
                "timestamp": pd.Timestamp("2024-01-01") + pd.Timedelta(hours=i),
                "open": open_price,
                "high": high_price,
                "low": low_price,
                "close": close_price,
                "volume": 1000 + np.random.randint(0, 1000),
            }
        )
        base_price = close_price

    # Bullish engulfing pattern at the end
    # Previous candle: small red
    prev_close = data[-1]["close"]
    data.append(
        {
            "timestamp": data[-1]["timestamp"] + pd.Timedelta(hours=1),
            "open": prev_close,
            "high": prev_close * 1.005,
            "low": prev_close * 0.995,
            "close": prev_close * 0.998,  # Small red candle
            "volume": 1200,
        }
    )

    # Engulfing candle: large green that engulfs previous
    prev_open = data[-1]["open"]
    prev_high = data[-1]["high"]
    prev_low = data[-1]["low"]
    engulfing_open = prev_close * 0.997  # Opens below previous low
    engulfing_close = prev_open * 1.02  # Closes above previous high
    data.append(
        {
            "timestamp": data[-1]["timestamp"] + pd.Timedelta(hours=1),
            "open": engulfing_open,
            "high": engulfing_close * 1.01,
            "low": engulfing_open * 0.99,
            "close": engulfing_close,  # Large green engulfing candle
            "volume": 2500,  # Higher volume
        }
    )

    # Confirmation candle
    data.append(
        {
            "timestamp": data[-1]["timestamp"] + pd.Timedelta(hours=1),
            "open": engulfing_close,
            "high": engulfing_close * 1.015,
            "low": engulfing_close * 0.995,
            "close": engulfing_close * 1.01,  # Continuation green
            "volume": 1800,
        }
    )

    df = pd.DataFrame(data)
    return df


def test_action_signal_guide():
    """Comprehensive test of ActionSignalGuide functionality"""
    print("=== ActionSignalGuide Comprehensive Validation ===\n")

    # Create sample data
    print("1. Creating sample market data...")
    df = create_sample_data()
    print(f"   Generated {len(df)} data points")
    print(".2f")
    print(".0f")
    print()

    # Initialize ActionSignalGuide
    print("2. Initializing ActionSignalGuide...")
    start_time = time.time()
    config = ActionSignalGuideConfig()
    guide = ActionSignalGuide(config=config)
    init_time = time.time() - start_time
    print(".3f")
    print()

    # Test basic functionality
    print("3. Testing basic functionality...")

    # Get guidance stats
    stats = guide.get_guidance_stats()
    print(f"   Guidance stats: {list(stats.keys())}")
    print(f"   Mode: {stats['mode']}")
    print(f"   Available signals: {stats['available_signals']}")

    # Get performance stats
    try:
        perf_stats = guide.generate_signal_performance_report()
        print(f"   Performance stats: {list(perf_stats.keys())}")
        print(f"   Cache size: {perf_stats.get('cache_size', 'N/A')}")
    except Exception:
        perf_stats = {"cache_size": "N/A", "memory": {"current_mb": "N/A"}}
        print("   Performance stats: generate_signal_performance_report not available")
        print("   Cache size: N/A")
    print()

    # Test signal generation (if possible)
    print("4. Testing signal generation capabilities...")
    try:
        # Try to generate signals with sample data - use a reasonable current_index
        current_index = 26  # Test at the confirmation candle after engulfing pattern
        print(f"   Attempting signal generation at index {current_index}...")
        print(f"   Data shape: {df.shape}")
        print(f"   Data columns: {list(df.columns)}")
        print(f"   Sample data at index {current_index}:")
        print(f"     Close: {df.iloc[current_index]['close']:.4f}")
        print(f"     Volume: {df.iloc[current_index]['volume']:.0f}")

        # Show context data around the engulfing pattern (last few candles)
        start_idx = max(0, current_index - 5)
        context_data = df.iloc[start_idx:]
        print("   Recent context data (engulfing pattern):")
        for idx, row in context_data.iterrows():
            direction = "GREEN" if row["close"] > row["open"] else "RED"
            engulfing_marker = " <-- ENGULFING" if idx == 25 else ""
            print(
                f"     {idx}: O={row['open']:.2f}, H={row['high']:.2f}, L={row['low']:.2f}, C={row['close']:.2f} ({direction}){engulfing_marker}"
            )

        signals = guide.generate_signals(df, current_index)
        print(f"   Generated {len(signals)} signals at index {current_index}")
        if signals:
            print(
                f"   Sample signal: direction={signals[0].direction:.3f}, confidence={signals[0].confidence:.3f}"
            )
        else:
            print("   No signals generated - checking recognizer status...")
            # Check if recognizers are initialized
            if hasattr(guide, "signal_generator") and guide.signal_generator:
                print(
                    f"   Signal generator has {len(guide.signal_generator.all_recognizers)} recognizers"
                )
                # Check sample recognizer
                if guide.signal_generator.all_recognizers:
                    sample_recognizer = guide.signal_generator.all_recognizers[0]
                    print(f"   Sample recognizer: {type(sample_recognizer).__name__}")
                    print(
                        f"   Has recognize method: {hasattr(sample_recognizer, 'recognize')}"
                    )
            else:
                print("   Signal generator not properly initialized")
    except Exception as e:
        print(f"   Signal generation test failed: {e}")
        import traceback

        traceback.print_exc()
    print()

    # Test configuration updates
    print("5. Testing configuration updates...")
    try:
        guide.update_config({"max_signals_per_bar": 10})
        print("   Configuration update successful")
    except Exception as e:
        print(f"   Configuration update failed: {e}")
    print()

    # Test memory management
    print("6. Testing memory management...")
    try:
        guide._cleanup_memory()
        print("   Memory cleanup successful")
    except Exception as e:
        print(f"   Memory cleanup failed: {e}")
    print()

    # Test validation report
    print("7. Testing validation report generation...")
    try:
        report = guide.generate_validation_report()
        print(f"   Validation report generated ({len(report)} characters)")
        # Show first few lines
        lines = report.split("\n")[:10]
        print("   Report preview:")
        for line in lines:
            print(f"     {line}")
    except Exception as e:
        print(f"   Validation report failed: {e}")
    print()

    # Performance analysis
    print("8. Performance analysis...")
    print(".3f")
    print(
        f"   Memory usage: {perf_stats.get('memory', {}).get('current_mb', 'N/A')} MB"
    )
    print()

    # Identify potential improvements
    print("9. Identifying potential improvements...")

    improvements = []

    # Check for missing features
    if not hasattr(guide, "generate_signals"):
        improvements.append("Signal generation method missing")

    # Check memory efficiency
    if perf_stats.get("cache_size", 0) > 10000:
        improvements.append("Cache size may be too large for memory efficiency")

    # Check signal count
    if len(guide.signal_history) == 0:
        improvements.append("No signals generated - may need better initialization")

    # Check error handling
    try:
        guide.update_config({"invalid_param": "test"})
        improvements.append("Configuration validation may be too permissive")
    except Exception:
        pass  # Good, validation works

    if improvements:
        print("   Potential improvements identified:")
        for imp in improvements:
            print(f"     - {imp}")
    else:
        print("   No major improvements identified")

    print("\n=== Validation Complete ===")


if __name__ == "__main__":
    test_action_signal_guide()
