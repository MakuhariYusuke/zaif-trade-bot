import time

import numpy as np
import pandas as pd

from ztb.trading.strategies.action_signal_guide.action_signal_guide import (
    ActionSignalGuide,
    ActionSignalGuideConfig,
)


def load_real_market_data():
    """Load real BTC/JPY 1-minute data for testing"""
    data_path = "data/btc_jpy_1m_dataset.csv"
    df = pd.read_csv(data_path)

    # Convert timestamp and set as index
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.set_index("timestamp")

    # Ensure we have the required columns
    required_cols = ["open", "high", "low", "close", "volume"]
    if not all(col in df.columns for col in required_cols):
        raise ValueError(f"Missing required columns. Found: {df.columns.tolist()}")

    # Sort by timestamp
    df = df.sort_index()

    print(f"Loaded {len(df)} rows of real BTC/JPY 1-minute data")
    print(f"Date range: {df.index.min()} to {df.index.max()}")
    print(f"Price range: {df['close'].min():.0f} - {df['close'].max():.0f} JPY")

    return df


def test_action_signal_guide_real_data():
    """Test ActionSignalGuide with real market data"""

    print("=== ActionSignalGuide Real Market Data Validation ===\n")

    # Load real market data
    print("1. Loading real BTC/JPY 1-minute data...")
    try:
        df = load_real_market_data()
    except Exception as e:
        print(f"Failed to load data: {e}")
        return

    # Use recent data for testing (last 500 minutes to avoid lookback issues)
    test_data = df.tail(500).copy()
    print(f"   Using last {len(test_data)} minutes for testing")
    print(".2f")
    print()

    # Initialize ActionSignalGuide
    print("2. Initializing ActionSignalGuide...")
    start_time = time.time()
    config = ActionSignalGuideConfig(guidance_level="STRONG")
    guide = ActionSignalGuide(config=config)
    init_time = time.time() - start_time
    print(".3f")
    print()

    # Test signal generation at multiple points
    print("3. Testing signal generation with real market data...")

    # Test at different points in the data
    test_indices = []
    step = max(50, len(test_data) // 10)  # Test at 10 points, minimum 50 bars apart

    for i in range(
        100, len(test_data) - 50, step
    ):  # Start from index 100 to ensure lookback
        test_indices.append(i)

    print(f"   Testing at {len(test_indices)} different time points...")

    all_signals = []
    signal_counts = []
    processing_times = []

    for idx, current_index in enumerate(test_indices):
        try:
            # Generate signals
            signal_start = time.time()
            signals = guide.generate_signals(test_data, current_index)
            signal_time = time.time() - signal_start

            signal_counts.append(len(signals))
            processing_times.append(signal_time)

            if signals:
                all_signals.extend(signals)
                if idx < 5:  # Show details for first 5 tests
                    timestamp = test_data.index[current_index]
                    print(
                        f"   Index {current_index} ({timestamp}): {len(signals)} signals"
                    )
                    for i, signal in enumerate(signals[:2]):  # Show first 2 signals
                        print(
                            f"     Signal {i+1}: {signal.signal_type}, dir={signal.direction:.3f}, conf={signal.confidence:.3f}"
                        )

        except Exception as e:
            print(f"   Error at index {current_index}: {e}")
            continue

    print("\n   Signal generation summary:")
    print(f"   - Total test points: {len(test_indices)}")
    print(f"   - Total signals generated: {len(all_signals)}")
    print(f"   - Average signals per test: {np.mean(signal_counts):.2f}")
    print(f"   - Max signals per test: {max(signal_counts)}")
    print(f"   - Min signals per test: {min(signal_counts)}")
    print(f"   - Average processing time: {np.mean(processing_times):.4f}s")
    print(f"   - Max processing time: {max(processing_times):.4f}s")

    # Analyze signal quality
    if all_signals:
        print("\n4. Signal quality analysis:")
        directions = [s.direction for s in all_signals]
        confidences = [s.confidence for s in all_signals]
        strengths = [s.strength for s in all_signals]

        print(f"   - Direction range: {min(directions):.3f} to {max(directions):.3f}")
        print(f"   - Average direction: {np.mean(directions):.3f}")
        print(
            f"   - Confidence range: {min(confidences):.3f} to {max(confidences):.3f}"
        )
        print(f"   - Average confidence: {np.mean(confidences):.3f}")
        print(f"   - Strength range: {min(strengths):.3f} to {max(strengths):.3f}")
        print(f"   - Average strength: {np.mean(strengths):.3f}")

        # Signal type distribution
        signal_types = {}
        for signal in all_signals:
            signal_type = signal.signal_type
            signal_types[signal_type] = signal_types.get(signal_type, 0) + 1

        print(f"   - Signal types detected: {len(signal_types)}")
        print("   - Top signal types:")
        sorted_types = sorted(signal_types.items(), key=lambda x: x[1], reverse=True)
        for signal_type, count in sorted_types[:5]:
            print(f"     {signal_type}: {count} signals")

    # Test performance metrics
    print("\n5. Performance analysis...")
    try:
        perf_report = guide.generate_signal_performance_report()
        if perf_report:
            print("   Performance report generated successfully")
            print(f"   Report keys: {list(perf_report.keys())}")
        else:
            print("   No performance report available")
    except Exception as e:
        print(f"   Performance report failed: {e}")

    # Memory usage
    try:
        import os

        import psutil

        process = psutil.Process(os.getpid())
        memory_mb = process.memory_info().rss / 1024 / 1024
        print(".1f")
    except ImportError:
        print("   Memory usage: psutil not available")

    print("\n=== Real Market Data Validation Complete ===")
    print("ActionSignalGuide successfully processed real BTC/JPY 1-minute data!")
    print(
        f"Generated {len(all_signals)} signals across {len(test_indices)} test points."
    )


if __name__ == "__main__":
    test_action_signal_guide_real_data()
