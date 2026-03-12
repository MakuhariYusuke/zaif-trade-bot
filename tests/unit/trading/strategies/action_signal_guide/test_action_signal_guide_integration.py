"""
Integration-style tests for ActionSignalGuide. These include data generation and heavy pattern recognizer checks.
We merged comprehensive and structured tests into this file.
"""

from datetime import datetime, timedelta

import numpy as np
import pandas as pd

from ztb.trading.strategies.action_signal_guide.action_signal_guide import (
    ActionSignalGuide,
    GuidanceLevel,
)

# Selected recognizers for integration checking
from ztb.trading.strategies.action_signal_guide.pattern_recognition.candlestick_patterns import (
    HammerRecognizer,
)
from ztb.trading.strategies.action_signal_guide.pattern_recognition.fibonacci_patterns import (
    FibonacciRetracementRecognizer,
)


def create_sample_data(num_bars: int = 100) -> pd.DataFrame:
    np.random.seed(42)
    base_price = 100.0
    prices = [0.0] * num_bars
    for i in range(num_bars):
        trend = 0.001 * i
        noise = np.random.normal(0, 0.02)
        price = base_price * (1 + trend + noise)
        prices[i] = price

    timestamps = [
        datetime.now() - timedelta(hours=num_bars - i) for i in range(num_bars)
    ]
    opens = [0.0] * num_bars
    highs = [0.0] * num_bars
    lows = [0.0] * num_bars
    closes = [0.0] * num_bars
    volumes = [0] * num_bars

    opens[0] = prices[0] * (1 + np.random.normal(0, 0.005))

    for i, close in enumerate(prices):
        high = close * (1 + abs(np.random.normal(0, 0.01)))
        low = close * (1 - abs(np.random.normal(0, 0.01)))
        if i < num_bars - 1:
            opens[i + 1] = (
                opens[i] if i == 0 else close * (1 + np.random.normal(0, 0.005))
            )
        volume = np.random.randint(1000, 10000)
        highs[i] = max(opens[i], high)
        lows[i] = min(opens[i], low)
        closes[i] = close
        volumes[i] = volume

    df = pd.DataFrame(
        {
            "timestamp": timestamps,
            "open": opens,
            "high": highs,
            "low": lows,
            "close": closes,
            "volume": volumes,
        }
    )
    df.set_index("timestamp", inplace=True)
    return df


def test_basic_functionality():
    data = create_sample_data(200)
    guide = ActionSignalGuide(guidance_level=GuidanceLevel.STRONG)

    total_signals = []
    for i in range(50, len(data)):
        signals = guide.generate_signals(data, i)
        total_signals.extend(signals)

    if total_signals:
        recent_signals = total_signals[-10:]
        consolidated = guide.get_consolidated_signal(recent_signals)
        assert consolidated is None or hasattr(consolidated, "description")


def test_pattern_recognizers():
    data = create_sample_data(100)
    hammer_recognizer = HammerRecognizer()
    fib_recognizer = FibonacciRetracementRecognizer()

    hammer_signals = []
    fib_signals = []

    for i in range(20, len(data)):
        hammer_signal = hammer_recognizer.recognize(data, i)
        if hammer_signal:
            hammer_signals.append(hammer_signal)

        fib_signal = fib_recognizer.recognize(data, i)
        if fib_signal:
            fib_signals.append(fib_signal)

    # The purpose is to run the recognizers without throwing errors and gather signals
    assert isinstance(hammer_signals, list)
    assert isinstance(fib_signals, list)
