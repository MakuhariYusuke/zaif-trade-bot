#!/usr/bin/env python3
"""
Unit tests for HierarchicalTrendAnalyzer.
"""

import numpy as np
import pandas as pd

from ztb.trading.strategies.action_signal_guide.pattern_recognition.trend_analyzer import (
    HierarchicalTrendAnalyzer,
)


def _build_trending_data(length: int = 180) -> pd.DataFrame:
    rng = np.random.default_rng(42)
    trend = np.linspace(100.0, 130.0, length)
    noise = rng.normal(0.0, 0.8, length)
    close = np.maximum(trend + noise, 1.0)
    open_price = np.roll(close, 1)
    open_price[0] = close[0]
    high = np.maximum(open_price, close) * (1.0 + rng.uniform(0.001, 0.01, length))
    low = np.minimum(open_price, close) * (1.0 - rng.uniform(0.001, 0.01, length))
    volume = rng.uniform(1000.0, 10000.0, length)
    return pd.DataFrame(
        {
            "open": open_price,
            "high": high,
            "low": low,
            "close": close,
            "volume": volume,
        }
    )


def test_recognize_uses_latest_with_default_index() -> None:
    recognizer = HierarchicalTrendAnalyzer({})
    data = _build_trending_data()

    result = recognizer.recognize(data)

    assert result is not None
    assert result.signal_type in {
        "trend_bullish",
        "trend_bearish",
        "trend_sideways",
        "trend_neutral",
    }
    assert "trend_strength" in result.metadata


def test_recognize_returns_none_on_out_of_range_index() -> None:
    recognizer = HierarchicalTrendAnalyzer({})
    data = _build_trending_data()

    result = recognizer.recognize(data, index=len(data) + 10)

    assert result is None


def test_adx_strength_is_computable_for_trend() -> None:
    recognizer = HierarchicalTrendAnalyzer({"enable_wave_analysis": False})
    data = _build_trending_data()

    result = recognizer.recognize(data, index=len(data) - 1)

    assert result is not None
    trend_strength = float(result.metadata.get("trend_strength", 0.0))
    assert trend_strength > 0.0

