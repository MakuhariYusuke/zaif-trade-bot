#!/usr/bin/env python3
"""
Unit tests for pattern-recognition MultiTimeframeAnalyzer helpers.
"""

import numpy as np
import pandas as pd

from ztb.trading.strategies.action_signal_guide.pattern_recognition.base import (
    PatternRecognizer,
)
from ztb.trading.strategies.action_signal_guide.pattern_recognition.multi_timeframe_analyzer import (
    MultiTimeframeAnalyzer,
)


def _build_ohlc(length: int, start: float = 100.0, step: float = 0.1) -> pd.DataFrame:
    close = np.array([start + step * i for i in range(length)], dtype=np.float64)
    open_price = np.roll(close, 1)
    open_price[0] = close[0]
    high = np.maximum(open_price, close) * 1.01
    low = np.minimum(open_price, close) * 0.99
    volume = np.full(length, 1000.0, dtype=np.float64)
    return pd.DataFrame(
        {
            "open": open_price,
            "high": high,
            "low": low,
            "close": close,
            "volume": volume,
        }
    )


def test_iter_multi_timeframe_frames_filters_invalid_payloads() -> None:
    valid = _build_ohlc(30)
    short = _build_ohlc(2)
    payload = {
        "15m": {"data": valid},
        "1m": {"data": short},
        "bad": {"data": "not_dataframe"},
        "missing": {},
    }

    frames = PatternRecognizer.iter_multi_timeframe_frames(payload, min_length=10)

    assert len(frames) == 1
    assert frames[0].equals(valid)


def test_consensus_tie_returns_neutral_strength() -> None:
    analyzer = MultiTimeframeAnalyzer()
    analysis_results = {
        "1m": {"trend_direction": 1, "trend_strength": 0.8},
        "5m": {"trend_direction": -1, "trend_strength": 0.8},
    }

    score = analyzer._calculate_level_consensus(analysis_results, ["1m", "5m"], "trend")

    assert score == 0.0


def test_alignment_handles_missing_primary_timeframe_data() -> None:
    analyzer = MultiTimeframeAnalyzer()
    data = {
        "1m": {"data": _build_ohlc(40)},
        "15m": {},  # primary timeframe exists but has no dataframe payload
    }

    analysis = analyzer.analyze_multi_timeframe_alignment(data, primary_timeframe="15m")

    assert "1m" in analysis
    assert analysis["1m"]["alignment_score"] == 1.0
    assert analysis["1m"]["is_aligned"] is True


def test_trend_strength_and_consolidation_guard_zero_prices() -> None:
    analyzer = MultiTimeframeAnalyzer()
    zero_df = pd.DataFrame(
        {
            "open": np.zeros(20, dtype=np.float64),
            "high": np.zeros(20, dtype=np.float64),
            "low": np.zeros(20, dtype=np.float64),
            "close": np.zeros(20, dtype=np.float64),
            "volume": np.ones(20, dtype=np.float64),
        }
    )

    assert analyzer._calculate_trend_strength(zero_df) == 0.0
    assert analyzer._is_consolidation(np.zeros(10, dtype=np.float64)) is False

