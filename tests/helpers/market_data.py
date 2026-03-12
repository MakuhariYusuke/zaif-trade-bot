from __future__ import annotations

import numpy as np
import pandas as pd


def _finalize_ohlcv_frame(
    frame: pd.DataFrame, *, include_timestamp: bool
) -> pd.DataFrame:
    if not include_timestamp:
        return frame
    return frame.reset_index(names="timestamp")


def make_realistic_intraday_ohlcv_data(
    rows: int = 144,
    *,
    seed: int = 42,
    start: str = "2024-01-01 09:00:00",
    freq: str = "5min",
    base_price: float = 50000.0,
    floor_price: float = 10000.0,
    include_timestamp: bool = True,
) -> pd.DataFrame:
    """Build deterministic intraday OHLCV data with trend, mean reversion, and noise."""
    rng = np.random.default_rng(seed)

    close = np.empty(rows, dtype=float)
    close[0] = base_price

    noise = rng.normal(0.0, 0.005, max(rows - 1, 0))
    volatility = rng.choice([0.002, 0.008], size=max(rows - 1, 0), p=[0.7, 0.3])

    for idx in range(1, rows):
        prev_close = close[idx - 1]
        trend = 0.0001 * np.sin((idx - 1) / 50.0)
        mean_reversion = (base_price - prev_close) * 0.001
        change = trend + mean_reversion + noise[idx - 1] * volatility[idx - 1]
        close[idx] = max(prev_close * (1.0 + change), floor_price)

    volatility_factor = rng.uniform(0.002, 0.01, rows)
    high = close * (1.0 + np.abs(rng.normal(0.0, volatility_factor)))
    low = close * (1.0 - np.abs(rng.normal(0.0, volatility_factor)))
    open_ = np.roll(close, 1)
    open_[0] = close[0] * (1.0 + rng.normal(0.0, 0.001))
    volume = rng.lognormal(mean=12.0, sigma=0.8, size=rows)

    frame = pd.DataFrame(
        {
            "open": open_,
            "high": high,
            "low": low,
            "close": close,
            "volume": volume,
        },
        index=pd.date_range(start, periods=rows, freq=freq),
    )
    return _finalize_ohlcv_frame(frame, include_timestamp=include_timestamp)


def make_exchange_random_walk_ohlcv_data(
    rows: int = 1000,
    *,
    seed: int = 42,
    start: str = "2023-01-01",
    freq: str = "1min",
    base_price: float = 100.0,
    return_scale: float = 0.01,
    intrabar_scale: float = 0.005,
    open_scale: float = 0.002,
    volume_logmean: float = 10.0,
    volume_logsigma: float = 1.0,
    include_timestamp: bool = True,
) -> pd.DataFrame:
    """Build multiplicative random-walk OHLCV data with log-normal volume."""
    rng = np.random.default_rng(seed)
    returns = rng.normal(0.0, return_scale, rows)
    close = base_price * np.cumprod(1.0 + returns)
    high = close * (1.0 + np.abs(rng.normal(0.0, intrabar_scale, rows)))
    low = close * (1.0 - np.abs(rng.normal(0.0, intrabar_scale, rows)))
    open_ = close * (1.0 + rng.normal(0.0, open_scale, rows))
    volume = rng.lognormal(mean=volume_logmean, sigma=volume_logsigma, size=rows)

    frame = pd.DataFrame(
        {
            "open": open_,
            "high": high,
            "low": low,
            "close": close,
            "volume": volume,
        },
        index=pd.date_range(start, periods=rows, freq=freq),
    )
    return _finalize_ohlcv_frame(frame, include_timestamp=include_timestamp)


def make_trending_ohlcv_data(
    rows: int = 100,
    *,
    seed: int = 42,
    start: str = "2023-01-01",
    freq: str = "1h",
    start_price: float = 100.0,
    end_price: float = 120.0,
    noise_scale: float = 2.0,
    volume_low: float = 1000.0,
    volume_high: float = 2000.0,
    include_timestamp: bool = False,
) -> pd.DataFrame:
    """Build deterministic trending OHLCV data for tests."""
    rng = np.random.default_rng(seed)
    close = np.linspace(start_price, end_price, rows) + rng.normal(0, noise_scale, rows)
    open_ = close - rng.uniform(1.0, 3.0, rows)
    high = close + rng.uniform(1.0, 3.0, rows)
    low = close - rng.uniform(1.0, 3.0, rows)
    volume = rng.uniform(volume_low, volume_high, rows)
    frame = pd.DataFrame(
        {
            "open": open_,
            "high": high,
            "low": low,
            "close": close,
            "volume": volume,
        },
        index=pd.date_range(start, periods=rows, freq=freq),
    )
    return _finalize_ohlcv_frame(frame, include_timestamp=include_timestamp)


def make_random_walk_ohlcv_data(
    rows: int = 320,
    *,
    seed: int = 0,
    start: str = "2025-01-01",
    freq: str = "min",
    base_price: float = 100.0,
    include_timestamp: bool = False,
) -> pd.DataFrame:
    """Build deterministic random-walk OHLCV data for signal-recognition tests."""
    rng = np.random.default_rng(seed)
    open_ = np.cumsum(rng.normal(0, 1.0, rows)) + base_price
    high = open_ + np.abs(rng.uniform(0.0, 1.5, rows))
    low = open_ - np.abs(rng.uniform(0.0, 1.5, rows))
    close = open_ + rng.normal(0.0, 0.5, rows)
    volume = rng.integers(100, 10000, rows)
    frame = pd.DataFrame(
        {
            "open": open_,
            "high": high,
            "low": low,
            "close": close,
            "volume": volume,
        },
        index=pd.date_range(start, periods=rows, freq=freq),
    )
    return _finalize_ohlcv_frame(frame, include_timestamp=include_timestamp)


def make_multi_regime_ohlcv_data(
    rows_per_regime: int = 192,
    *,
    seed: int = 42,
    start: str = "2023-01-01",
    freq: str = "1min",
    base_price: float = 100.0,
    include_timestamp: bool = True,
) -> pd.DataFrame:
    """Build deterministic OHLCV data with four distinct market regimes."""
    rng = np.random.default_rng(seed)

    bull = base_price + np.linspace(0, 12, rows_per_regime) + rng.normal(0, 0.6, rows_per_regime)
    range_high = (
        base_price + 12 + rng.normal(0, 2.2, rows_per_regime)
    )
    bear = (
        base_price
        + 12
        + np.linspace(0, -10, rows_per_regime)
        + rng.normal(0, 0.8, rows_per_regime)
    )
    range_low = base_price + 2 + rng.normal(0, 0.25, rows_per_regime)

    close = np.concatenate([bull, range_high, bear, range_low])
    total_rows = len(close)
    high = close + np.abs(rng.normal(0, 0.45, total_rows))
    low = close - np.abs(rng.normal(0, 0.45, total_rows))
    open_ = np.roll(close, 1)
    open_[0] = base_price
    volume = rng.uniform(100.0, 2000.0, total_rows)

    frame = pd.DataFrame(
        {
            "open": open_,
            "high": high,
            "low": low,
            "close": close,
            "volume": volume,
        }
    )
    frame.index = pd.date_range(start, periods=total_rows, freq=freq)
    return _finalize_ohlcv_frame(frame, include_timestamp=include_timestamp)
