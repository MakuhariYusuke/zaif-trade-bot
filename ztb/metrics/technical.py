"""
Technical Analysis Metrics.

This module provides common technical analysis indicators.
It uses existing feature generators from ztb.features.generators.technical where possible.
"""

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from ztb.features.generators.technical.momentum.rsi import compute_rsi
from ztb.features.generators.technical.volatility.return_std import (
    compute_return_stddev,
)
from ztb.trading.constants import TRADING_DAYS_PER_YEAR
from ztb.utils.talib_wrapper import TaLibWrapper

# Global instance to share cache for SMA
_talib = TaLibWrapper()

def calculate_rsi(
    prices: list[float] | NDArray[np.float64] | pd.Series, period: int = 14
) -> float:
    """
    Calculate RSI (Relative Strength Index).
    Returns the latest RSI value.

    Args:
        prices: Price data (list, numpy array, or pandas Series)
        period: RSI period (default: 14)

    Returns:
        Latest RSI value (0-100). Returns 50.0 if insufficient data.
    """
    # Convert to DataFrame for generator
    if isinstance(prices, list):
        df = pd.DataFrame({"close": prices})
    elif isinstance(prices, np.ndarray):
        df = pd.DataFrame({"close": prices})
    elif isinstance(prices, pd.Series):
        df = pd.DataFrame({"close": prices.values})
    else:
        return 50.0

    if len(df) < period + 1:
        return 50.0

    try:
        # Use feature generator
        rsi_series = compute_rsi(df, period=period)

        # Get last valid value
        last_val = rsi_series.iloc[-1]
        return float(last_val) if not pd.isna(last_val) else 50.0
    except Exception:
        return 50.0

def calculate_volatility(
    prices: list[float] | NDArray[np.float64] | pd.Series,
    window: int = 20,
    annualize: bool = False,
    trading_days: int = TRADING_DAYS_PER_YEAR,
) -> float:
    """
    Calculate volatility (standard deviation of returns).
    Returns the latest volatility value.

    Args:
        prices: Price data
        window: Window size
        annualize: Whether to annualize the volatility (default: False)
        trading_days: Number of trading days for annualization (default: TRADING_DAYS_PER_YEAR)

    Returns:
        Latest volatility value. Returns 0.0 if insufficient data.
    """
    # Convert to DataFrame for generator
    if isinstance(prices, list):
        df = pd.DataFrame({"close": prices})
    elif isinstance(prices, np.ndarray):
        df = pd.DataFrame({"close": prices})
    elif isinstance(prices, pd.Series):
        df = pd.DataFrame({"close": prices.values})
    else:
        return 0.0

    if len(df) < window:
        return 0.0

    try:
        # Use feature generator
        vol_series = compute_return_stddev(df, period=window)

        last_val = vol_series.iloc[-1]
        val = float(last_val) if not pd.isna(last_val) else 0.0

        if annualize:
            val = val * np.sqrt(trading_days)

        return val
    except Exception:
        return 0.0

def calculate_volatility_from_returns(
    returns: list[float] | NDArray[np.float64] | pd.Series,
    window: int = 20,
    annualize: bool = False,
    trading_days: int = TRADING_DAYS_PER_YEAR,
) -> float:
    """
    Calculate volatility from returns series.
    Returns the latest volatility value.

    Args:
        returns: Returns data
        window: Window size
        annualize: Whether to annualize the volatility (default: False)
        trading_days: Number of trading days for annualization (default: TRADING_DAYS_PER_YEAR)

    Returns:
        Latest volatility value. Returns 0.0 if insufficient data.
    """
    if isinstance(returns, list):
        returns_series = pd.Series(returns)
    elif isinstance(returns, np.ndarray):
        returns_series = pd.Series(returns)
    elif isinstance(returns, pd.Series):
        returns_series = returns
    else:
        return 0.0

    if len(returns_series) < window:
        return 0.0

    try:
        vol = returns_series.rolling(window=window).std()
        last_val = vol.iloc[-1]
        val = float(last_val) if not pd.isna(last_val) else 0.0

        if annualize:
            val = val * np.sqrt(trading_days)

        return val
    except Exception:
        return 0.0

def calculate_rolling_volatility(
    returns: pd.Series | NDArray[np.float64],
    window: int = 20,
    annualize: bool = False,
    trading_days: int = TRADING_DAYS_PER_YEAR,
) -> pd.Series:
    """
    Calculate rolling volatility from returns series.
    Returns the full series of volatility values.

    Args:
        returns: Returns data (pandas Series or numpy array)
        window: Window size
        annualize: Whether to annualize the volatility (default: False)
        trading_days: Number of trading days for annualization (default: TRADING_DAYS_PER_YEAR)

    Returns:
        Series of volatility values.
    """
    if isinstance(returns, np.ndarray):
        returns_series = pd.Series(returns)
    elif isinstance(returns, pd.Series):
        returns_series = returns
    else:
        return pd.Series(dtype=float)

    try:
        vol = returns_series.rolling(window=window).std()

        if annualize:
            vol = vol * np.sqrt(trading_days)

        return vol
    except Exception:
        return pd.Series(dtype=float)

def calculate_rolling_sma(
    prices: list[float] | NDArray[np.float64] | pd.Series, period: int
) -> pd.Series:
    """
    Calculate rolling Simple Moving Average (SMA).
    Returns the full series of SMA values.

    Args:
        prices: Price data
        period: SMA period

    Returns:
        Series of SMA values.
    """
    if isinstance(prices, list):
        prices_arr = np.array(prices, dtype=np.float64)
    elif isinstance(prices, pd.Series):
        prices_arr = prices.values  # type: ignore
    else:
        prices_arr = prices

    if len(prices_arr) == 0:
        return pd.Series(dtype=float)

    try:
        sma_values = _talib.sma(prices_arr, period)
        return pd.Series(
            sma_values, index=prices.index if isinstance(prices, pd.Series) else None
        )
    except Exception:
        # Fallback
        return pd.Series(prices_arr).rolling(window=period).mean()

def calculate_rolling_rsi(
    prices: list[float] | NDArray[np.float64] | pd.Series, period: int = 14
) -> pd.Series:
    """
    Calculate rolling RSI (Relative Strength Index).
    Returns the full series of RSI values.

    Args:
        prices: Price data
        period: RSI period (default: 14)

    Returns:
        Series of RSI values.
    """
    # Convert to DataFrame for generator
    if isinstance(prices, list):
        df = pd.DataFrame({"close": prices})
    elif isinstance(prices, np.ndarray):
        df = pd.DataFrame({"close": prices})
    elif isinstance(prices, pd.Series):
        df = pd.DataFrame({"close": prices.values}, index=prices.index)
    else:
        return pd.Series(dtype=float)

    if len(df) < period + 1:
        return pd.Series(dtype=float)

    try:
        # Use feature generator
        rsi_series = compute_rsi(df, period=period)
        return rsi_series
    except Exception:
        return pd.Series(dtype=float)

def calculate_sma(
    prices: list[float] | NDArray[np.float64] | pd.Series, period: int
) -> float:
    """
    Calculate Simple Moving Average (SMA).
    Returns the latest SMA value.

    Args:
        prices: Price data
        period: SMA period

    Returns:
        Latest SMA value. Returns 0.0 if insufficient data.
    """
    if isinstance(prices, list):
        prices_arr = np.array(prices, dtype=np.float64)
    elif isinstance(prices, pd.Series):
        prices_arr = prices.values  # type: ignore
    else:
        prices_arr = prices

    if len(prices_arr) < period:
        return float(prices_arr[-1]) if len(prices_arr) > 0 else 0.0

    try:
        sma_values = _talib.sma(prices_arr, period)
        valid_sma = sma_values[~np.isnan(sma_values)]
        return float(valid_sma[-1]) if len(valid_sma) > 0 else 0.0
    except Exception:
        # Fallback
        return float(np.mean(prices_arr[-period:]))

def calculate_macd(
    prices: list[float] | NDArray[np.float64] | pd.Series,
    fast_period: int = 12,
    slow_period: int = 26,
    signal_period: int = 9,
) -> tuple[pd.Series, pd.Series, pd.Series]:
    """
    Calculate MACD (Moving Average Convergence Divergence).
    Returns (macd, signal, histogram) series.

    Args:
        prices: Price data
        fast_period: Fast EMA period
        slow_period: Slow EMA period
        signal_period: Signal EMA period

    Returns:
        tuple of (macd, signal, histogram) series.
    """
    if isinstance(prices, list):
        prices_arr = np.array(prices, dtype=np.float64)
    elif isinstance(prices, pd.Series):
        prices_arr = prices.values  # type: ignore
    else:
        prices_arr = prices

    if len(prices_arr) < slow_period:
        empty = pd.Series(dtype=float)
        return empty, empty, empty

    try:
        macd, signal, hist = _talib.macd(
            prices_arr, fast_period, slow_period, signal_period
        )

        index = prices.index if isinstance(prices, pd.Series) else None
        return (
            pd.Series(macd, index=index),
            pd.Series(signal, index=index),
            pd.Series(hist, index=index),
        )
    except Exception:
        # Fallback using pandas
        s = pd.Series(prices_arr)
        ema_fast = s.ewm(span=fast_period, adjust=False).mean()
        ema_slow = s.ewm(span=slow_period, adjust=False).mean()
        macd = ema_fast - ema_slow
        signal = macd.ewm(span=signal_period, adjust=False).mean()
        hist = macd - signal

        if isinstance(prices, pd.Series):
            macd.index = prices.index
            signal.index = prices.index
            hist.index = prices.index

        return macd, signal, hist

def calculate_bollinger_bands(
    prices: list[float] | NDArray[np.float64] | pd.Series,
    period: int = 20,
    std_dev: float = 2.0,
) -> tuple[pd.Series, pd.Series, pd.Series]:
    """
    Calculate Bollinger Bands.
    Returns (upper, middle, lower) series.

    Args:
        prices: Price data
        period: Moving average period
        std_dev: Standard deviation multiplier

    Returns:
        tuple of (upper, middle, lower) series.
    """
    if isinstance(prices, list):
        prices_arr = np.array(prices, dtype=np.float64)
    elif isinstance(prices, pd.Series):
        prices_arr = prices.values  # type: ignore
    else:
        prices_arr = prices

    if len(prices_arr) < period:
        empty = pd.Series(dtype=float)
        return empty, empty, empty

    try:
        upper, middle, lower = _talib.bbands(prices_arr, period, std_dev, std_dev)

        index = prices.index if isinstance(prices, pd.Series) else None
        return (
            pd.Series(upper, index=index),
            pd.Series(middle, index=index),
            pd.Series(lower, index=index),
        )
    except Exception:
        # Fallback using pandas
        s = pd.Series(prices_arr)
        middle = s.rolling(window=period).mean()
        std = s.rolling(window=period).std()
        upper = middle + (std * std_dev)
        lower = middle - (std * std_dev)

        if isinstance(prices, pd.Series):
            upper.index = prices.index
            middle.index = prices.index
            lower.index = prices.index

        return upper, middle, lower

def calculate_atr(
    high: list[float] | NDArray[np.float64] | pd.Series,
    low: list[float] | NDArray[np.float64] | pd.Series,
    close: list[float] | NDArray[np.float64] | pd.Series,
    period: int = 14,
) -> float:
    """
    Calculate ATR (Average True Range).
    Returns the latest ATR value.

    Args:
        high: High prices
        low: Low prices
        close: Close prices
        period: ATR period (default: 14)

    Returns:
        Latest ATR value. Returns 0.0 if insufficient data.
    """
    # Convert to DataFrame for generator
    try:
        df = pd.DataFrame(
            {
                "high": high
                if isinstance(high, (pd.Series, np.ndarray))
                else np.array(high),
                "low": low
                if isinstance(low, (pd.Series, np.ndarray))
                else np.array(low),
                "close": close
                if isinstance(close, (pd.Series, np.ndarray))
                else np.array(close),
            }
        )

        if len(df) < period + 1:
            return 0.0

        # Use feature generator
        from ztb.features.generators.technical.volatility.atr import compute_atr

        atr_series = compute_atr(df, period=period)

        last_val = atr_series.iloc[-1]
        return float(last_val) if not pd.isna(last_val) else 0.0
    except Exception:
        return 0.0

def calculate_rolling_atr(
    high: list[float] | NDArray[np.float64] | pd.Series,
    low: list[float] | NDArray[np.float64] | pd.Series,
    close: list[float] | NDArray[np.float64] | pd.Series,
    period: int = 14,
) -> pd.Series:
    """
    Calculate rolling ATR (Average True Range).
    Returns the full series of ATR values.

    Args:
        high: High prices
        low: Low prices
        close: Close prices
        period: ATR period (default: 14)

    Returns:
        Series of ATR values.
    """
    try:
        df = pd.DataFrame(
            {
                "high": high
                if isinstance(high, (pd.Series, np.ndarray))
                else np.array(high),
                "low": low
                if isinstance(low, (pd.Series, np.ndarray))
                else np.array(low),
                "close": close
                if isinstance(close, (pd.Series, np.ndarray))
                else np.array(close),
            }
        )

        if isinstance(close, pd.Series):
            df.index = close.index

        if len(df) < period + 1:
            return pd.Series(dtype=float)

        # Use feature generator
        from ztb.features.generators.technical.volatility.atr import compute_atr

        return compute_atr(df, period=period)
    except Exception:
        return pd.Series(dtype=float)

def calculate_adx(
    high: list[float] | NDArray[np.float64] | pd.Series,
    low: list[float] | NDArray[np.float64] | pd.Series,
    close: list[float] | NDArray[np.float64] | pd.Series,
    period: int = 14,
) -> float:
    """
    Calculate ADX (Average Directional Index).
    Returns the latest ADX value.

    Args:
        high: High prices
        low: Low prices
        close: Close prices
        period: ADX period (default: 14)

    Returns:
        Latest ADX value. Returns 0.0 if insufficient data.
    """
    try:
        df = pd.DataFrame(
            {
                "high": high
                if isinstance(high, (pd.Series, np.ndarray))
                else np.array(high),
                "low": low
                if isinstance(low, (pd.Series, np.ndarray))
                else np.array(low),
                "close": close
                if isinstance(close, (pd.Series, np.ndarray))
                else np.array(close),
            }
        )

        if len(df) < period * 2:  # ADX needs more data to stabilize
            return 0.0

        # Use feature generator
        from ztb.features.generators.technical.trend.adx import compute_adx

        adx_series = compute_adx(df, period=period)

        last_val = adx_series.iloc[-1]
        return float(last_val) if not pd.isna(last_val) else 0.0
    except Exception:
        return 0.0

def calculate_rolling_adx(
    high: list[float] | NDArray[np.float64] | pd.Series,
    low: list[float] | NDArray[np.float64] | pd.Series,
    close: list[float] | NDArray[np.float64] | pd.Series,
    period: int = 14,
) -> pd.Series:
    """
    Calculate rolling ADX (Average Directional Index).
    Returns the full series of ADX values.

    Args:
        high: High prices
        low: Low prices
        close: Close prices
        period: ADX period (default: 14)

    Returns:
        Series of ADX values.
    """
    try:
        df = pd.DataFrame(
            {
                "high": high
                if isinstance(high, (pd.Series, np.ndarray))
                else np.array(high),
                "low": low
                if isinstance(low, (pd.Series, np.ndarray))
                else np.array(low),
                "close": close
                if isinstance(close, (pd.Series, np.ndarray))
                else np.array(close),
            }
        )

        if isinstance(close, pd.Series):
            df.index = close.index

        if len(df) < period * 2:
            return pd.Series(dtype=float)

        # Use feature generator
        from ztb.features.generators.technical.trend.adx import compute_adx

        return compute_adx(df, period=period)
    except Exception:
        return pd.Series(dtype=float)

def calculate_stochastic(
    high: list[float] | NDArray[np.float64] | pd.Series,
    low: list[float] | NDArray[np.float64] | pd.Series,
    close: list[float] | NDArray[np.float64] | pd.Series,
    fastk_period: int = 14,
    slowk_period: int = 3,
    slowd_period: int = 3,
) -> tuple[pd.Series, pd.Series]:
    """
    Calculate Stochastic Oscillator.
    Returns (slowk, slowd) series.

    Args:
        high: High prices
        low: Low prices
        close: Close prices
        fastk_period: Time period for building the Fast-K line
        slowk_period: Smoothing for making the Slow-K line
        slowd_period: Smoothing for making the Slow-D line

    Returns:
        tuple of (slowk, slowd) series.
    """
    try:
        # Convert inputs to numpy arrays
        high_arr = (
            np.array(high, dtype=np.float64)
            if isinstance(high, (list, pd.Series))
            else high
        )
        low_arr = (
            np.array(low, dtype=np.float64)
            if isinstance(low, (list, pd.Series))
            else low
        )
        close_arr = (
            np.array(close, dtype=np.float64)
            if isinstance(close, (list, pd.Series))
            else close
        )

        if len(close_arr) < fastk_period:
            empty = pd.Series(dtype=float)
            return empty, empty

        slowk, slowd = _talib.stoch(
            high_arr,
            low_arr,
            close_arr,
            fastk_period=fastk_period,
            slowk_period=slowk_period,
            slowd_period=slowd_period,
        )

        index = close.index if isinstance(close, pd.Series) else None
        return (
            pd.Series(slowk, index=index),
            pd.Series(slowd, index=index),
        )
    except Exception:
        # Fallback using pandas
        # %K = (Close - Lowest Low) / (Highest High - Lowest Low) * 100
        # Slow %K = SMA(%K)
        # Slow %D = SMA(Slow %K)

        s_close = pd.Series(close) if not isinstance(close, pd.Series) else close
        s_high = pd.Series(high) if not isinstance(high, pd.Series) else high
        s_low = pd.Series(low) if not isinstance(low, pd.Series) else low

        lowest_low = s_low.rolling(window=fastk_period).min()
        highest_high = s_high.rolling(window=fastk_period).max()

        # Avoid division by zero
        denom = highest_high - lowest_low
        denom = denom.replace(0, np.nan)

        fastk = ((s_close - lowest_low) / denom) * 100
        slowk = fastk.rolling(window=slowk_period).mean()
        slowd = slowk.rolling(window=slowd_period).mean()

        return slowk, slowd

def calculate_stochastic_fast(
    high: list[float] | NDArray[np.float64] | pd.Series,
    low: list[float] | NDArray[np.float64] | pd.Series,
    close: list[float] | NDArray[np.float64] | pd.Series,
    fastk_period: int = 14,
    fastd_period: int = 3,
) -> tuple[pd.Series, pd.Series]:
    """
    Calculate Stochastic Fast.
    Returns (fastk, fastd) series.

    Args:
        high: High prices
        low: Low prices
        close: Close prices
        fastk_period: Time period for building the Fast-K line
        fastd_period: Smoothing for making the Fast-D line

    Returns:
        tuple of (fastk, fastd) series.
    """
    try:
        # Convert inputs to numpy arrays
        high_arr = (
            np.array(high, dtype=np.float64)
            if isinstance(high, (list, pd.Series))
            else high
        )
        low_arr = (
            np.array(low, dtype=np.float64)
            if isinstance(low, (list, pd.Series))
            else low
        )
        close_arr = (
            np.array(close, dtype=np.float64)
            if isinstance(close, (list, pd.Series))
            else close
        )

        if len(close_arr) < fastk_period:
            empty = pd.Series(dtype=float)
            return empty, empty

        fastk, fastd = _talib.stochf(
            high_arr,
            low_arr,
            close_arr,
            fastk_period=fastk_period,
            fastd_period=fastd_period,
        )

        index = close.index if isinstance(close, pd.Series) else None
        return (
            pd.Series(fastk, index=index),
            pd.Series(fastd, index=index),
        )
    except Exception:
        # Fallback using pandas
        s_close = pd.Series(close) if not isinstance(close, pd.Series) else close
        s_high = pd.Series(high) if not isinstance(high, pd.Series) else high
        s_low = pd.Series(low) if not isinstance(low, pd.Series) else low

        lowest_low = s_low.rolling(window=fastk_period).min()
        highest_high = s_high.rolling(window=fastk_period).max()

        # Avoid division by zero
        denom = highest_high - lowest_low
        denom = denom.replace(0, np.nan)

        fastk = ((s_close - lowest_low) / denom) * 100
        fastd = fastk.rolling(window=fastd_period).mean()

        return fastk, fastd

def calculate_williams_r(
    high: list[float] | NDArray[np.float64] | pd.Series,
    low: list[float] | NDArray[np.float64] | pd.Series,
    close: list[float] | NDArray[np.float64] | pd.Series,
    period: int = 14,
) -> pd.Series:
    """
    Calculate Williams %R.
    Returns the full series of Williams %R values.

    Args:
        high: High prices
        low: Low prices
        close: Close prices
        period: Time period (default: 14)

    Returns:
        Series of Williams %R values.
    """
    try:
        df = pd.DataFrame(
            {
                "high": high
                if isinstance(high, (pd.Series, np.ndarray))
                else np.array(high),
                "low": low
                if isinstance(low, (pd.Series, np.ndarray))
                else np.array(low),
                "close": close
                if isinstance(close, (pd.Series, np.ndarray))
                else np.array(close),
            }
        )

        if isinstance(close, pd.Series):
            df.index = close.index

        if len(df) < period:
            return pd.Series(dtype=float)

        # Use feature generator
        from ztb.features.generators.technical.momentum.williams_r import WilliamsR

        feature = WilliamsR(period=period)
        result = feature.compute(df)
        return result["williams_r"]
    except Exception:
        return pd.Series(dtype=float)
