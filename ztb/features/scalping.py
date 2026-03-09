# Scalping features for high-frequency trading
# スキャルピング向け高頻度取引特徴量

import numpy as np
import pandas as pd

from ztb.features.core.registry import FeatureRegistry

# Register scalping features
register = FeatureRegistry.register

@register("price_velocity")
def price_velocity(df: pd.DataFrame) -> pd.Series:
    """Price velocity - rate of price change over short periods"""
    close = df["close"].values.astype(np.float64, copy=False)
    velocity = np.zeros(len(close), dtype=np.float64)
    if len(close) > 1:
        prev_close = close[:-1]
        valid = prev_close != 0
        velocity[1:][valid] = (close[1:][valid] - prev_close[valid]) / prev_close[valid]
    return pd.Series(velocity, index=df.index, name="price_velocity")

@register("micro_trend")
def micro_trend(df: pd.DataFrame, window: int = 5) -> pd.Series:
    """Micro trend indicator for very short timeframes - simplified for performance"""
    close = df["close"].values.astype(np.float64, copy=False)
    trend = np.zeros(len(close), dtype=np.float64)
    if window > 0 and len(close) > window:
        past_price = close[:-window]
        valid = past_price != 0
        trend[window:][valid] = (close[window:][valid] - past_price[valid]) / past_price[valid]

    return pd.Series(trend, index=df.index, name="micro_trend")

@register("price_acceleration")
def price_acceleration(df: pd.DataFrame, window: int = 3) -> pd.Series:
    """Price acceleration - second derivative of price (rate of change of velocity)"""
    velocity_values = price_velocity(df).values
    acceleration = np.zeros(len(velocity_values), dtype=np.float64)
    if window > 1 and len(velocity_values) >= window:
        acceleration[window:] = (
            velocity_values[window - 1 : -1] - velocity_values[: -window]
        ) / (window - 1)

    return pd.Series(acceleration, index=df.index, name="price_acceleration")

@register("volume_surge")
def volume_surge(
    df: pd.DataFrame, window: int = 5, threshold: float = 2.0
) -> pd.Series:
    """Volume surge detector - identifies sudden volume increases"""
    volume = df["volume"].values.astype(np.float64, copy=False)
    surge = np.zeros(len(volume), dtype=np.float64)
    if window > 0 and len(volume) > window:
        vol_series = pd.Series(volume)
        mean_vol = vol_series.rolling(window, min_periods=window).mean().shift(1).to_numpy()
        std_vol = vol_series.rolling(window, min_periods=window).std(ddof=0).shift(1).to_numpy()
        valid = (std_vol > 0) & (mean_vol > 0)
        z_score = np.zeros(len(volume), dtype=np.float64)
        z_score[valid] = (volume[valid] - mean_vol[valid]) / std_vol[valid]
        surge[valid] = (z_score[valid] > threshold).astype(np.float64)

    return pd.Series(surge, index=df.index, name="volume_surge")

@register("momentum_divergence")
def momentum_divergence(
    df: pd.DataFrame, fast_window: int = 3, slow_window: int = 8
) -> pd.Series:
    """Momentum divergence between fast and slow moving averages"""
    close = df["close"].values
    divergence = np.zeros_like(close, dtype=np.float64)

    for i in range(slow_window, len(close)):
        # Fast momentum (recent trend)
        fast_start = max(0, i - fast_window)
        fast_change = (
            (close[i] - close[fast_start]) / close[fast_start]
            if close[fast_start] != 0
            else 0.0
        )

        # Slow momentum (longer trend)
        slow_start = max(0, i - slow_window)
        slow_change = (
            (close[i] - close[slow_start]) / close[slow_start]
            if close[slow_start] != 0
            else 0.0
        )

        # Divergence: fast minus slow
        divergence[i] = fast_change - slow_change

    return pd.Series(divergence, index=df.index, name="momentum_divergence")

@register("tick_volume_ratio")
def tick_volume_ratio(df: pd.DataFrame, window: int = 10) -> pd.Series:
    """Tick volume ratio - current volume relative to recent average"""
    volume = df["volume"].values.astype(np.float64, copy=False)
    ratio = np.zeros(len(volume), dtype=np.float64)
    if window > 0 and len(volume) > window:
        avg_volume = (
            pd.Series(volume)
            .rolling(window, min_periods=window)
            .mean()
            .shift(1)
            .to_numpy()
        )
        valid = avg_volume > 0
        ratio[valid] = volume[valid] / avg_volume[valid]
        ratio[window:][~valid[window:]] = 1.0

    return pd.Series(ratio, index=df.index, name="tick_volume_ratio")

@register("order_flow_imbalance")
def order_flow_imbalance(df: pd.DataFrame) -> pd.Series:
    """Order flow imbalance indicator"""
    high = df["high"].values.astype(np.float64, copy=False)
    low = df["low"].values.astype(np.float64, copy=False)
    close = df["close"].values.astype(np.float64, copy=False)
    imbalance = np.zeros(len(close), dtype=np.float64)
    if len(close) == 0:
        return pd.Series(imbalance, index=df.index, name="order_flow_imbalance")

    prev_close = np.empty(len(close), dtype=np.float64)
    prev_close[0] = close[0]
    prev_close[1:] = close[:-1]

    body_size = np.abs(close - prev_close)
    max_close = np.maximum(close, prev_close)
    min_close = np.minimum(close, prev_close)
    upper_wick = high - max_close
    lower_wick = min_close - low

    valid = body_size > 0
    imbalance[valid] = (upper_wick[valid] - lower_wick[valid]) / body_size[valid]
    imbalance[0] = 0.0
    return pd.Series(imbalance, index=df.index, name="order_flow_imbalance")

@register("micro_volatility")
def micro_volatility(df: pd.DataFrame, window: int = 5) -> pd.Series:
    """Micro volatility for scalping - percentage returns"""
    close = df["close"].values.astype(np.float64, copy=False)
    volatility = np.zeros(len(close), dtype=np.float64)
    if len(close) == 0 or window <= 1 or window > len(close):
        return pd.Series(volatility, index=df.index, name="micro_volatility")

    pct_returns = np.zeros(len(close) - 1, dtype=np.float64)
    prev_close = close[:-1]
    valid = prev_close != 0
    pct_returns[valid] = (close[1:][valid] - prev_close[valid]) / prev_close[valid]

    rolling_std = (
        pd.Series(pct_returns)
        .rolling(window - 1, min_periods=window - 1)
        .std(ddof=0)
        .to_numpy()
    )
    volatility[window:] = rolling_std[window - 2 : len(close) - 2]
    return pd.Series(volatility, index=df.index, name="micro_volatility")

@register("spread_pressure")
def spread_pressure(df: pd.DataFrame) -> pd.Series:
    """Spread pressure indicator for scalping"""
    high = df["high"].values.astype(np.float64, copy=False)
    low = df["low"].values.astype(np.float64, copy=False)
    close = df["close"].values.astype(np.float64, copy=False)
    pressure = np.zeros(len(close), dtype=np.float64)
    if len(close) > 1:
        prev_close = close[:-1]
        valid = prev_close > 0
        spread = np.zeros(len(close) - 1, dtype=np.float64)
        body_ratio = np.zeros(len(close) - 1, dtype=np.float64)
        spread[valid] = (high[1:][valid] - low[1:][valid]) / prev_close[valid]
        body_ratio[valid] = np.abs(close[1:][valid] - prev_close[valid]) / prev_close[valid]
        pressure[1:] = spread / (body_ratio + 1e-6)
    return pd.Series(pressure, index=df.index, name="spread_pressure")

@register("momentum_burst")
def momentum_burst(df: pd.DataFrame, window: int = 3) -> pd.Series:
    """Momentum burst detector for scalping entries"""
    close = df["close"].values.astype(np.float64, copy=False)
    volume = df["volume"].values.astype(np.float64, copy=False)
    burst = np.zeros(len(close), dtype=np.float64)
    if window > 0 and len(close) > window:
        past_price = close[:-window]
        price_change = np.zeros(len(close) - window, dtype=np.float64)
        valid_price = past_price != 0
        price_change[valid_price] = (close[window:][valid_price] - past_price[valid_price]) / past_price[valid_price]

        volume_avg = (
            pd.Series(volume)
            .rolling(window, min_periods=window)
            .mean()
            .shift(1)
            .to_numpy()[window:]
        )
        volume_ratio = np.ones(len(price_change), dtype=np.float64)
        valid_vol = volume_avg > 0
        volume_ratio[valid_vol] = volume[window:][valid_vol] / volume_avg[valid_vol]

        burst[window:] = price_change * np.log(volume_ratio + 1.0)
    return pd.Series(burst, index=df.index, name="momentum_burst")

@register("liquidity_surge")
def liquidity_surge(df: pd.DataFrame, window: int = 5) -> pd.Series:
    """Liquidity surge detector"""
    volume = df["volume"].values.astype(np.float64, copy=False)
    surge = np.zeros(len(volume), dtype=np.float64)
    if window > 0 and len(volume) > window:
        recent_max = (
            pd.Series(volume)
            .rolling(window, min_periods=window)
            .max()
            .shift(1)
            .to_numpy()
        )
        valid = recent_max > 0
        surge[valid] = volume[valid] / recent_max[valid]
        surge[window:][~valid[window:]] = 1.0
    return pd.Series(surge, index=df.index, name="liquidity_surge")

@register("realized_volatility")
def realized_volatility(df: pd.DataFrame, window: int = 10) -> pd.Series:
    """
    Realized Volatility (RV) - measure of price variability over a period

    RV = sqrt(sum of squared returns over the period)
    Higher RV indicates higher price uncertainty/volatility
    """
    close = df["close"].values.astype(np.float64, copy=False)
    rv = np.zeros(len(close), dtype=np.float64)
    if len(close) == 0 or window <= 1 or window > len(close):
        return pd.Series(rv, index=df.index, name="realized_volatility")

    pct_returns = np.zeros(len(close) - 1, dtype=np.float64)
    prev_close = close[:-1]
    valid = prev_close != 0
    pct_returns[valid] = (close[1:][valid] - prev_close[valid]) / prev_close[valid]

    squared_returns = pct_returns ** 2
    cumsum = np.concatenate(([0.0], np.cumsum(squared_returns)))
    end_idx = np.arange(window, len(close), dtype=np.int64)
    rv[end_idx] = np.sqrt(cumsum[end_idx - 1] - cumsum[end_idx - window])

    return pd.Series(rv, index=df.index, name="realized_volatility")
