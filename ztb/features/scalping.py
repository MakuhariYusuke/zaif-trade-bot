# Scalping features for high-frequency trading
# スキャルピング向け高頻度取引特徴量

import numpy as np
import pandas as pd

from .registry import FeatureRegistry

# Register scalping features
register = FeatureRegistry.register


@register("price_velocity")
def price_velocity(df: pd.DataFrame) -> pd.Series:
    """Price velocity - rate of price change over short periods"""
    close = df["close"].values
    velocity = np.zeros_like(close, dtype=np.float64)
    for i in range(1, len(close)):
        if close[i - 1] != 0:  # Avoid division by zero
            velocity[i] = (close[i] - close[i - 1]) / close[i - 1]  # Percentage change
        else:
            velocity[i] = 0.0
    return pd.Series(velocity, index=df.index, name="price_velocity")


@register("micro_trend")
def micro_trend(df: pd.DataFrame, window: int = 5) -> pd.Series:
    """Micro trend indicator for very short timeframes - simplified for performance"""
    close = df["close"].values
    trend = np.zeros_like(close, dtype=np.float64)

    for i in range(window, len(close)):
        # Simple trend: (current - past) / past over window
        past_price = close[i - window]
        if past_price != 0:
            trend[i] = (close[i] - past_price) / past_price
        else:
            trend[i] = 0.0

    return pd.Series(trend, index=df.index, name="micro_trend")


@register("price_acceleration")
def price_acceleration(df: pd.DataFrame, window: int = 3) -> pd.Series:
    """Price acceleration - second derivative of price (rate of change of velocity)"""
    velocity_values = price_velocity(df).values
    acceleration = np.zeros_like(velocity_values, dtype=np.float64)

    for i in range(window, len(velocity_values)):
        # Calculate acceleration as change in velocity
        accel_window = np.asarray(velocity_values[i - window : i])
        if len(accel_window) >= window:
            diff_values = np.diff(accel_window)
            acceleration[i] = np.mean(diff_values) if len(diff_values) > 0 else 0.0
        else:
            acceleration[i] = 0.0

    return pd.Series(acceleration, index=df.index, name="price_acceleration")


@register("volume_surge")
def volume_surge(
    df: pd.DataFrame, window: int = 5, threshold: float = 2.0
) -> pd.Series:
    """Volume surge detector - identifies sudden volume increases"""
    volume = df["volume"].values
    surge = np.zeros_like(volume, dtype=np.float64)

    for i in range(window, len(volume)):
        # Calculate rolling mean and std
        vol_window = np.asarray(volume[i - window : i])
        mean_vol = np.mean(vol_window)
        std_vol = np.std(vol_window)

        if std_vol > 0 and mean_vol > 0:
            # Z-score of current volume
            current_vol = volume[i]
            z_score = (current_vol - mean_vol) / std_vol
            surge[i] = 1.0 if z_score > threshold else 0.0
        else:
            surge[i] = 0.0

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
    volume = df["volume"].values
    ratio = np.zeros_like(volume, dtype=np.float64)

    for i in range(window, len(volume)):
        vol_window = np.asarray(volume[i - window : i])
        avg_volume = np.mean(vol_window) if len(vol_window) > 0 else 0.0
        ratio[i] = volume[i] / avg_volume if avg_volume > 0 else 1.0

    return pd.Series(ratio, index=df.index, name="tick_volume_ratio")


@register("order_flow_imbalance")
def order_flow_imbalance(df: pd.DataFrame) -> pd.Series:
    """Order flow imbalance indicator"""
    high = df["high"].values
    low = df["low"].values
    close = df["close"].values
    imbalance = np.zeros_like(close)
    for i in range(1, len(close)):
        # Simplified order flow: buying pressure vs selling pressure
        body_size = abs(close[i] - close[i - 1])
        upper_wick = high[i] - max(close[i], close[i - 1])
        lower_wick = min(close[i], close[i - 1]) - low[i]

        if body_size > 0:
            imbalance[i] = (upper_wick - lower_wick) / body_size
        else:
            imbalance[i] = 0.0
    return pd.Series(imbalance, index=df.index, name="order_flow_imbalance")


@register("micro_volatility")
def micro_volatility(df: pd.DataFrame, window: int = 5) -> pd.Series:
    """Micro volatility for scalping - percentage returns"""
    close = df["close"].values
    volatility = np.zeros_like(close, dtype=np.float64)
    for i in range(window, len(close)):
        prices = np.asarray(close[i - window : i])
        if len(prices) > 1:
            # Calculate percentage returns
            returns = []
            for j in range(1, len(prices)):
                if prices[j - 1] != 0:
                    ret = (prices[j] - prices[j - 1]) / prices[j - 1]
                    returns.append(ret)
            if returns:
                volatility[i] = np.std(returns)
    return pd.Series(volatility, index=df.index, name="micro_volatility")


@register("spread_pressure")
def spread_pressure(df: pd.DataFrame) -> pd.Series:
    """Spread pressure indicator for scalping"""
    high = df["high"].values
    low = df["low"].values
    close = df["close"].values
    pressure = np.zeros_like(close)
    for i in range(1, len(close)):
        spread = (high[i] - low[i]) / close[i - 1] if close[i - 1] > 0 else 0.0
        body_ratio = (
            abs(close[i] - close[i - 1]) / close[i - 1] if close[i - 1] > 0 else 0.0
        )
        pressure[i] = spread / (body_ratio + 1e-6)  # Avoid division by zero
    return pd.Series(pressure, index=df.index, name="spread_pressure")


@register("momentum_burst")
def momentum_burst(df: pd.DataFrame, window: int = 3) -> pd.Series:
    """Momentum burst detector for scalping entries"""
    close = df["close"].values
    volume = df["volume"].values
    burst = np.zeros_like(close, dtype=np.float64)
    for i in range(window, len(close)):
        # Price momentum (percentage change)
        past_price = close[i - window]
        if past_price != 0:
            price_change = (close[i] - past_price) / past_price
        else:
            price_change = 0.0

        # Volume confirmation
        volume_window = np.asarray(volume[i - window : i])
        volume_avg = np.mean(volume_window) if len(volume_window) > 0 else 0.0
        volume_ratio = volume[i] / volume_avg if volume_avg > 0 else 1.0

        burst[i] = price_change * np.log(volume_ratio + 1.0)
    return pd.Series(burst, index=df.index, name="momentum_burst")


@register("liquidity_surge")
def liquidity_surge(df: pd.DataFrame, window: int = 5) -> pd.Series:
    """Liquidity surge detector"""
    volume = df["volume"].values
    surge = np.zeros_like(volume, dtype=np.float64)
    for i in range(window, len(volume)):
        volume_window = np.asarray(volume[i - window : i])
        recent_max = np.max(volume_window) if len(volume_window) > 0 else 0.0
        surge[i] = volume[i] / recent_max if recent_max > 0 else 1.0
    return pd.Series(surge, index=df.index, name="liquidity_surge")
