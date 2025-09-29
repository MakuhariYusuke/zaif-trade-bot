"""
ADX (Average Directional Index) implementation.
トレンド強度を測定する方向性移動指標
"""

import pandas as pd

from ztb.features.registry import FeatureRegistry


@FeatureRegistry.register("ADX")
def compute_adx(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """Compute ADX (Average Directional Index)"""
    high = df["high"]
    low = df["low"]
    close = df["close"]

    # True Range
    tr1 = high - low
    tr2 = (high - close.shift(1)).abs()
    tr3 = (low - close.shift(1)).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

    # Directional Movement
    dm_plus = high - high.shift(1)
    dm_minus = low.shift(1) - low

    dm_plus = dm_plus.where(dm_plus > dm_minus, 0)
    dm_plus = dm_plus.where(dm_plus > 0, 0)

    dm_minus = dm_minus.where(dm_minus > dm_plus, 0)
    dm_minus = dm_minus.where(dm_minus > 0, 0)

    # Smooth with EMA
    tr_smooth = tr.ewm(span=period, adjust=False).mean()
    dm_plus_smooth = (
        pd.Series(dm_plus, index=df.index).ewm(span=period, adjust=False).mean()
    )
    dm_minus_smooth = (
        pd.Series(dm_minus, index=df.index).ewm(span=period, adjust=False).mean()
    )

    # Directional Indicators
    plus_di = 100 * dm_plus_smooth / tr_smooth
    minus_di = 100 * dm_minus_smooth / tr_smooth

    # DX and ADX
    denominator = plus_di + minus_di
    denominator_safe = denominator.replace(0, 1e-6)
    dx = 100 * (plus_di - minus_di).abs() / denominator_safe
    adx = dx.ewm(span=period, adjust=False).mean()

    return adx


@FeatureRegistry.register("PlusDI")
def compute_plus_di(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """Compute +DI (Positive Directional Indicator)"""
    high = df["high"]
    low = df["low"]
    close = df["close"]

    # True Range
    tr1 = high - low
    tr2 = (high - close.shift(1)).abs()
    tr3 = (low - close.shift(1)).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

    # Directional Movement
    dm_plus = high - high.shift(1)
    dm_minus = low.shift(1) - low

    dm_plus = dm_plus.where(dm_plus > dm_minus, 0)
    dm_plus = dm_plus.where(dm_plus > 0, 0)

    dm_minus = dm_minus.where(dm_minus > dm_plus, 0)
    dm_minus = dm_minus.where(dm_minus > 0, 0)

    # Smooth with EMA
    tr_smooth = tr.ewm(span=period, adjust=False).mean()
    dm_plus_smooth = (
        pd.Series(dm_plus, index=df.index).ewm(span=period, adjust=False).mean()
    )

    # +DI
    plus_di = 100 * dm_plus_smooth / tr_smooth

    return plus_di


@FeatureRegistry.register("MinusDI")
def compute_minus_di(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """Compute -DI (Negative Directional Indicator)"""
    high = df["high"]
    low = df["low"]
    close = df["close"]

    # True Range
    tr1 = high - low
    tr2 = (high - close.shift(1)).abs()
    tr3 = (low - close.shift(1)).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

    # Directional Movement
    dm_plus = high - high.shift(1)
    dm_minus = low.shift(1) - low

    dm_plus = dm_plus.where(dm_plus > dm_minus, 0)
    dm_plus = dm_plus.where(dm_plus > 0, 0)

    dm_minus = dm_minus.where(dm_minus > dm_plus, 0)
    dm_minus = dm_minus.where(dm_minus > 0, 0)

    # Smooth with EMA
    tr_smooth = tr.ewm(span=period, adjust=False).mean()
    dm_minus_smooth = (
        pd.Series(dm_minus, index=df.index).ewm(span=period, adjust=False).mean()
    )

    # -DI
    minus_di = 100 * dm_minus_smooth / tr_smooth

    return minus_di
