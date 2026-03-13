#!/usr/bin/env python3
"""
v459 Expanded Feature Generator

73# Review対応: 8特徴→22特徴への拡張
- RSI厳選 (7→3)
- トレンド追加 (SMA, EMA)
- モメンタム追加 (Stochastic, Williams%R, ROC)
- ボラティリティ強化 (ATR, BB)
- 出来高活用 (volume_ratio, OBV)
- 時間特徴 (hour/day sin/cos)
- レジーム特徴 (vol_ratio, vol_rank)
"""

import os
import sys
from pathlib import Path
from datetime import datetime
from typing import Optional
import numpy as np
import pandas as pd

# Project root
project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_root))

# TA-Lib (optional fallback to pandas-ta)
try:
    import talib as ta
    USE_TALIB = True
except ImportError:
    import pandas_ta as pta
    USE_TALIB = False
    print("⚠️ TA-Lib not found, using pandas-ta fallback")

from ztb.utils.logging_utils import get_logger

logger = get_logger(__name__)

# Paths
INPUT_PATH = project_root / "data" / "btc_jpy_1m_dataset.csv"
OUTPUT_PATH = project_root / "data" / "btc_jpy_1m_v459_expanded_features.parquet"


def load_ohlcv_data(path: Path) -> pd.DataFrame:
    """Load OHLCV data from CSV."""
    logger.info(f"Loading data from {path}...")
    
    df = pd.read_csv(path)
    
    # Ensure timestamp column exists and is datetime
    if 'timestamp' in df.columns:
        df['timestamp'] = pd.to_datetime(df['timestamp'])
    elif 'Datetime' in df.columns:
        df['timestamp'] = pd.to_datetime(df['Datetime'])
        df = df.drop(columns=['Datetime'])
    elif 'date' in df.columns:
        df['timestamp'] = pd.to_datetime(df['date'])
        df = df.drop(columns=['date'])
    else:
        raise ValueError("No timestamp column found")
    
    # Ensure required columns
    required = ['open', 'high', 'low', 'close', 'volume']
    # Handle case-insensitive column names
    df.columns = df.columns.str.lower()
    
    for col in required:
        if col not in df.columns:
            raise ValueError(f"Missing required column: {col}")
    
    # Sort by timestamp
    df = df.sort_values('timestamp').reset_index(drop=True)
    
    logger.info(f"Loaded {len(df)} rows, columns: {list(df.columns)}")
    return df


def resample_ohlcv(df: pd.DataFrame, freq: str) -> pd.DataFrame:
    """Resample OHLCV data to different timeframe."""
    df_resampled = df.set_index('timestamp').resample(freq).agg({
        'open': 'first',
        'high': 'max',
        'low': 'min',
        'close': 'last',
        'volume': 'sum'
    }).dropna()
    # Convert to float64 for TA-Lib
    for col in ['open', 'high', 'low', 'close', 'volume']:
        df_resampled[col] = df_resampled[col].astype(np.float64)
    return df_resampled


def compute_rsi(close: pd.Series, period: int = 14) -> pd.Series:
    """Compute RSI."""
    if USE_TALIB:
        return pd.Series(ta.RSI(close.values, timeperiod=period), index=close.index)
    else:
        return close.ta.rsi(length=period)


def compute_sma(close: pd.Series, period: int) -> pd.Series:
    """Compute SMA."""
    if USE_TALIB:
        return pd.Series(ta.SMA(close.values, timeperiod=period), index=close.index)
    else:
        return close.rolling(period).mean()


def compute_ema(close: pd.Series, period: int) -> pd.Series:
    """Compute EMA."""
    if USE_TALIB:
        return pd.Series(ta.EMA(close.values, timeperiod=period), index=close.index)
    else:
        return close.ewm(span=period, adjust=False).mean()


def compute_atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    """Compute ATR."""
    if USE_TALIB:
        return pd.Series(ta.ATR(high.values, low.values, close.values, timeperiod=period), index=close.index)
    else:
        tr1 = high - low
        tr2 = abs(high - close.shift(1))
        tr3 = abs(low - close.shift(1))
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        return tr.rolling(period).mean()


def compute_stochastic(high: pd.Series, low: pd.Series, close: pd.Series, 
                       fastk_period: int = 14, slowk_period: int = 3) -> pd.Series:
    """Compute Stochastic %K."""
    if USE_TALIB:
        slowk, slowd = ta.STOCH(high.values, low.values, close.values, 
                                fastk_period=fastk_period, slowk_period=slowk_period, 
                                slowk_matype=0, slowd_period=3, slowd_matype=0)
        return pd.Series(slowk, index=close.index)
    else:
        lowest_low = low.rolling(fastk_period).min()
        highest_high = high.rolling(fastk_period).max()
        fastk = 100 * (close - lowest_low) / (highest_high - lowest_low + 1e-8)
        return fastk.rolling(slowk_period).mean()


def compute_williams_r(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    """Compute Williams %R."""
    if USE_TALIB:
        return pd.Series(ta.WILLR(high.values, low.values, close.values, timeperiod=period), index=close.index)
    else:
        highest_high = high.rolling(period).max()
        lowest_low = low.rolling(period).min()
        return -100 * (highest_high - close) / (highest_high - lowest_low + 1e-8)


def compute_roc(close: pd.Series, period: int = 10) -> pd.Series:
    """Compute Rate of Change."""
    if USE_TALIB:
        return pd.Series(ta.ROC(close.values, timeperiod=period), index=close.index)
    else:
        return (close - close.shift(period)) / close.shift(period) * 100


def compute_bbands(close: pd.Series, period: int = 20, nbdev: float = 2.0):
    """Compute Bollinger Bands."""
    if USE_TALIB:
        upper, middle, lower = ta.BBANDS(close.values, timeperiod=period, nbdevup=nbdev, nbdevdn=nbdev)
        return pd.Series(upper, index=close.index), pd.Series(middle, index=close.index), pd.Series(lower, index=close.index)
    else:
        middle = close.rolling(period).mean()
        std = close.rolling(period).std()
        upper = middle + nbdev * std
        lower = middle - nbdev * std
        return upper, middle, lower


def compute_obv(close: pd.Series, volume: pd.Series) -> pd.Series:
    """Compute On-Balance Volume."""
    if USE_TALIB:
        return pd.Series(ta.OBV(close.values, volume.values), index=close.index)
    else:
        direction = np.sign(close.diff())
        return (direction * volume).cumsum()


def generate_v459_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Generate v459 expanded feature set (22 features).
    
    Categories:
    1. Price Change (2): log_return, close_change_pct
    2. RSI Selected (3): RSI_M1, RSI_H1, RSI_D1
    3. Trend (3): SMA20_ratio, SMA50_ratio, EMA_slope
    4. Momentum (3): Stochastic_K, Williams_R, ROC
    5. Volatility (3): ATR_norm, ReturnStdDev, BB_position
    6. Volume (2): volume_ratio, OBV_slope
    7. Time (4): hour_sin, hour_cos, day_sin, day_cos
    8. Regime (2): vol_ratio, vol_rank
    """
    logger.info("Generating v459 expanded features (22 features)...")
    
    features = pd.DataFrame(index=df.index)
    
    # Convert to float64 for TA-Lib compatibility
    close = df['close'].astype(np.float64)
    high = df['high'].astype(np.float64)
    low = df['low'].astype(np.float64)
    volume = df['volume'].astype(np.float64)
    timestamp = df['timestamp']
    
    # =========================================================================
    # 1. Price Change (2)
    # =========================================================================
    logger.info("  Computing price change features...")
    features['log_return'] = np.log(close / close.shift(1))
    features['close_change_pct'] = (close - close.shift(1)) / close.shift(1)
    
    # =========================================================================
    # 2. RSI Selected (3) - MTF
    # =========================================================================
    logger.info("  Computing RSI features (MTF)...")
    
    # M1 (base)
    features['RSI_M1'] = compute_rsi(close, 14) / 100  # Normalize to 0-1
    
    # H1 (1 hour = 60 minutes)
    df_h1 = resample_ohlcv(df, '1h')
    rsi_h1 = compute_rsi(df_h1['close'], 14) / 100
    # Map back to minute data
    rsi_h1_mapped = rsi_h1.reindex(timestamp).ffill()
    features['RSI_H1'] = rsi_h1_mapped.values
    
    # D1 (1 day = 1440 minutes)
    df_d1 = resample_ohlcv(df, '1d')
    rsi_d1 = compute_rsi(df_d1['close'], 14) / 100
    rsi_d1_mapped = rsi_d1.reindex(timestamp).ffill()
    features['RSI_D1'] = rsi_d1_mapped.values
    
    # =========================================================================
    # 3. Trend (3)
    # =========================================================================
    logger.info("  Computing trend features...")
    sma20 = compute_sma(close, 20)
    sma50 = compute_sma(close, 50)
    ema12 = compute_ema(close, 12)
    
    features['SMA20_ratio'] = (close / sma20) - 1  # Deviation from SMA20
    features['SMA50_ratio'] = (close / sma50) - 1  # Deviation from SMA50
    features['EMA_slope'] = (ema12 - ema12.shift(5)) / (ema12.shift(5) + 1e-8)  # EMA slope
    
    # =========================================================================
    # 4. Momentum (3)
    # =========================================================================
    logger.info("  Computing momentum features...")
    features['Stochastic_K'] = compute_stochastic(high, low, close, 14, 3) / 100  # Normalize to 0-1
    features['Williams_R'] = compute_williams_r(high, low, close, 14) / -100  # Normalize to 0-1
    features['ROC'] = compute_roc(close, 10) / 100  # Normalize (approx)
    
    # =========================================================================
    # 5. Volatility (3)
    # =========================================================================
    logger.info("  Computing volatility features...")
    atr = compute_atr(high, low, close, 14)
    features['ATR_norm'] = atr / close  # Price-normalized ATR
    
    returns = close.pct_change()
    features['ReturnStdDev'] = returns.rolling(20).std()
    
    upper, middle, lower = compute_bbands(close, 20, 2.0)
    features['BB_position'] = (close - lower) / (upper - lower + 1e-8)  # Position within bands (0-1)
    
    # =========================================================================
    # 6. Volume (2)
    # =========================================================================
    logger.info("  Computing volume features...")
    volume_sma = volume.rolling(20).mean()
    features['volume_ratio'] = volume / (volume_sma + 1e-8)
    
    obv = compute_obv(close, volume)
    # Normalize OBV by using percentage change instead of absolute slope
    obv_pct_change = obv.pct_change(periods=10)
    features['OBV_slope'] = obv_pct_change.clip(-1, 1)  # Clip extreme values
    
    # =========================================================================
    # 7. Time (4)
    # =========================================================================
    logger.info("  Computing time features...")
    features['hour_sin'] = np.sin(2 * np.pi * timestamp.dt.hour / 24)
    features['hour_cos'] = np.cos(2 * np.pi * timestamp.dt.hour / 24)
    features['day_sin'] = np.sin(2 * np.pi * timestamp.dt.dayofweek / 7)
    features['day_cos'] = np.cos(2 * np.pi * timestamp.dt.dayofweek / 7)
    
    # =========================================================================
    # 8. Regime (2)
    # =========================================================================
    logger.info("  Computing regime features...")
    vol_short = returns.rolling(20).std()
    vol_long = returns.rolling(100).std()
    features['vol_ratio'] = (vol_short / (vol_long + 1e-8)).clip(0, 5)
    features['vol_rank'] = vol_short.rolling(1000, min_periods=100).rank(pct=True)
    
    # =========================================================================
    # Finalize
    # =========================================================================
    logger.info("  Finalizing features...")
    
    # Add OHLCV for reference
    features['timestamp'] = timestamp
    features['open'] = df['open']
    features['high'] = df['high']
    features['low'] = df['low']
    features['close'] = df['close']
    features['volume'] = df['volume']
    
    # Handle NaN/Inf
    features = features.replace([np.inf, -np.inf], np.nan)
    
    # Forward fill then backward fill for remaining NaN
    features = features.ffill().bfill()
    
    # Count remaining NaN
    nan_counts = features.isna().sum()
    if nan_counts.any():
        logger.warning(f"Remaining NaN counts:\n{nan_counts[nan_counts > 0]}")
    
    # Convert to float32 for efficiency (except timestamp)
    float_cols = [c for c in features.columns if c != 'timestamp']
    features[float_cols] = features[float_cols].astype(np.float32)
    
    logger.info(f"Generated {len([c for c in features.columns if c not in ['timestamp', 'open', 'high', 'low', 'close', 'volume']])} features")
    logger.info(f"Feature columns: {[c for c in features.columns if c not in ['timestamp', 'open', 'high', 'low', 'close', 'volume']]}")
    
    return features


def main():
    """Main entry point."""
    print("=" * 80)
    print("v459 Expanded Feature Generator")
    print("=" * 80)
    print(f"Input:  {INPUT_PATH}")
    print(f"Output: {OUTPUT_PATH}")
    print("=" * 80)
    
    # Load data
    df = load_ohlcv_data(INPUT_PATH)
    
    # Generate features
    features_df = generate_v459_features(df)
    
    # Save
    logger.info(f"Saving to {OUTPUT_PATH}...")
    features_df.to_parquet(OUTPUT_PATH, index=False)
    
    # Summary
    print("\n" + "=" * 80)
    print("✅ Feature Generation Complete")
    print("=" * 80)
    print(f"Output shape: {features_df.shape}")
    print(f"Feature columns ({len([c for c in features_df.columns if c not in ['timestamp', 'open', 'high', 'low', 'close', 'volume']])} features):")
    feature_cols = [c for c in features_df.columns if c not in ['timestamp', 'open', 'high', 'low', 'close', 'volume']]
    for i, col in enumerate(feature_cols, 1):
        stats = features_df[col].describe()
        print(f"  {i:2d}. {col:20s}: mean={stats['mean']:+.4f}, std={stats['std']:.4f}, min={stats['min']:+.4f}, max={stats['max']:+.4f}")
    print("=" * 80)


if __name__ == "__main__":
    main()
