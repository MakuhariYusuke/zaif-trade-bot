"""
Data preprocessing utilities for ZTB evaluation.

This module provides common data alignment and preprocessing functions
to ensure consistent evaluation across all features.
"""

import pandas as pd
from typing import Optional, cast, Set, Dict, List


def align_series(
    df: pd.DataFrame,
    tz: str = "UTC",
    min_periods: int = 5,
    fill_method: Optional[str] = "ffill"
) -> pd.DataFrame:
    """
    Align and preprocess time series data for consistent evaluation.

    Args:
        df: Input DataFrame with datetime index
        tz: Timezone to convert to (default: UTC)
        min_periods: Minimum periods required for valid data
        fill_method: Method to fill missing values ('ffill', 'bfill', or None)

    Returns:
        Aligned DataFrame with standardized preprocessing
    """
    if df.empty:
        return df

    # Ensure datetime index
    if not isinstance(df.index, pd.DatetimeIndex):
        if 'timestamp' in df.columns:
            df = df.set_index('timestamp')
        elif 'date' in df.columns:
            df = df.set_index('date')
        else:
            raise ValueError("DataFrame must have datetime index or 'timestamp'/'date' column")

    if not isinstance(df.index, pd.DatetimeIndex):
        # 文字列等を datetime へ
        df.index = pd.to_datetime(df.index)
    # ここで型を確実化
    dt_index = df.index

    # Timezone handling (安全アクセス)
    if dt_index.tz is None:
        dt_index = dt_index.tz_localize(tz)
    else:
        dt_index = dt_index.tz_convert(tz)
    df.index = dt_index

    # Sort & deduplicate
    df = df.sort_index()
    df = df[~df.index.duplicated(keep='first')]

    # Fill missing values
    if fill_method:
        if fill_method == "ffill":
            df = df.ffill()
        elif fill_method == "bfill":
            df = df.bfill()
        else:
            raise ValueError(f"Unsupported fill_method: {fill_method}")

    if len(df) < min_periods:
        raise ValueError(f"Insufficient data: {len(df)} rows, minimum {min_periods} required")

    return df


def prepare_ohlc_data(
    df: pd.DataFrame,
    required_columns: Optional[List[str]] = None
) -> pd.DataFrame:
    """
    Prepare OHLC data with standard validation and preprocessing.

    Args:
        df: Raw OHLC DataFrame
        required_columns: List of required column names

    Returns:
        Preprocessed OHLC DataFrame
    """
    if required_columns is None:
        required_columns = ['open', 'high', 'low', 'close']

    # Check required columns
    missing_cols = [col for col in required_columns if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")

    # Apply common alignment
    df = align_series(df, tz="UTC", min_periods=30)

    # Validate OHLC relationships
    invalid_ohlc = (
        (df['high'] < df['low']) |
        (df['open'] < df['low']) |
        (df['open'] > df['high']) |
        (df['close'] < df['low']) |
        (df['close'] > df['high'])
    )
    if invalid_ohlc.any():
        raise ValueError("Invalid OHLC relationships detected")

    return df


def generate_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Generate time-based features from datetime index.

    Args:
        df: DataFrame with datetime index

    Returns:
        DataFrame with additional time features
    """
    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError("DataFrame must have DatetimeIndex")
    # 明示的コピー
    df = df.copy()
    dt_index = cast(pd.DatetimeIndex, df.index)
    df['DOW'] = dt_index.dayofweek
    df['HourOfDay'] = dt_index.hour
    return df


class SmartPreprocessor:
    """
    Smart preprocessor that calculates only required columns.
    Significantly faster than CommonPreprocessor for features with limited dependencies.
    """

    # Class-level cache for shared calculations across instances
    _global_cache: Dict[str, pd.Series] = {}
    _cache_data_hash: Optional[str] = None

    def __init__(self, required: Set[str]):
        """
        Initialize with required calculations.

        Args:
            required: Set of required calculation keys
                      Supported: 'return', 'abs_return', 'ema:{span}',
                               'rolling_mean:{win}', 'rolling_std:{win}',
                               'rolling_max:{win}', 'rolling_min:{win}'
        """
        self.required = required
        self._instance_cache: Dict[str, pd.Series] = {}

    def preprocess(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Preprocess DataFrame with only required calculations.
        Uses global cache for shared calculations across instances.

        Args:
            df: Input OHLC DataFrame

        Returns:
            DataFrame with required columns added
        """
        out = df.copy()

        # Check if data has changed (simple hash-based check)
        current_hash = str(hash(tuple(out['close'].values[:10])))  # Sample-based hash
        if current_hash != self._cache_data_hash:
            # Data changed, clear global cache
            self._global_cache.clear()
            self._cache_data_hash = current_hash

        # Basic return calculations
        if "return" in self.required:
            col = "return"
            if col not in out.columns:
                cache_key = f"return_{current_hash}"
                if cache_key in self._global_cache:
                    out[col] = self._global_cache[cache_key]
                else:
                    out[col] = out["close"].pct_change().fillna(0)
                    self._global_cache[cache_key] = out[col].copy()

        if "abs_return" in self.required:
            col = "abs_return"
            if col not in out.columns:
                cache_key = f"abs_return_{current_hash}"
                if cache_key in self._global_cache:
                    out[col] = self._global_cache[cache_key]
                else:
                    base_return = out["close"].pct_change().fillna(0)
                    out[col] = base_return.abs()
                    self._global_cache[cache_key] = out[col].copy()

        # EMA calculations
        ema_keys = [k for k in self.required if k.startswith("ema:")]
        for key in ema_keys:
            span = int(key.split(":")[1])
            col = f"ema_{span}"
            if col not in out.columns:
                cache_key = f"ema_{span}_{current_hash}"
                if cache_key in self._global_cache:
                    out[col] = self._global_cache[cache_key]
                else:
                    out[col] = out["close"].ewm(span=span, adjust=False).mean()
                    self._global_cache[cache_key] = out[col].copy()

        # Rolling calculations
        rolling_keys = [k for k in self.required if k.startswith(("rolling_mean:", "rolling_std:", "rolling_max:", "rolling_min:"))]
        for key in rolling_keys:
            parts = key.split(":")
            calc_type = parts[0]
            win = int(parts[1])

            if calc_type == "rolling_mean":
                col = f"rolling_mean_{win}"
                if col not in out.columns:
                    cache_key = f"rolling_mean_{win}_{current_hash}"
                    if cache_key in self._global_cache:
                        out[col] = self._global_cache[cache_key]
                    else:
                        out[col] = out["close"].rolling(win).mean()
                        self._global_cache[cache_key] = out[col].copy()
            elif calc_type == "rolling_std":
                col = f"rolling_std_{win}"
                if col not in out.columns:
                    cache_key = f"rolling_std_{win}_{current_hash}"
                    if cache_key in self._global_cache:
                        out[col] = self._global_cache[cache_key]
                    else:
                        out[col] = out["close"].rolling(win).std()
                        self._global_cache[cache_key] = out[col].copy()
            elif calc_type == "rolling_max":
                col = f"rolling_max_{win}"
                if col not in out.columns:
                    cache_key = f"rolling_max_{win}_{current_hash}"
                    if cache_key in self._global_cache:
                        out[col] = self._global_cache[cache_key]
                    else:
                        out[col] = out["close"].rolling(win).max()
                        self._global_cache[cache_key] = out[col].copy()
            elif calc_type == "rolling_min":
                col = f"rolling_min_{win}"
                if col not in out.columns:
                    cache_key = f"rolling_min_{win}_{current_hash}"
                    if cache_key in self._global_cache:
                        out[col] = self._global_cache[cache_key]
                    else:
                        out[col] = out["close"].rolling(win).min()
                        self._global_cache[cache_key] = out[col].copy()

        return out