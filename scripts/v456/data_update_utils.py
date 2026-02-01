"""
Shared helpers for BTC/JPY OHLCV data updates.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import pandas as pd


DEFAULT_CANDIDATES = [
    "data/btc_jpy_real_dataset.csv",
    "data/btc_jpy_1m_v456.csv",
    "data/btc_jpy_1m_v455.csv",
    "data/btc_jpy_1m_v454.csv",
]

REQUIRED_COLUMNS = ["open", "high", "low", "close", "volume"]


def resolve_data_file(project_root: Path, data_file: Optional[Path]) -> Optional[Path]:
    """Resolve an output file path or auto-detect the default dataset."""
    if data_file is not None:
        path = Path(data_file)
        return path if path.is_absolute() else project_root / path

    for candidate in DEFAULT_CANDIDATES:
        path = project_root / candidate
        if path.exists():
            return path
    return None


def ensure_datetime_index(df: pd.DataFrame, tz: str = "UTC") -> pd.DataFrame:
    """Ensure a UTC DatetimeIndex named 'timestamp'."""
    df = df.copy()
    if not isinstance(df.index, pd.DatetimeIndex):
        if "timestamp" in df.columns:
            df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
            df = df.dropna(subset=["timestamp"]).set_index("timestamp")
        else:
            df.index = pd.to_datetime(df.index, utc=True, errors="coerce")
            df = df[~df.index.isna()]

    if df.index.tz is None:
        df.index = df.index.tz_localize(tz)
    else:
        df.index = df.index.tz_convert(tz)

    df.index.name = "timestamp"
    return df


def normalize_ohlcv_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize OHLCV column names and order."""
    df = df.copy()
    if isinstance(df.columns, pd.MultiIndex):
        # yfinance 0.2.37+ returns MultiIndex like ('Open', 'BTC-JPY')
        # Use level 0 (column names) not level -1 (ticker)
        df.columns = df.columns.get_level_values(0)

    df.columns = [str(col).strip().lower().replace(" ", "_") for col in df.columns]

    for col in ("adj_close", "adjclose"):
        if col in df.columns:
            df = df.drop(columns=[col])

    missing = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing:
        raise ValueError(f"Missing OHLCV columns: {missing}")

    return df[REQUIRED_COLUMNS]


def clean_ohlcv(df: pd.DataFrame) -> pd.DataFrame:
    """Drop invalid rows and enforce numeric OHLCV."""
    df = df.copy()
    for col in REQUIRED_COLUMNS:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=REQUIRED_COLUMNS)
    df = df[np.isfinite(df[REQUIRED_COLUMNS]).all(axis=1)]
    df = df[df["high"] >= df["low"]]
    return df


def prepare_new_ohlcv(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize and clean new OHLCV data."""
    df = normalize_ohlcv_columns(df)
    df = ensure_datetime_index(df)
    df = clean_ohlcv(df)
    return df.sort_index()


def validate_ohlcv(
    df: pd.DataFrame,
    min_rows: int = 1,
    expected_interval_seconds: Optional[int] = None,
    require_minute_alignment: bool = True,
    require_volume: bool = False,
) -> Tuple[bool, str]:
    """Validate OHLCV data quality for updates."""
    if df is None or df.empty:
        return False, "empty dataset"

    missing = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing:
        return False, f"missing columns: {missing}"

    if len(df) < min_rows:
        return False, f"too few rows: {len(df)} < {min_rows}"

    if require_minute_alignment:
        if (df.index.second != 0).any() or (df.index.microsecond != 0).any():
            return False, "timestamp not minute-aligned"

    if require_volume and float(df["volume"].sum()) <= 0.0:
        return False, "volume is zero for all rows"

    if expected_interval_seconds is not None and len(df) >= 3:
        deltas = df.index.to_series().diff().dropna().dt.total_seconds()
        if not deltas.empty:
            median = float(deltas.median())
            tolerance = expected_interval_seconds * 0.2
            if abs(median - expected_interval_seconds) > tolerance:
                return False, f"median interval {median:.1f}s"

    return True, ""


def filter_new_rows(existing_df: pd.DataFrame, new_df: pd.DataFrame) -> pd.DataFrame:
    """Filter new rows strictly after the last timestamp."""
    if existing_df.empty:
        return new_df
    last_timestamp = existing_df.index.max()
    return new_df[new_df.index > last_timestamp]


def merge_ohlcv(existing_df: pd.DataFrame, new_df: pd.DataFrame) -> pd.DataFrame:
    """Merge OHLCV dataframes, preferring newer rows."""
    merged = pd.concat([existing_df, new_df], axis=0)
    merged = merged[~merged.index.duplicated(keep="last")]
    return merged.sort_index()


def load_ohlcv_csv(path: Path) -> pd.DataFrame:
    """Load OHLCV CSV with robust timestamp handling."""
    df = pd.read_csv(path)
    if "timestamp" not in df.columns:
        df = pd.read_csv(path, index_col=0, parse_dates=True)
    df = ensure_datetime_index(df)
    df.index.name = "timestamp"
    return df.sort_index()


def save_ohlcv_csv(path: Path, df: pd.DataFrame) -> None:
    """Persist OHLCV data with timestamp index."""
    df = df.copy()
    df.index.name = "timestamp"
    df.to_csv(path)


def fetch_yahoo_ohlcv(
    ticker: str = "BTC-JPY",
    interval: str = "1m",
    period: Optional[str] = "7d",
    start: Optional[pd.Timestamp] = None,
    end: Optional[pd.Timestamp] = None,
) -> pd.DataFrame:
    """Fetch OHLCV data from Yahoo Finance via yfinance."""
    import yfinance as yf

    try:
        if start is not None or end is not None:
            df = yf.download(
                ticker,
                start=start,
                end=end,
                interval=interval,
                progress=False,
                auto_adjust=False,
            )
        else:
            df = yf.download(
                ticker,
                interval=interval,
                period=period,
                progress=False,
                auto_adjust=False,
            )
        
        # 空データチェック
        if df is None or df.empty:
            print("[Yahoo] Warning: Empty data returned")
            return pd.DataFrame()
        
        # マルチインデックスの場合はフラット化
        # yfinance 0.2.37+ returns MultiIndex like ('Open', 'BTC-JPY')
        # Use level 0 (column names) not level -1 (ticker)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        
        return df
        
    except Exception as e:
        print(f"[Yahoo] Error fetching data: {e}")
        return pd.DataFrame()
