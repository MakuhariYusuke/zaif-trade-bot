"""
v460 Data loader — Parquet 読込 + train/eval 分割.

001# §1/§4 準拠.

003# レビュー反映:
  #12: direction NaN→0 を NaN 維持に修正
  #13: set() による非決定的カラム順序を sorted() に修正
  #14: pd.read_parquet に columns= を渡して必要列のみ読込
"""

from __future__ import annotations

import logging
from functools import lru_cache
from pathlib import Path
from typing import TypedDict

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from ztb.utils.run_manifest import compute_file_hash as _compute_shared_file_hash

logger = logging.getLogger(__name__)

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent


@lru_cache(maxsize=256)
def _read_schema_names_cached(
    path_str: str,
    mtime_ns: int,
    size: int,
) -> tuple[str, ...]:
    del mtime_ns, size
    return tuple(pq.read_schema(path_str).names)


def _read_schema_names(path: Path) -> tuple[str, ...]:
    stat = path.stat()
    return _read_schema_names_cached(str(path), stat.st_mtime_ns, stat.st_size)


def load_parquet(
    path: str | Path,
    feature_cols: list[str] | None = None,
) -> pd.DataFrame:
    """Load a Parquet file, optionally selecting columns.

    Args:
        path: Absolute or project-relative path.
        feature_cols: If given, only these + timestamp/close columns are loaded.

    Returns:
        DataFrame with all or selected columns.
    """
    p = Path(path)
    if not p.is_absolute():
        p = _PROJECT_ROOT / p

    if not p.exists():
        raise FileNotFoundError(f"Data file not found: {p}")

    if feature_cols:
        # 003# #13: sorted() for deterministic column order (set() was non-deterministic)
        # 003# #14: pass columns= to read_parquet for selective I/O
        keep = sorted(set(feature_cols + ["close"]))

        # Peek at available columns for timestamp detection
        schema_cols = _read_schema_names(p)
        for c in ["timestamp", "datetime", "dt"]:
            if c in schema_cols and c not in keep:
                keep.append(c)
        missing = [c for c in keep if c not in schema_cols]
        if missing:
            raise KeyError(f"Missing columns in {p.name}: {missing}")

        df = pd.read_parquet(p, columns=keep)
    else:
        df = pd.read_parquet(p)

    logger.info(f"Loaded {p.name}: shape={df.shape}")

    return df


def split_train_eval(
    df: pd.DataFrame,
    train_end_index: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split DataFrame into train and eval by index.

    Args:
        train_end_index: Last index (exclusive) of training data.

    Returns:
        (train_df, eval_df)
    """
    train = df.iloc[:train_end_index].copy()
    eval_ = df.iloc[train_end_index:].copy()
    logger.info(f"Split: train={len(train)}, eval={len(eval_)}")
    return train, eval_


def generate_targets(
    df: pd.DataFrame,
    horizons: list[int],
    target_types: list[str],
) -> pd.DataFrame:
    """Generate target columns for multiple horizons and types.

    Args:
        df: DataFrame with 'close' column.
        horizons: List of forward horizons (1, 5, 15 etc).
        target_types: List of target types (direction, magnitude, volatility).

    Returns:
        DataFrame with added target columns named ``target_{type}_h{horizon}``.
    """
    df = df.copy()
    close = df["close"]

    for h in horizons:
        future_close = close.shift(-h)
        ret = (future_close - close) / close

        for ttype in target_types:
            col_name = f"target_{ttype}_h{h}"
            if ttype == "direction":
                # 003# #12: preserve NaN where ret is NaN (not ret > 0 → False → 0)
                direction = pd.Series(np.nan, index=df.index, dtype="float32")
                valid = ret.notna()
                direction[valid] = (ret[valid] > 0).astype(np.int32)
                df[col_name] = direction
            elif ttype == "magnitude":
                df[col_name] = ret.astype(np.float32)
            elif ttype == "volatility":
                # Rolling stdev of log returns over the horizon window.
                # BUG FIX: rolling(1).std() is NaN (ddof=1 with 1 sample).
                # Use min window of 2 to produce valid results for h=1.
                log_ret = np.log(close / close.shift(1))
                vol_window = max(h, 2)
                df[col_name] = log_ret.rolling(vol_window).std().shift(-h).astype(np.float32)
            else:
                raise ValueError(f"Unknown target type: {ttype}")

    return df


def compute_data_hash(path: str | Path) -> str:
    """SHA-256 hash of a data file for G0 verification."""
    p = Path(path)
    if not p.is_absolute():
        p = _PROJECT_ROOT / p

    return _compute_shared_file_hash(p)


NaNRatioCheck = TypedDict(
    "NaNRatioCheck",
    {
        "total_cells": int,
        "nan_cells": int,
        "ratio": float,
        "pass": bool,
    },
)


def check_nan_ratio(df: pd.DataFrame, max_ratio: float = 0.01) -> NaNRatioCheck:
    """Check NaN ratio of DataFrame.

    Returns:
        {"total_cells": int, "nan_cells": int, "ratio": float, "pass": bool}
    """
    total = df.size
    nans = int(df.isna().sum().sum())
    ratio = nans / max(total, 1)
    return {
        "total_cells": total,
        "nan_cells": nans,
        "ratio": round(ratio, 6),
        "pass": ratio <= max_ratio,
    }
