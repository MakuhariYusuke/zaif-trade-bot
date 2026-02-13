"""
v460 Data loader — Parquet 読込 + train/eval 分割.

001# §1/§4 準拠.
"""

from __future__ import annotations

import hashlib
import logging
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent


def load_parquet(
    path: str | Path,
    feature_cols: Optional[list[str]] = None,
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

    df = pd.read_parquet(p)
    logger.info(f"Loaded {p.name}: shape={df.shape}")

    if feature_cols:
        # Always keep close for target generation
        keep = list(set(feature_cols + ["close"]))
        missing = [c for c in keep if c not in df.columns]
        if missing:
            raise KeyError(f"Missing columns in {p.name}: {missing}")
        # Also keep timestamp/index columns if present
        for c in ["timestamp", "datetime", "dt"]:
            if c in df.columns and c not in keep:
                keep.append(c)
        df = df[keep]

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
                df[col_name] = (ret > 0).astype(np.int32)
            elif ttype == "magnitude":
                df[col_name] = ret.astype(np.float32)
            elif ttype == "volatility":
                # Rolling stdev of returns over the horizon window
                log_ret = np.log(close / close.shift(1))
                df[col_name] = log_ret.rolling(h).std().shift(-h).astype(np.float32)
            else:
                raise ValueError(f"Unknown target type: {ttype}")

    return df


def compute_data_hash(path: str | Path) -> str:
    """SHA-256 hash of a data file for G0 verification."""
    p = Path(path)
    if not p.is_absolute():
        p = _PROJECT_ROOT / p

    sha = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            sha.update(chunk)
    return sha.hexdigest()


def check_nan_ratio(df: pd.DataFrame, max_ratio: float = 0.01) -> dict[str, Any]:
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
