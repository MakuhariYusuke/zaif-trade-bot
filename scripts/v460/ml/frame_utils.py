"""Shared DataFrame helpers for v460 ML / analysis scripts."""

from __future__ import annotations

from collections import defaultdict
from datetime import datetime, timezone, tzinfo

import numpy as np
import pandas as pd

_LOCAL_TZ: tzinfo = datetime.now().astimezone().tzinfo or timezone.utc


def compute_local_hour_fraction(timestamps: pd.Series) -> pd.Series:
    """Epoch 秒からローカル時刻ベースの小数 hour をベクトル化計算."""
    ts_utc = pd.to_datetime(timestamps.astype(float), unit="s", utc=True)
    ts_local = ts_utc.dt.tz_convert(_LOCAL_TZ)
    return (
        ts_local.dt.hour.astype(float)
        + ts_local.dt.minute.astype(float) / 60.0
        + ts_local.dt.second.astype(float) / 3600.0
        + ts_local.dt.microsecond.astype(float) / 3_600_000_000.0
    )


def compute_local_hour_cyclic(timestamps: pd.Series) -> tuple[pd.Series, pd.Series]:
    """ローカル時刻ベースの hour cyclic feature を返す."""
    hours = compute_local_hour_fraction(timestamps)
    radians = 2.0 * np.pi * hours / 24.0
    return np.sin(radians), np.cos(radians)


def compute_utc_hour(timestamps: pd.Series) -> pd.Series:
    """Epoch 秒から UTC hour をベクトル化計算."""
    return pd.to_datetime(timestamps.astype(float), unit="s", utc=True).dt.hour


def _select_side_hour_combos(
    grouped: pd.DataFrame,
    *,
    min_count: int,
    mean_mask: np.ndarray,
) -> set[tuple[str, int]]:
    """集約済み side×hour 統計から該当組み合わせを抽出."""
    if grouped.empty:
        return set()
    count_mask = grouped["count"].to_numpy(dtype=np.int64, copy=False) >= min_count
    selected = grouped.index[count_mask & mean_mask]
    return {
        (str(side), int(hour))
        for side, hour in selected.to_flat_index()
    }


def collect_bad_side_hours(
    frame: pd.DataFrame,
    *,
    pnl_col: str,
    threshold: float,
    min_count: int,
    side_col: str = "side",
    hour_col: str = "utc_hour",
) -> set[tuple[str, int]]:
    """side×hour のうち閾値未満の組み合わせを抽出."""
    required = {side_col, hour_col, pnl_col}
    if not required.issubset(frame.columns):
        return set()

    grouped = (
        frame[[side_col, hour_col, pnl_col]]
        .dropna(subset=[pnl_col])
        .groupby([side_col, hour_col], observed=True)[pnl_col]
        .agg(["count", "mean"])
    )
    return _select_side_hour_combos(
        grouped,
        min_count=min_count,
        mean_mask=grouped["mean"].to_numpy(dtype=np.float64, copy=False) < threshold,
    )


def collect_good_side_hours(
    frame: pd.DataFrame,
    *,
    pnl_col: str,
    threshold: float,
    min_count: int,
    side_col: str = "side",
    hour_col: str = "utc_hour",
) -> set[tuple[str, int]]:
    """side×hour のうち閾値超過の組み合わせを抽出."""
    required = {side_col, hour_col, pnl_col}
    if not required.issubset(frame.columns):
        return set()

    grouped = (
        frame[[side_col, hour_col, pnl_col]]
        .dropna(subset=[pnl_col])
        .groupby([side_col, hour_col], observed=True)[pnl_col]
        .agg(["count", "mean"])
    )
    return _select_side_hour_combos(
        grouped,
        min_count=min_count,
        mean_mask=grouped["mean"].to_numpy(dtype=np.float64, copy=False) > threshold,
    )


def _match_side_hour_combos(
    frame: pd.DataFrame,
    combos: set[tuple[str, int]],
    *,
    side_col: str,
    hour_col: str,
) -> pd.Series:
    """side×hour の一致マスクを返す."""
    matched = pd.Series(False, index=frame.index)
    hours_by_side: dict[str, set[int]] = defaultdict(set)
    for side, hour in combos:
        hours_by_side[str(side)].add(int(hour))

    side_values = frame[side_col]
    hour_values = frame[hour_col]
    for side, hours in hours_by_side.items():
        matched |= (side_values == side) & hour_values.isin(hours)
    return matched


def exclude_side_hour_combos(
    frame: pd.DataFrame,
    skip_combos: set[tuple[str, int]],
    *,
    side_col: str = "side",
    hour_col: str = "utc_hour",
) -> pd.DataFrame:
    """side×hour の除外対象をベクトル化マスクで落とす."""
    if not skip_combos or frame.empty:
        return frame
    blocked = _match_side_hour_combos(
        frame,
        skip_combos,
        side_col=side_col,
        hour_col=hour_col,
    )
    return frame.loc[~blocked]


def include_side_hour_combos(
    frame: pd.DataFrame,
    allowed_combos: set[tuple[str, int]],
    *,
    side_col: str = "side",
    hour_col: str = "utc_hour",
) -> pd.DataFrame:
    """許可された side×hour のみをベクトル化マスクで残す."""
    if not allowed_combos or frame.empty:
        return frame.iloc[0:0]
    allowed = _match_side_hour_combos(
        frame,
        allowed_combos,
        side_col=side_col,
        hour_col=hour_col,
    )
    return frame.loc[allowed]
