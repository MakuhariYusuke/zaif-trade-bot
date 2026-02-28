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
    bad_combos: set[tuple[str, int]] = set()
    for (side, hour), row in grouped.iterrows():
        if int(row["count"]) >= min_count and float(row["mean"]) < threshold:
            bad_combos.add((str(side), int(hour)))
    return bad_combos


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

    hours_by_side: dict[str, set[int]] = defaultdict(set)
    for side, hour in skip_combos:
        hours_by_side[str(side)].add(int(hour))

    blocked = pd.Series(False, index=frame.index)
    side_values = frame[side_col]
    hour_values = frame[hour_col]
    for side, hours in hours_by_side.items():
        blocked |= (side_values == side) & hour_values.isin(hours)
    return frame.loc[~blocked]
