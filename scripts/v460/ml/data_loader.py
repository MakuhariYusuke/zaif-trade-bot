"""057# ML Data Loader: fill records → ML-ready DataFrame.

fill_records_*.jsonl を読み込み、AS/Fill 分類用の特徴量を生成する。
"""

from __future__ import annotations

import glob
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

_DEFAULT_RESULTS_DIR = Path("results/v460/fill_test")


def load_fill_records(
    results_dir: Optional[Path] = None,
) -> pd.DataFrame:
    """fill_records_*.jsonl を読み込んで DataFrame に変換.

    Returns:
        全レコードの DataFrame (cancelled 含む).
    """
    d = results_dir or _DEFAULT_RESULTS_DIR
    files = sorted(glob.glob(str(d / "fill_records_*.jsonl")))
    if not files:
        raise FileNotFoundError(f"No fill_records_*.jsonl in {d}")

    rows: list[dict] = []
    for f in files:
        with open(f) as fh:
            for line in fh:
                rows.append(json.loads(line))

    df = pd.DataFrame(rows)
    logger.info(f"Loaded {len(df)} records from {len(files)} files")
    return df


def build_as_features(
    df: pd.DataFrame,
    *,
    require_spread: bool = False,
) -> tuple[pd.DataFrame, pd.Series]:
    """AS 分類器用の特徴量を構築.

    Args:
        df: load_fill_records() の出力.
        require_spread: True の場合 spread_at_order 必須 (件数減).

    Returns:
        (X, y) タプル. y は adverse_selected_raw (bool → int).
    """
    # filled かつ AS ラベル有りのみ
    mask = df["filled"].astype(bool) & df["adverse_selected_raw"].notna()
    data = df.loc[mask].copy()

    if require_spread:
        data = data.dropna(subset=["spread_at_order", "spread_offset_ratio"])

    if len(data) < 10:
        raise ValueError(f"Insufficient labeled samples: {len(data)}")

    # === 特徴量生成 ===
    features: dict[str, pd.Series] = {}

    # F1: queue_wait_sec (log transform — 右裾が長い)
    features["log_queue_wait"] = np.log1p(data["queue_wait_sec"].astype(float))

    # F2: side (binary)
    features["side_buy"] = (data["side"] == "buy").astype(int)

    # F3: hour_of_day (cyclic encoding)
    ts = data["timestamp"].astype(float)
    hours = ts.apply(lambda t: datetime.fromtimestamp(t).hour)
    features["hour_sin"] = np.sin(2 * np.pi * hours / 24)
    features["hour_cos"] = np.cos(2 * np.pi * hours / 24)

    # F4: fill_price relative to mid (edge proxy)
    if "mid_at_fill" in data.columns:
        mid = data["mid_at_fill"].astype(float)
        fill = data["fill_price"].astype(float)
        # buy: fill < mid が有利, sell: fill > mid が有利
        # 統一指標: (fill - mid) / mid * 10000 * side_sign
        side_sign = data["side"].map({"buy": -1, "sell": 1}).astype(float)
        features["edge_bps"] = (fill - mid) / mid * 10000 * side_sign

    # F5: spread_at_order (JPY, available for subset)
    if "spread_at_order" in data.columns and not require_spread:
        spread = data["spread_at_order"].astype(float)
        features["spread_jpy"] = spread.fillna(spread.median())
    elif require_spread:
        features["spread_jpy"] = data["spread_at_order"].astype(float)

    # F6: spread_offset_ratio
    if "spread_offset_ratio" in data.columns and not require_spread:
        ratio = data["spread_offset_ratio"].astype(float)
        features["offset_ratio"] = ratio.fillna(ratio.median())
    elif require_spread:
        features["offset_ratio"] = data["spread_offset_ratio"].astype(float)

    # F7: regime (one-hot, available for subset)
    if "regime" in data.columns:
        regime = data["regime"].fillna("unknown")
        for val in ["trending", "ranging", "high_vol"]:
            features[f"regime_{val}"] = (regime == val).astype(int)

    X = pd.DataFrame(features, index=data.index)
    y = data["adverse_selected_raw"].astype(int)

    logger.info(
        f"AS features: {X.shape[1]} features, {len(X)} samples, "
        f"AS rate={y.mean():.1%}"
    )
    return X, y


def build_fill_features(
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.Series]:
    """Fill/Timeout 分類器用の特徴量を構築.

    Args:
        df: load_fill_records() の出力.

    Returns:
        (X, y) タプル. y は filled (bool → int).
    """
    # cancelled=True でも cancel_reason を使い分ける
    data = df.copy()
    # cancel_reason が 'timeout' or filled のみ (error 系は除外)
    valid_mask = data["filled"].astype(bool) | (
        data.get("cancel_reason", pd.Series(dtype=str)).isin(
            ["timeout", "order_timeout", None, float("nan")]
        )
        | data.get("cancel_reason", pd.Series(dtype=str)).isna()
    )
    data = data.loc[valid_mask]

    if len(data) < 20:
        raise ValueError(f"Insufficient samples: {len(data)}")

    features: dict[str, pd.Series] = {}

    # F1: side
    features["side_buy"] = (data["side"] == "buy").astype(int)

    # F2: hour
    ts = data["timestamp"].astype(float)
    hours = ts.apply(lambda t: datetime.fromtimestamp(t).hour)
    features["hour_sin"] = np.sin(2 * np.pi * hours / 24)
    features["hour_cos"] = np.cos(2 * np.pi * hours / 24)

    # F3: spread_offset_ratio (if available)
    if "spread_offset_ratio" in data.columns:
        ratio = data["spread_offset_ratio"].astype(float)
        features["offset_ratio"] = ratio.fillna(ratio.median())

    # F4: spread_at_order
    if "spread_at_order" in data.columns:
        spread = data["spread_at_order"].astype(float)
        features["spread_jpy"] = spread.fillna(spread.median())

    # F5: regime
    if "regime" in data.columns:
        regime = data["regime"].fillna("unknown")
        for val in ["trending", "ranging", "high_vol"]:
            features[f"regime_{val}"] = (regime == val).astype(int)

    X = pd.DataFrame(features, index=data.index)
    y = data["filled"].astype(int)

    logger.info(
        f"Fill features: {X.shape[1]} features, {len(X)} samples, "
        f"fill rate={y.mean():.1%}"
    )
    return X, y
