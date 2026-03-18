"""057# ML Data Loader: fill records → ML-ready DataFrame.

fill_records_*.jsonl を読み込み、AS/Fill 分類用の特徴量を生成する。
"""

from __future__ import annotations

from collections import OrderedDict
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from scripts.v460.ml.frame_utils import compute_local_hour_cyclic

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class _FillRecordCacheEntry:
    """fill_records ロード結果のキャッシュ."""

    file_signature: tuple[tuple[str, int, int], ...]
    df: pd.DataFrame


_FILL_RECORDS_CACHE_MAX_ENTRIES = 8
_FILL_RECORDS_CACHE: OrderedDict[
    tuple[str, tuple[str, ...] | None, bool, int | None],
    _FillRecordCacheEntry,
] = OrderedDict()


def clear_fill_records_cache() -> None:
    """fill_records DataFrame cache を明示的に解放する."""
    _FILL_RECORDS_CACHE.clear()


def get_fill_records_cache_stats() -> dict[str, int]:
    """fill_records cache の件数を返す."""
    return {
        "fill_records_cache_entries": len(_FILL_RECORDS_CACHE),
    }


def _normalize_run_id_filter(
    run_id_filter: Optional[str | list[str]],
) -> tuple[str, ...] | None:
    """run_id_filter をキャッシュキー用に正規化."""
    if run_id_filter is None:
        return None
    if isinstance(run_id_filter, str):
        return (run_id_filter,)
    return tuple(run_id_filter)


def _build_file_signature(files: list[Path]) -> tuple[tuple[str, int, int], ...]:
    """mtime + size ベースの軽量シグネチャ."""
    signature: list[tuple[str, int, int]] = []
    for f in files:
        try:
            st = f.stat()
            signature.append((f.name, st.st_mtime_ns, st.st_size))
        except OSError:
            signature.append((f.name, -1, -1))
    return tuple(signature)


def _load_fill_records_cache(
    cache_key: tuple[str, tuple[str, ...] | None, bool, int | None],
    file_signature: tuple[tuple[str, int, int], ...],
) -> pd.DataFrame | None:
    """シグネチャ一致時に copy を返す."""
    cached = _FILL_RECORDS_CACHE.get(cache_key)
    if cached is None or cached.file_signature != file_signature:
        return None
    _FILL_RECORDS_CACHE.move_to_end(cache_key)
    return cached.df.copy(deep=True)


def _store_fill_records_cache(
    cache_key: tuple[str, tuple[str, ...] | None, bool, int | None],
    file_signature: tuple[tuple[str, int, int], ...],
    df: pd.DataFrame,
) -> None:
    """キャッシュ保存。上限超過時は最古エントリを削除."""
    _FILL_RECORDS_CACHE[cache_key] = _FillRecordCacheEntry(
        file_signature=file_signature,
        df=df,
    )
    _FILL_RECORDS_CACHE.move_to_end(cache_key)
    while len(_FILL_RECORDS_CACHE) > _FILL_RECORDS_CACHE_MAX_ENTRIES:
        _FILL_RECORDS_CACHE.popitem(last=False)


def _normalize_max_files(max_files: Optional[int]) -> int | None:
    """max_files の入力を正規化."""
    if max_files is None:
        return None
    if max_files <= 0:
        raise ValueError("max_files must be >= 1")
    return int(max_files)


def make_preprocessing_pipeline(
    model: object,
) -> Pipeline:
    """Imputer + Scaler + Model の Pipeline を構築.

    059# P0-1: 前処理リーク防止。fold 内で fit する。
    """
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("model", model),
    ])

_DEFAULT_RESULTS_DIR = Path("results/v460/fill_test")


def load_fill_records(
    results_dir: Optional[Path] = None,
    *,
    run_id_filter: Optional[str | list[str]] = None,
    exclude_missing_run_id: bool = False,
    max_files: Optional[int] = None,
) -> pd.DataFrame:
    """fill_records_*.jsonl を読み込んで DataFrame に変換.

    Args:
        results_dir: JSONL ファイルのディレクトリ.
        run_id_filter: 指定した run_id のみに絞り込む (122# E13).
            str なら単一 run_id, list なら複数 run_id でフィルタ.
        exclude_missing_run_id: True で run_id が None/欠損のレコードを除外.
        max_files: 最新側から読み込む fill_records ファイル数の上限.

    Returns:
        全レコードの DataFrame (cancelled 含む).
    """
    from ztb.metrics.fill_quality import (
        fill_records_to_dataframe,
        iter_fill_records,
        list_fill_record_files,
    )

    d = results_dir or _DEFAULT_RESULTS_DIR
    max_files_limit = _normalize_max_files(max_files)
    files = list_fill_record_files(d, include_emergency=False)
    if not files:
        raise FileNotFoundError(f"No fill_records_*.jsonl in {d}")
    if max_files_limit is not None and len(files) > max_files_limit:
        files = files[-max_files_limit:]

    run_id_filter_key = _normalize_run_id_filter(run_id_filter)
    run_id_filter_set = set(run_id_filter_key) if run_id_filter_key is not None else None
    cache_key = (
        str(d.resolve()),
        run_id_filter_key,
        bool(exclude_missing_run_id),
        max_files_limit,
    )
    file_signature = _build_file_signature(files)
    cached_df = _load_fill_records_cache(cache_key, file_signature)
    if cached_df is not None:
        return cached_df

    seen_ids: set[str] = set()
    records = []
    for path in files:
        for record in iter_fill_records(path):
            if record.cycle_id in seen_ids:
                continue
            record_run_id = getattr(record, "run_id", None)
            if exclude_missing_run_id and (record_run_id is None or record_run_id == ""):
                continue
            if run_id_filter_set is not None and record_run_id not in run_id_filter_set:
                continue
            seen_ids.add(record.cycle_id)
            records.append(record)
    df = fill_records_to_dataframe(records)
    total = len(df)

    logger.info(
        f"Loaded {len(df)} records from {len(files)} files"
        + (f" (filtered from {total})" if len(df) != total else "")
    )
    _store_fill_records_cache(cache_key, file_signature, df)
    return df.copy(deep=True)


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

    # F3: hour_of_day (cyclic encoding) — 059# NEW-03: 小数時刻で統一
    ts = data["timestamp"]
    hour_sin, hour_cos = compute_local_hour_cyclic(ts)
    features["hour_sin"] = hour_sin
    features["hour_cos"] = hour_cos

    # F4: fill_price relative to mid (edge proxy)
    if "mid_at_fill" in data.columns:
        mid = data["mid_at_fill"].astype(float)
        fill = data["fill_price"].astype(float)
        # buy: fill < mid が有利, sell: fill > mid が有利
        # 統一指標: (fill - mid) / mid * 10000 * side_sign
        side_sign = data["side"].map({"buy": -1, "sell": 1}).astype(float)
        features["edge_bps"] = (fill - mid) / mid * 10000 * side_sign

    # F5: spread_at_order (JPY, available for subset)
    # NOTE: require_spread=True → dropna 済なので全値 valid.
    #       require_spread=False → NaN 保持。CV fold 内 SimpleImputer が補完 (059# P0-1).
    if "spread_at_order" in data.columns:
        features["spread_jpy"] = data["spread_at_order"].astype(float)

    # F6: spread_offset_ratio
    if "spread_offset_ratio" in data.columns:
        features["offset_ratio"] = data["spread_offset_ratio"].astype(float)

    # F7: regime (one-hot, available for subset)
    # 176# 横展開: trending_up/trending_down も regime_trending=1 に含める
    if "regime" in data.columns:
        regime = data["regime"].fillna("unknown")
        features["regime_trending"] = regime.str.startswith("trending").astype(int)
        features["regime_ranging"] = (regime == "ranging").astype(int)
        features["regime_high_vol"] = (regime == "high_vol").astype(int)

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
    # 059# NEW-05: cancel_reason カラム不在時の安全な処理
    if "cancel_reason" in data.columns:
        cancel_reason = data["cancel_reason"]
        valid_mask = data["filled"].astype(bool) | (
            cancel_reason.isin(["timeout", "order_timeout"])
            | cancel_reason.isna()
        )
    else:
        valid_mask = pd.Series(True, index=data.index)
    data = data.loc[valid_mask]

    if len(data) < 20:
        raise ValueError(f"Insufficient samples: {len(data)}")

    features: dict[str, pd.Series] = {}

    # F1: side
    features["side_buy"] = (data["side"] == "buy").astype(int)

    # F2: hour — 059# NEW-03: 小数時刻で統一
    ts = data["timestamp"]
    hour_sin, hour_cos = compute_local_hour_cyclic(ts)
    features["hour_sin"] = hour_sin
    features["hour_cos"] = hour_cos

    # 060# NOTE: spread_jpy / offset_ratio は Fill 分類器から除外.
    # AS classifier 側で require_spread=True により対応済.
    # Fill では時系列前半が全 NaN のため TSCV で無効化される.

    # F5: regime
    # 176# 横展開: trending_up/trending_down も regime_trending=1 に含める
    if "regime" in data.columns:
        regime = data["regime"].fillna("unknown")
        features["regime_trending"] = regime.str.startswith("trending").astype(int)
        features["regime_ranging"] = (regime == "ranging").astype(int)
        features["regime_high_vol"] = (regime == "high_vol").astype(int)

    X = pd.DataFrame(features, index=data.index)
    y = data["filled"].astype(int)

    logger.info(
        f"Fill features: {X.shape[1]} features, {len(X)} samples, "
        f"fill rate={y.mean():.1%}"
    )
    return X, y
