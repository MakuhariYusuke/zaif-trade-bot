from __future__ import annotations

import gzip
import json
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import TypeAlias, Callable

import numpy as np
import pandas as pd

from ztb.io.jsonl import read_tail_jsonl_objects


_DEFAULT_RESULTS_DIR = Path("results/v460/fill_test")
_DEFAULT_RAW_DIR = Path("data/v460/raw")
JsonRow: TypeAlias = dict[str, object]


@dataclass(frozen=True, slots=True)
class RealEnrichedFillBundle:
    smoke_enriched_df: pd.DataFrame
    trainable_enriched_df: pd.DataFrame


@lru_cache(maxsize=8)
def _cached_recent_fill_records_df(
    *,
    sample_rows: int,
    results_dir_str: str,
) -> pd.DataFrame:
    results_dir = Path(results_dir_str)
    files = sorted(results_dir.glob("fill_records_*.jsonl"))
    if not files:
        return pd.DataFrame()
    if len(files) > 1:
        files = files[:-1]

    chunks: list[pd.DataFrame] = []
    remaining = sample_rows
    for path in reversed(files):
        if remaining <= 0:
            break
        rows = read_tail_jsonl_objects(path, limit=remaining)
        if not rows:
            continue
        frame = pd.DataFrame(rows)
        if len(frame) > remaining:
            chunks.append(frame.tail(remaining))
            remaining = 0
            break
        chunks.append(frame)
        remaining -= len(frame)

    if chunks:
        return pd.concat(reversed(chunks), ignore_index=True)
    return pd.DataFrame()


@lru_cache(maxsize=4)
def cached_latest_fill_records_file(
    results_dir: Path = _DEFAULT_RESULTS_DIR,
) -> Path | None:
    return latest_fill_records_file(results_dir)


def latest_fill_records_file(
    results_dir: Path = _DEFAULT_RESULTS_DIR,
) -> Path | None:
    files = sorted(results_dir.glob("fill_records_*.jsonl"))
    if not files:
        return None
    return files[-1]


def has_fill_records(
    results_dir: Path = _DEFAULT_RESULTS_DIR,
) -> bool:
    return cached_latest_fill_records_file(results_dir) is not None


@lru_cache(maxsize=4)
def has_fill_records_and_raw_data(
    results_dir: Path = _DEFAULT_RESULTS_DIR,
    raw_dir: Path = _DEFAULT_RAW_DIR,
) -> bool:
    return has_fill_records(results_dir) and (raw_dir / "orderbook").exists()


def write_jsonl_sample(path: Path, rows: list[JsonRow]) -> None:
    path.write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n",
        encoding="utf-8",
    )


def write_jsonl_gz(path: Path, rows: list[JsonRow]) -> None:
    with gzip.open(path, "wt", encoding="utf-8", compresslevel=1) as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False))
            f.write("\n")


def load_recent_fill_records_df(
    *,
    sample_rows: int,
    results_dir: Path = _DEFAULT_RESULTS_DIR,
) -> pd.DataFrame:
    """最新側から最大 sample_rows 件を高速取得する."""
    return _cached_recent_fill_records_df(
        sample_rows=sample_rows,
        results_dir_str=str(results_dir.resolve()),
    ).copy(deep=True)


def load_minimum_feature_ready_fill_df(
    *,
    tmp_path: Path,
    load_fn: Callable[[Path], pd.DataFrame],
    feature_builder: Callable[[pd.DataFrame], tuple[pd.DataFrame, object]],
    min_rows: int = 30,
    min_feature_rows: int = 15,
    candidate_limits: tuple[int, ...] = (94, 100, 160, 220),
    results_dir: Path = _DEFAULT_RESULTS_DIR,
) -> pd.DataFrame:
    """最新 fill_records から成立条件を満たす最小限の sample を構築する."""
    latest_file = cached_latest_fill_records_file(results_dir)
    if latest_file is None:
        return pd.DataFrame()
    return write_minimum_feature_ready_fill_sample(
        latest_file=latest_file,
        tmp_path=tmp_path,
        load_fn=load_fn,
        feature_builder=feature_builder,
        min_rows=min_rows,
        min_feature_rows=min_feature_rows,
        candidate_limits=candidate_limits,
    )


def write_minimum_feature_ready_fill_sample(
    *,
    latest_file: Path,
    tmp_path: Path,
    load_fn: Callable[[Path], pd.DataFrame],
    feature_builder: Callable[[pd.DataFrame], tuple[pd.DataFrame, object]],
    min_rows: int = 30,
    min_feature_rows: int = 15,
    candidate_limits: tuple[int, ...] = (94, 100, 160, 220),
) -> pd.DataFrame:
    """成立条件を満たす最小限の実データ sample を tmp_path へ書いて返す."""
    last_df = pd.DataFrame()
    for limit in candidate_limits:
        sample_rows = [
            row for row in read_tail_jsonl_objects(latest_file, limit=limit)
            if isinstance(row, dict)
        ]
        if not sample_rows:
            continue
        sample_path = tmp_path / latest_file.name
        write_jsonl_sample(sample_path, sample_rows)
        df = load_fn(tmp_path)
        last_df = df
        if len(df) < min_rows:
            continue
        try:
            X, _ = feature_builder(df)
        except ValueError:
            continue
        if len(X) >= min_feature_rows:
            return df
    return last_df


def select_minimum_trainable_fill_df(
    *,
    initial_rows: int,
    fallback_rows: int,
    expanded_rows: int,
    min_train_samples: int,
    enrich_fn: Callable[[pd.DataFrame], pd.DataFrame],
    results_dir: Path = _DEFAULT_RESULTS_DIR,
) -> pd.DataFrame:
    """学習成立条件を満たす最小限の fill_df/enriched_df を選ぶ."""
    return _cached_minimum_trainable_fill_df(
        initial_rows=initial_rows,
        fallback_rows=fallback_rows,
        expanded_rows=expanded_rows,
        min_train_samples=min_train_samples,
        enrich_fn=enrich_fn,
        results_dir_str=str(results_dir.resolve()),
    ).copy(deep=True)


def select_real_enriched_fill_bundle(
    *,
    smoke_sample_sizes: tuple[int, ...],
    initial_rows: int,
    fallback_rows: int,
    expanded_rows: int,
    min_train_samples: int,
    enrich_fn: Callable[[pd.DataFrame], pd.DataFrame],
    results_dir: Path = _DEFAULT_RESULTS_DIR,
    required_column: str = "spread_bps_ob",
) -> RealEnrichedFillBundle:
    smoke_df, trainable_df = _cached_real_enriched_fill_bundle(
        smoke_sample_sizes=smoke_sample_sizes,
        initial_rows=initial_rows,
        fallback_rows=fallback_rows,
        expanded_rows=expanded_rows,
        min_train_samples=min_train_samples,
        enrich_fn=enrich_fn,
        results_dir_str=str(results_dir.resolve()),
        required_column=required_column,
    )
    return RealEnrichedFillBundle(
        smoke_enriched_df=smoke_df.copy(deep=True),
        trainable_enriched_df=trainable_df.copy(deep=True),
    )


@lru_cache(maxsize=4)
def _cached_minimum_trainable_fill_df(
    *,
    initial_rows: int,
    fallback_rows: int,
    expanded_rows: int,
    min_train_samples: int,
    enrich_fn: Callable[[pd.DataFrame], pd.DataFrame],
    results_dir_str: str,
) -> pd.DataFrame:
    """学習用 minimal enriched sample の shared cache."""
    results_dir = Path(results_dir_str)
    max_rows = max(initial_rows, fallback_rows, expanded_rows)
    recent_fill_df = load_recent_fill_records_df(
        sample_rows=max_rows,
        results_dir=results_dir,
    )
    if recent_fill_df.empty:
        return pd.DataFrame()

    trainable_mask = (
        recent_fill_df["filled"].astype(bool).to_numpy(copy=False)
        & recent_fill_df["post_fill_30s_pnl"].notna().to_numpy(copy=False)
    )
    reverse_cumsum = np.cumsum(trainable_mask[::-1], dtype=np.int32)
    enough_trainable = np.flatnonzero(reverse_cumsum >= min_train_samples)
    if enough_trainable.size > 0:
        selected_rows = int(enough_trainable[0]) + 1
    else:
        selected_rows = expanded_rows

    if selected_rows > initial_rows and selected_rows < fallback_rows:
        selected_rows = fallback_rows
    selected_rows = min(selected_rows, expanded_rows)
    return enrich_fn(recent_fill_df.tail(selected_rows))


@lru_cache(maxsize=4)
def _cached_real_enriched_fill_bundle(
    *,
    smoke_sample_sizes: tuple[int, ...],
    initial_rows: int,
    fallback_rows: int,
    expanded_rows: int,
    min_train_samples: int,
    enrich_fn: Callable[[pd.DataFrame], pd.DataFrame],
    results_dir_str: str,
    required_column: str = "spread_bps_ob",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    results_dir = Path(results_dir_str)
    sorted_sizes = tuple(sorted(smoke_sample_sizes))
    max_rows = max(expanded_rows, sorted_sizes[-1] if sorted_sizes else 0)
    recent_fill_df = load_recent_fill_records_df(
        sample_rows=max_rows,
        results_dir=results_dir,
    )
    if recent_fill_df.empty:
        return pd.DataFrame(), pd.DataFrame()

    enriched_max = enrich_fn(recent_fill_df.tail(max_rows))
    smoke_df = pd.DataFrame()
    if required_column in enriched_max.columns:
        for sample_rows in sorted_sizes:
            candidate = enriched_max.tail(sample_rows).copy()
            if candidate[required_column].notna().any():
                smoke_df = candidate
                break

    trainable_mask = (
        recent_fill_df["filled"].astype(bool).to_numpy(copy=False)
        & recent_fill_df["post_fill_30s_pnl"].notna().to_numpy(copy=False)
    )
    reverse_cumsum = np.cumsum(trainable_mask[::-1], dtype=np.int32)
    enough_trainable = np.flatnonzero(reverse_cumsum >= min_train_samples)
    if enough_trainable.size > 0:
        selected_rows = int(enough_trainable[0]) + 1
    else:
        selected_rows = expanded_rows

    if selected_rows > initial_rows and selected_rows < fallback_rows:
        selected_rows = fallback_rows
    selected_rows = min(selected_rows, expanded_rows)
    trainable_df = enriched_max.tail(selected_rows).copy()
    return smoke_df, trainable_df


def select_minimum_smoke_enriched_fill_df(
    *,
    sample_sizes: tuple[int, ...],
    enrich_fn: Callable[[pd.DataFrame], pd.DataFrame],
    results_dir: Path = _DEFAULT_RESULTS_DIR,
    required_column: str = "spread_bps_ob",
) -> pd.DataFrame:
    """観測系 smoke が成立する最小限の enriched fill sample を選ぶ."""
    return _cached_minimum_smoke_enriched_fill_df(
        sample_sizes=sample_sizes,
        enrich_fn=enrich_fn,
        results_dir_str=str(results_dir.resolve()),
        required_column=required_column,
    ).copy(deep=True)


@lru_cache(maxsize=4)
def _cached_minimum_smoke_enriched_fill_df(
    *,
    sample_sizes: tuple[int, ...],
    enrich_fn: Callable[[pd.DataFrame], pd.DataFrame],
    results_dir_str: str,
    required_column: str = "spread_bps_ob",
) -> pd.DataFrame:
    """観測系 smoke enriched sample の shared cache."""
    results_dir = Path(results_dir_str)
    if not sample_sizes:
        return pd.DataFrame()

    sorted_sizes = tuple(sorted(sample_sizes))
    recent_fill_df = load_recent_fill_records_df(
        sample_rows=sorted_sizes[-1],
        results_dir=results_dir,
    )
    if recent_fill_df.empty:
        return pd.DataFrame()

    enriched_max = enrich_fn(recent_fill_df.tail(sorted_sizes[-1]))
    if required_column not in enriched_max.columns:
        return pd.DataFrame()

    for sample_rows in sorted_sizes:
        enriched = enriched_max.tail(sample_rows).copy()
        if enriched[required_column].notna().any():
            return enriched
    return pd.DataFrame()
