from __future__ import annotations

import gzip
import json
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd

from ztb.io.jsonl import read_tail_jsonl_objects


_DEFAULT_RESULTS_DIR = Path("results/v460/fill_test")


def write_jsonl_sample(path: Path, rows: list[dict[str, Any]]) -> None:
    path.write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n",
        encoding="utf-8",
    )


def write_jsonl_gz(path: Path, rows: list[dict[str, Any]]) -> None:
    with gzip.open(path, "wt", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False))
            f.write("\n")


def load_recent_fill_records_df(
    *,
    sample_rows: int,
    results_dir: Path = _DEFAULT_RESULTS_DIR,
) -> pd.DataFrame:
    """最新側から最大 sample_rows 件を高速取得する."""
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
