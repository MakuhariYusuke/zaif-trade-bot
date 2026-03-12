"""Shared update pipeline for v456 market data source scripts."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Optional

import pandas as pd

from scripts.v456.data_update_utils import (
    filter_new_rows,
    load_ohlcv_csv,
    merge_ohlcv,
    prepare_new_ohlcv,
    resolve_data_file,
    save_ohlcv_csv,
    validate_ohlcv,
)


def _close_fetcher(fetcher: Any) -> None:
    close = getattr(fetcher, "close", None)
    if callable(close):
        try:
            close()
        except Exception:
            pass


def collect_new_rows(
    existing_df: pd.DataFrame,
    fetcher: Any,
    source_name: str,
    days: int,
    validate_kwargs: dict[str, Any],
    max_retries: int = 3,
) -> Optional[pd.DataFrame]:
    """Fetch, normalize, filter, and validate new rows from one source."""
    try:
        df_new = fetcher.fetch_recent_ohlc(days=days, max_retries=max_retries)
    except Exception as exc:
        print(f"[{source_name}] Error fetching data: {exc}")
        return None

    if df_new is None or df_new.empty:
        print(f"[{source_name}] No data fetched")
        return None

    try:
        df_new = prepare_new_ohlcv(df_new)
    except Exception as exc:
        print(f"[{source_name}] Invalid data format: {exc}")
        return None

    df_new_filtered = filter_new_rows(existing_df, df_new)
    if df_new_filtered.empty:
        print(f"[{source_name}] No new data after last timestamp")
        return None

    ok, reason = validate_ohlcv(df_new_filtered, **validate_kwargs)
    if not ok:
        print(f"[{source_name}] Data rejected: {reason}")
        return None

    print(f"[{source_name}] ✓ Fetched {len(df_new_filtered)} new records")
    return df_new_filtered


def update_file_with_fetcher(
    *,
    project_root: Path,
    data_file: Optional[Path],
    source_name: str,
    days: int,
    fetcher_factory: Callable[..., Any],
    fetcher_kwargs: dict[str, Any],
    validate_kwargs: dict[str, Any],
    max_retries: int = 3,
) -> bool:
    """Run full update flow for one source and persist to CSV."""
    target = resolve_data_file(project_root, data_file)
    if target is None or not target.exists():
        print("Error: Data file not found. Checked default candidates.")
        return False

    print(f"Target file: {target}")
    print("Loading existing data...")
    try:
        df_existing = load_ohlcv_csv(target)
    except Exception as exc:
        print(f"Error loading existing data: {exc}")
        return False

    print(f"Existing data range: {df_existing.index.min()} to {df_existing.index.max()}")
    print(f"Existing rows: {len(df_existing)}")

    print(f"\n[{source_name}] Fetching new data...")
    fetcher = fetcher_factory(**fetcher_kwargs)
    try:
        df_new_filtered = collect_new_rows(
            existing_df=df_existing,
            fetcher=fetcher,
            source_name=source_name,
            days=days,
            validate_kwargs=validate_kwargs,
            max_retries=max_retries,
        )
    finally:
        _close_fetcher(fetcher)

    if df_new_filtered is None:
        return False

    print(f"New data range: {df_new_filtered.index.min()} to {df_new_filtered.index.max()}")
    print("\nMerging data...")
    df_merged = merge_ohlcv(df_existing, df_new_filtered)
    print(f"Merged data: {len(df_merged)} total records")
    print(f"Merged range: {df_merged.index.min()} to {df_merged.index.max()}")

    print(f"\nSaving to {target}...")
    try:
        save_ohlcv_csv(target, df_merged)
    except Exception as exc:
        print(f"Error saving file: {exc}")
        return False

    print(f"✓ Successfully updated {target}")
    print(f"  Added {len(df_new_filtered)} new records")

    # Explicit cleanup for large runs.
    del df_new_filtered
    del df_merged
    return True
