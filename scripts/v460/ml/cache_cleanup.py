"""Shared cleanup helpers for long-lived ML scripts.

Keep module-level DataFrame caches bounded and explicitly releasable so
training / analysis CLIs do not retain large frames longer than needed.
"""

from __future__ import annotations

import gc
import logging

from scripts.v460.ml.data_loader import (
    clear_fill_records_cache,
    get_fill_records_cache_stats,
)
from scripts.v460.ml.feature_enricher import (
    clear_raw_load_caches,
    get_raw_load_cache_stats,
)
from ztb.io.advanced_csv import clear_read_csv_cache, get_read_csv_cache_stats
from ztb.utils.memory_utils import get_memory_usage


def get_ml_data_cache_stats() -> dict[str, int]:
    """Return aggregate stats for ML data-loading caches."""
    stats = {}
    stats.update(get_fill_records_cache_stats())
    stats.update(get_raw_load_cache_stats())
    csv_stats = get_read_csv_cache_stats()
    stats["advanced_csv_cache_entries"] = csv_stats.get("entries", 0)
    stats["total_ml_cache_entries"] = (
        stats.get("fill_records_cache_entries", 0)
        + stats.get("orderbook_cache_entries", 0)
        + stats.get("trades_cache_entries", 0)
        + stats.get("advanced_csv_cache_entries", 0)
    )
    return stats


def clear_ml_data_caches(*, collect_garbage: bool = False) -> dict[str, int]:
    """Clear ML data-loading caches and optionally trigger GC."""
    clear_fill_records_cache()
    clear_raw_load_caches()
    clear_read_csv_cache()
    stats = get_ml_data_cache_stats()
    if collect_garbage:
        stats["gc_collected"] = gc.collect()
    return stats


def clear_ml_data_caches_with_log(
    logger: logging.Logger,
    *,
    context: str,
    collect_garbage: bool = False,
) -> dict[str, int]:
    """Clear ML data caches and emit a small diagnostic log.

    Long-lived ML CLIs tend to retain fairly large DataFrames in module-level
    caches. We keep the cleanup behavior centralized so every entrypoint uses
    the same retention policy and the same observability.
    """
    before_usage = get_memory_usage()
    stats = clear_ml_data_caches(collect_garbage=collect_garbage)
    after_usage = get_memory_usage()
    stats["rss_before_mb"] = int(before_usage.get("rss", 0.0))
    stats["rss_after_mb"] = int(after_usage.get("rss", 0.0))
    stats["rss_delta_mb"] = stats["rss_after_mb"] - stats["rss_before_mb"]
    stats["memory_cache_total_entries_before"] = int(
        before_usage.get("cache_total_entries", 0.0)
    )
    stats["memory_cache_total_entries_after"] = int(
        after_usage.get("cache_total_entries", 0.0)
    )
    total_entries = stats.get("total_ml_cache_entries", 0)
    gc_collected = stats.get("gc_collected", 0)
    rss_delta = stats.get("rss_delta_mb", 0)
    log_fn = (
        logger.info
        if total_entries > 0 or gc_collected > 0 or rss_delta != 0
        else logger.debug
    )
    log_fn("[%s] cleared ML data caches: %s", context, stats)
    return stats
