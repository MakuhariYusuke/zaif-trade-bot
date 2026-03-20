"""Shared SAC memory monitoring helpers."""

from __future__ import annotations

from ztb.utils.memory_utils import get_memory_usage


def build_post_cycle_memory_details(previous_rss_mb: float) -> dict[str, float]:
    """Build a compact memory diagnostic payload for SAC retrain cycles."""
    usage = get_memory_usage()
    current_rss_mb = float(usage.get("rss", 0.0))
    return {
        "rss_mb": current_rss_mb,
        "rss_delta_mb": current_rss_mb - previous_rss_mb if previous_rss_mb > 0 else 0.0,
        "cache_total_entries": float(usage.get("cache_total_entries", 0.0)),
    }


__all__ = ["build_post_cycle_memory_details"]
