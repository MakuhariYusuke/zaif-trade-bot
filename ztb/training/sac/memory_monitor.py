"""Shared SAC memory monitoring helpers."""

from __future__ import annotations

from typing import TypedDict

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


class PostCycleMemoryStatus(TypedDict):
    rss_mb: float
    rss_delta_mb: float
    cache_total_entries: float
    leak_warning: bool
    rss_warning: bool


def build_post_cycle_memory_status(
    previous_rss_mb: float,
    *,
    rss_warning_mb: float,
    leak_delta_warning_mb: float = 100.0,
) -> PostCycleMemoryStatus:
    """Build memory details plus reusable warning flags."""
    details = build_post_cycle_memory_details(previous_rss_mb)
    current_rss_mb = float(details.get("rss_mb", 0.0))
    rss_delta_mb = float(details.get("rss_delta_mb", 0.0))
    return {
        "rss_mb": current_rss_mb,
        "rss_delta_mb": rss_delta_mb,
        "cache_total_entries": float(details.get("cache_total_entries", 0.0)),
        "leak_warning": previous_rss_mb > 0 and rss_delta_mb > leak_delta_warning_mb,
        "rss_warning": current_rss_mb > rss_warning_mb,
    }


__all__ = ["build_post_cycle_memory_details", "build_post_cycle_memory_status"]
