#!/usr/bin/env python3
"""
memory_monitor.py
Memory monitoring utilities for development and testing
"""

import logging
import threading
import time
from collections import deque
from typing import Any, TypedDict, cast

import psutil

from ztb.trading.environment.constants import BYTES_PER_MB
from ztb.utils.config import ZTBConfig
from ztb.utils.memory_utils import get_memory_usage as get_memory_snapshot

logger = logging.getLogger(__name__)


class PostCycleMemoryStatus(TypedDict):
    rss_mb: float
    rss_delta_mb: float
    cache_total_entries: float
    leak_warning: bool
    rss_warning: bool

class BackgroundMemoryMonitor:
    """Advanced memory monitoring with history and alerting."""

    def __init__(self, config: ZTBConfig | None = None):
        self.config = config or ZTBConfig()
        self.history_size = self.config.get_int("ZTB_MEMORY_HISTORY_SIZE", 100)
        self.memory_history: deque = deque(maxlen=self.history_size)
        self.alert_threshold_mb = self.config.get_int(
            "ZTB_MEMORY_ALERT_THRESHOLD_MB", 1500
        )
        self.warning_threshold_mb = self.config.get_int(
            "ZTB_MEMORY_WARNING_THRESHOLD_MB", 1000
        )
        self.monitoring_active = False
        self.monitor_thread: threading.Thread | None = None
        self._lock = threading.Lock()
        self._stop_event = threading.Event()

    def start_monitoring(self, interval_seconds: float = 5.0) -> None:
        """
        Start background memory monitoring.

        Args:
            interval_seconds: Monitoring interval
        """
        if self.monitoring_active:
            return

        self.monitoring_active = True
        self._stop_event.clear()
        self.monitor_thread = threading.Thread(
            target=self._monitor_loop, args=(interval_seconds,), daemon=True
        )
        self.monitor_thread.start()
        logger.info("Memory monitoring started")

    def stop_monitoring(self) -> None:
        """Stop background memory monitoring."""
        self.monitoring_active = False
        self._stop_event.set()
        if self.monitor_thread:
            self.monitor_thread.join(timeout=1.0)
            self.monitor_thread = None

        # Memory leak prevention: force garbage collection
        import gc
        collected = gc.collect()
        if collected > 0:
            logger.debug(f"Memory monitor cleanup: garbage collection freed {collected} objects")

        logger.info("Memory monitoring stopped")

    def _monitor_loop(self, interval: float) -> None:
        """Background monitoring loop."""
        while not self._stop_event.is_set():
            try:
                self.record_memory_usage()
                self._check_alerts()
                if self._stop_event.wait(timeout=interval):
                    break
            except Exception as e:
                logger.error(f"Memory monitoring error: {e}")
                if self._stop_event.wait(timeout=interval):
                    break

    def record_memory_usage(self) -> float:
        """
        Record current memory usage.

        Returns:
            Current memory usage in MB
        """
        memory_mb = get_memory_usage()
        timestamp = time.time()

        with self._lock:
            self.memory_history.append({"timestamp": timestamp, "memory_mb": memory_mb})

        return memory_mb

    def _check_alerts(self) -> None:
        """Check for memory alerts."""
        if not self.memory_history:
            return

        current_memory = self.memory_history[-1]["memory_mb"]

        if current_memory > self.alert_threshold_mb:
            logger.error(
                f"CRITICAL: Memory usage exceeded alert threshold: "
                f"{current_memory:.1f}MB > {self.alert_threshold_mb}MB"
            )
        elif current_memory > self.warning_threshold_mb:
            logger.warning(
                f"WARNING: Memory usage exceeded warning threshold: "
                f"{current_memory:.1f}MB > {self.warning_threshold_mb}MB"
            )

    def get_memory_stats(self) -> dict[str, Any]:
        """
        Get memory usage statistics.

        Returns:
            Dictionary with memory statistics
        """
        if not self.memory_history:
            return {"current_mb": 0.0, "average_mb": 0.0, "peak_mb": 0.0, "samples": 0}

        with self._lock:
            memories = [entry["memory_mb"] for entry in self.memory_history]

        return {
            "current_mb": memories[-1],
            "average_mb": sum(memories) / len(memories),
            "peak_mb": max(memories),
            "samples": len(memories),
        }

    def get_memory_trend(self) -> str:
        """
        Get memory usage trend.

        Returns:
            Trend description
        """
        if len(self.memory_history) < 2:
            return "insufficient_data"

        with self._lock:
            recent = list(self.memory_history)[-10:]  # Last 10 samples

        if len(recent) < 2:
            return "insufficient_data"

        # Simple linear trend
        start_memory = recent[0]["memory_mb"]
        end_memory = recent[-1]["memory_mb"]
        change = end_memory - start_memory

        if change > 10:  # 10MB increase
            return "increasing"
        elif change < -10:  # 10MB decrease
            return "decreasing"
        else:
            return "stable"

# Global monitor instance
_memory_monitor: BackgroundMemoryMonitor | None = None

def get_memory_monitor() -> BackgroundMemoryMonitor:
    """Get global memory monitor instance."""
    global _memory_monitor
    if _memory_monitor is None:
        _memory_monitor = BackgroundMemoryMonitor()
    return _memory_monitor

def check_memory_usage(threshold_mb: int = 1000) -> None:
    """
    Check current memory usage and warn if above threshold.

    Args:
        threshold_mb: Memory usage threshold in MB
    """
    config = ZTBConfig()
    if config.get_bool("ZTB_DEV_MEMORY_WARN"):
        process = psutil.Process()
        memory_mb = process.memory_info().rss / BYTES_PER_MB
        if memory_mb > threshold_mb:
            print(
                f"WARNING: High memory usage: {memory_mb:.1f}MB (threshold: {threshold_mb}MB)"
            )

def get_memory_usage() -> float:
    """
    Get current memory usage in MB.

    Returns:
        Memory usage in MB
    """
    process = psutil.Process()
    return cast(float, process.memory_info().rss / BYTES_PER_MB)


def build_post_cycle_memory_details(previous_rss_mb: float) -> dict[str, float]:
    """Build a compact post-cycle memory payload from the shared memory snapshot."""
    usage = get_memory_snapshot()
    current_rss_mb = float(usage.get("rss", 0.0))
    return {
        "rss_mb": current_rss_mb,
        "rss_delta_mb": current_rss_mb - previous_rss_mb if previous_rss_mb > 0 else 0.0,
        "cache_total_entries": float(usage.get("cache_total_entries", 0.0)),
    }


def build_post_cycle_memory_status(
    previous_rss_mb: float,
    *,
    rss_warning_mb: float,
    leak_delta_warning_mb: float = 100.0,
) -> PostCycleMemoryStatus:
    """Build post-cycle details plus reusable warning flags."""
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

def log_memory_usage(label: str = "") -> None:
    """
    Log current memory usage with optional label.

    Args:
        label: Optional label for the log message
    """
    memory_mb = get_memory_usage()
    label_str = f" [{label}]" if label else ""
    print(f"Memory usage{label_str}: {memory_mb:.1f}MB")
