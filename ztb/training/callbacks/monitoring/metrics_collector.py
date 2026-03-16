#!/usr/bin/env python3
"""
Enhanced Metrics Collection System.

This module provides comprehensive metrics collection capabilities that work
both during training and in standalone monitoring scenarios.
"""
from __future__ import annotations

import json
import logging
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Callable

from ztb.io.state_persistence import read_state_payload, write_state_payload

from ..performance.memory_optimizer import LRUCache, MemoryPool, WeakRefRegistry

@dataclass
class MetricDefinition:
    """Definition of a metric to be collected."""

    name: str
    description: str = ""
    unit: str = ""
    metric_type: str = "gauge"  # gauge, counter, histogram, summary
    tags: dict[str, str] = field(default_factory=dict)
    collect_interval: float | None = None  # For periodic collection

@dataclass
class MetricValue:
    """A single metric measurement."""

    name: str
    value: int | float
    timestamp: datetime
    tags: dict[str, str] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

@dataclass
class MetricSeries:
    """Time series data for a metric."""

    definition: MetricDefinition
    values: deque = field(default_factory=lambda: deque(maxlen=10000))
    aggregations: dict[str, Any] = field(default_factory=dict)

    def add_value(self, value: MetricValue) -> None:
        """Add a new value to the series."""
        self.values.append(value)
        self._update_aggregations()

    def _update_aggregations(self) -> None:
        """Update rolling aggregations."""
        if not self.values:
            return

        values = [v.value for v in self.values]
        self.aggregations.update(
            {
                "count": len(values),
                "sum": sum(values),
                "avg": sum(values) / len(values),
                "min": min(values),
                "max": max(values),
                "latest": values[-1],
                "latest_timestamp": self.values[-1].timestamp.isoformat(),
            }
        )

        # Calculate rate if we have enough data points
        if len(self.values) >= 2:
            time_diff = (
                self.values[-1].timestamp - self.values[0].timestamp
            ).total_seconds()
            if time_diff > 0:
                self.aggregations["rate_per_second"] = (
                    values[-1] - values[0]
                ) / time_diff

class MetricsCollector:
    """
    Enhanced metrics collector that works both during training and standalone.

    Features:
    - Real-time metrics collection
    - Historical data storage
    - Configurable retention policies
    - Export capabilities (JSON, CSV)
    - Integration with training callbacks
    """

    def __init__(self, retention_hours: int = 24, max_series_size: int = 10000):
        self.logger = logging.getLogger(__name__)
        self.retention_hours = retention_hours
        self.max_series_size = max_series_size

        # Core data structures
        self.metric_definitions: dict[str, MetricDefinition] = {}
        self.metric_series: dict[str, MetricSeries] = {}
        self.custom_collectors: list[Callable[[], list[MetricValue]]] = []

        # Memory optimization components
        self.metrics_cache = LRUCache(
            max_size=2000
        )  # Cache for frequently accessed metrics
        self.value_pool = MemoryPool(
            pool_size=500
        )  # Reserved for short-lived objects only
        self.weak_ref_registry = WeakRefRegistry()

        # Threading and synchronization
        self._lock = threading.RLock()
        self._collection_thread: threading.Thread | None = None
        self._running = False
        self._collection_interval = 5.0  # seconds

        # Storage
        self._storage_path: Path | None = None

        # Performance tracking
        self.performance_stats = {
            "cache_hits": 0,
            "cache_misses": 0,
            "objects_pooled": 0,
            "objects_created": 0,
        }

    def register_metric(self, definition: MetricDefinition) -> None:
        """Register a metric definition."""
        with self._lock:
            if definition.name in self.metric_definitions:
                self.logger.warning(
                    f"Metric '{definition.name}' already registered, updating"
                )
                self.metric_definitions[definition.name] = definition
                if definition.name in self.metric_series:
                    self.metric_series[definition.name].definition = definition
                else:
                    self.metric_series[definition.name] = MetricSeries(
                        definition=definition,
                        values=deque(maxlen=self.max_series_size),
                    )
            else:
                self.logger.info(f"Registered metric: {definition.name}")
                self.metric_definitions[definition.name] = definition
                self.metric_series[definition.name] = MetricSeries(
                    definition=definition,
                    values=deque(maxlen=self.max_series_size),
                )

    def add_metric_value(
        self,
        name: str,
        value: int | float,
        tags: dict[str, str] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Add a metric value with memory optimization."""
        with self._lock:
            if name not in self.metric_definitions:
                # Auto-register unknown metrics
                self.register_metric(MetricDefinition(name=name))

            # Values stored in a series must remain immutable by reference.
            # Reusing pooled instances here would corrupt historical values.
            metric_value = MetricValue(
                name=name,
                value=value,
                timestamp=datetime.now(),
                tags=tags or {},
                metadata=metadata or {},
            )
            self.performance_stats["objects_created"] += 1

            self.metric_series[name].add_value(metric_value)
            self._invalidate_latest_metrics_cache()

    def add_custom_collector(self, collector: Callable[[], list[MetricValue]]) -> None:
        """Add a custom metrics collector function."""
        with self._lock:
            self.custom_collectors.append(collector)
            self.logger.info(f"Added custom collector: {collector.__name__}")

    def collect_system_metrics(self) -> None:
        """Collect system-level metrics."""
        try:
            import GPUtil
            import psutil

            # CPU metrics
            self.add_metric_value("system.cpu.percent", psutil.cpu_percent(interval=1))
            self.add_metric_value("system.cpu.count", psutil.cpu_count())

            # Memory metrics
            memory = psutil.virtual_memory()
            self.add_metric_value("system.memory.percent", memory.percent)
            self.add_metric_value("system.memory.used_gb", memory.used / (1024**3))
            self.add_metric_value(
                "system.memory.available_gb", memory.available / (1024**3)
            )

            # Disk metrics
            disk = psutil.disk_usage("/")
            self.add_metric_value("system.disk.percent", disk.percent)
            self.add_metric_value("system.disk.used_gb", disk.used / (1024**3))
            self.add_metric_value("system.disk.free_gb", disk.free / (1024**3))

            # GPU metrics (if available)
            try:
                gpus = GPUtil.getGPUs()
                for i, gpu in enumerate(gpus):
                    self.add_metric_value(
                        f"system.gpu.{i}.memory_percent", gpu.memoryUtil * 100
                    )
                    self.add_metric_value(
                        f"system.gpu.{i}.load_percent", gpu.load * 100
                    )
                    self.add_metric_value(
                        f"system.gpu.{i}.temperature", gpu.temperature
                    )
            except Exception:
                pass  # GPU monitoring not available

        except ImportError:
            self.logger.debug(
                "psutil not available, skipping system metrics collection"
            )

    def collect_custom_metrics(self) -> None:
        """Collect metrics from custom collectors."""
        for collector in self.custom_collectors:
            try:
                values = collector()
                for value in values:
                    self.add_metric_value(
                        value.name, value.value, value.tags, value.metadata
                    )
            except Exception as e:
                self.logger.error(
                    f"Error in custom collector {collector.__name__}: {e}"
                )

    def start_collection(self, interval: float = 5.0) -> None:
        """Start periodic metrics collection."""
        if self._running:
            self.logger.warning("Metrics collection already running")
            return

        self._collection_interval = interval
        self._running = True
        self._collection_thread = threading.Thread(
            target=self._collection_loop, name="metrics-collector", daemon=True
        )
        self._collection_thread.start()
        self.logger.info(f"Started metrics collection (interval: {interval}s)")

    def stop_collection(self) -> None:
        """Stop periodic metrics collection."""
        if not self._running:
            return

        self._running = False
        if self._collection_thread:
            self._collection_thread.join(timeout=5.0)
        self.logger.info("Stopped metrics collection")

    def _collection_loop(self) -> None:
        """Main collection loop."""
        while self._running:
            try:
                self.collect_system_metrics()
                self.collect_custom_metrics()
                self._cleanup_old_data()
            except Exception as e:
                self.logger.error(f"Error in metrics collection loop: {e}")

            time.sleep(self._collection_interval)

    def _cleanup_old_data(self) -> None:
        """Clean up old metric data based on retention policy."""
        cutoff_time = datetime.now() - timedelta(hours=self.retention_hours)
        pruned = False

        with self._lock:
            for series in self.metric_series.values():
                # Remove old values
                while series.values and series.values[0].timestamp < cutoff_time:
                    series.values.popleft()
                    pruned = True
                series._update_aggregations()
            if pruned:
                self._invalidate_latest_metrics_cache()

    def _invalidate_latest_metrics_cache(self) -> None:
        """Invalidate derived cache after state mutations."""
        self.metrics_cache.remove("latest_metrics")

    @staticmethod
    def _coerce_str_dict(value: object) -> dict[str, str]:
        """Coerce a JSON object into a `dict[str, str]`."""
        if not isinstance(value, dict):
            return {}
        return {str(k): str(v) for k, v in value.items()}

    @staticmethod
    def _coerce_object_dict(value: object) -> dict[str, Any]:
        """Coerce a JSON object into a `dict[str, Any]`."""
        if not isinstance(value, dict):
            return {}
        return {str(k): v for k, v in value.items()}

    def _serialize_metrics_payload(self) -> dict[str, Any]:
        """Build serializable metrics payload used for export and state."""
        data: dict[str, Any] = {
            "export_time": datetime.now().isoformat(),
            "retention_hours": self.retention_hours,
            "metrics": {},
        }

        with self._lock:
            for name, series in self.metric_series.items():
                data["metrics"][name] = {
                    "definition": {
                        "name": series.definition.name,
                        "description": series.definition.description,
                        "unit": series.definition.unit,
                        "type": series.definition.metric_type,
                        "tags": series.definition.tags,
                    },
                    "values": [
                        {
                            "timestamp": v.timestamp.isoformat(),
                            "value": v.value,
                            "tags": v.tags,
                            "metadata": v.metadata,
                        }
                        for v in series.values
                    ],
                    "aggregations": series.aggregations,
                }
        return data

    def _restore_metrics_payload(self, payload: dict[str, Any]) -> None:
        """Restore metric state from a serialized payload."""
        metrics_payload = payload.get("metrics", {})
        if not isinstance(metrics_payload, dict):
            return

        with self._lock:
            self.metric_definitions.clear()
            self.metric_series.clear()

            for metric_name, metric_data in metrics_payload.items():
                if not isinstance(metric_name, str) or not isinstance(metric_data, dict):
                    continue

                definition_data = metric_data.get("definition", {})
                if not isinstance(definition_data, dict):
                    continue

                definition = MetricDefinition(
                    name=str(definition_data.get("name", metric_name)),
                    description=str(definition_data.get("description", "")),
                    unit=str(definition_data.get("unit", "")),
                    metric_type=str(definition_data.get("type", "gauge")),
                    tags=self._coerce_str_dict(definition_data.get("tags", {})),
                )
                self.register_metric(definition)
                series_name = definition.name

                value_entries = metric_data.get("values", [])
                if not isinstance(value_entries, list):
                    continue

                for value_data in value_entries:
                    if not isinstance(value_data, dict):
                        continue

                    timestamp_raw = value_data.get("timestamp")
                    value_raw = value_data.get("value")
                    if not isinstance(timestamp_raw, str) or not isinstance(
                        value_raw, (int, float)
                    ):
                        continue

                    try:
                        timestamp = datetime.fromisoformat(timestamp_raw)
                    except ValueError:
                        continue

                    metric_value = MetricValue(
                        name=series_name,
                        value=value_raw,
                        timestamp=timestamp,
                        tags=self._coerce_str_dict(value_data.get("tags", {})),
                        metadata=self._coerce_object_dict(
                            value_data.get("metadata", {})
                        ),
                    )
                    self.metric_series[series_name].add_value(metric_value)

            self._invalidate_latest_metrics_cache()

    def get_metric_series(self, name: str) -> MetricSeries | None:
        """Get a metric series by name."""
        return self.metric_series.get(name)

    def get_all_metrics(self) -> dict[str, dict[str, Any]]:
        """Get all current metric values and aggregations."""
        result = {}
        with self._lock:
            for name, series in self.metric_series.items():
                if series.values:
                    result[name] = {
                        "definition": {
                            "name": series.definition.name,
                            "description": series.definition.description,
                            "unit": series.definition.unit,
                            "type": series.definition.metric_type,
                        },
                        "current": series.aggregations,
                        "history_count": len(series.values),
                    }
        return result

    def get_metrics_in_range(
        self, name: str, start_time: datetime, end_time: datetime
    ) -> list[MetricValue]:
        """Get metric values within a time range."""
        series = self.metric_series.get(name)
        if not series:
            return []

        return [v for v in series.values if start_time <= v.timestamp <= end_time]

    def export_metrics(self, filepath: str | Path, format: str = "json") -> None:
        """Export metrics data to file."""
        filepath = Path(filepath)

        if format == "json":
            self._export_json(filepath)
        elif format == "csv":
            self._export_csv(filepath)
        else:
            raise ValueError(f"Unsupported export format: {format}")

    def _export_json(self, filepath: Path) -> None:
        """Export metrics as JSON."""
        data = self._serialize_metrics_payload()
        write_state_payload(filepath, data)

        self.logger.info(f"Exported metrics to {filepath}")

    def _export_csv(self, filepath: Path) -> None:
        """Export metrics as CSV."""
        import csv

        with open(filepath, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["metric_name", "timestamp", "value", "tags", "metadata"])

            with self._lock:
                for name, series in self.metric_series.items():
                    for value in series.values:
                        writer.writerow(
                            [
                                name,
                                value.timestamp.isoformat(),
                                value.value,
                                json.dumps(value.tags),
                                json.dumps(value.metadata),
                            ]
                        )

        self.logger.info(f"Exported metrics to {filepath}")

    def set_storage_path(self, path: str | Path) -> None:
        """set path for persistent storage."""
        self._storage_path = Path(path)
        self._storage_path.mkdir(parents=True, exist_ok=True)

    def save_state(self) -> None:
        """Save current metrics state to disk."""
        if not self._storage_path:
            return

        state_file = self._storage_path / "metrics_state.json"
        self.export_metrics(state_file, "json")

    def load_state(self) -> None:
        """Load metrics state from disk."""
        if not self._storage_path:
            return

        state_file = self._storage_path / "metrics_state.json"
        if not state_file.exists():
            return

        try:
            payload = read_state_payload(state_file)
            self._restore_metrics_payload(payload)

            self.logger.info(f"Loaded metrics state from {state_file}")

        except Exception as e:
            self.logger.error(f"Failed to load metrics state from {state_file}: {e}")

    def get_latest_metrics(self) -> dict[str, Any]:
        """Get latest metrics for all series with caching."""
        with self._lock:
            # Check cache first
            cache_key = "latest_metrics"
            cached_result = self.metrics_cache.get(cache_key)

            if cached_result:
                self.performance_stats["cache_hits"] += 1
                return cached_result

            # Generate fresh result
            result = {}
            for name, series in self.metric_series.items():
                if series.values:
                    latest_value = series.values[-1]  # Most recent value
                    result[name] = {
                        "current": {
                            "value": latest_value.value,
                            "timestamp": latest_value.timestamp.isoformat(),
                            "tags": latest_value.tags,
                            "metadata": latest_value.metadata,
                        },
                        "definition": {
                            "name": series.definition.name,
                            "description": series.definition.description,
                            "unit": series.definition.unit,
                            "type": series.definition.metric_type,
                            "tags": series.definition.tags,
                        },
                        "count": len(series.values),
                        "latest_timestamp": latest_value.timestamp.isoformat(),
                    }

            # Cache the result
            self.metrics_cache.put(cache_key, result)
            self.performance_stats["cache_misses"] += 1

            return result

    # get_all_metrics originally had a second alias implementation here; duplicate removed.

    def get_performance_stats(self) -> dict[str, Any]:
        """Get performance statistics for memory optimization."""
        with self._lock:
            return {
                "cache_stats": {
                    "size": len(self.metrics_cache.cache),
                    "max_size": self.metrics_cache.max_size,
                    "hit_rate": self._calculate_cache_hit_rate(),
                },
                "pool_stats": {
                    "pool_size": len(self.value_pool.pool),
                    "max_pool_size": self.value_pool.max_pool_size,
                    "objects_pooled": self.performance_stats["objects_pooled"],
                    "objects_created": self.performance_stats["objects_created"],
                },
                "registry_stats": {
                    "registered_objects": len(self.weak_ref_registry.registry)
                },
            }

    def _calculate_cache_hit_rate(self) -> float:
        """Calculate cache hit rate."""
        total = (
            self.performance_stats["cache_hits"]
            + self.performance_stats["cache_misses"]
        )
        return self.performance_stats["cache_hits"] / total if total > 0 else 0.0

# Global metrics collector instance
_global_metrics_collector: MetricsCollector | None = None

def get_global_metrics_collector() -> MetricsCollector:
    """Get the global metrics collector instance."""
    global _global_metrics_collector
    if _global_metrics_collector is None:
        _global_metrics_collector = MetricsCollector()
    return _global_metrics_collector

def collect_metric(
    name: str,
    value: int | float,
    tags: dict[str, str] | None = None,
    metadata: dict[str, Any] | None = None,
) -> None:
    """Convenience function to collect a metric."""
    get_global_metrics_collector().add_metric_value(name, value, tags, metadata)
