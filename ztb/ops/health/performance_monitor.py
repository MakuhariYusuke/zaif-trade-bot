"""
Performance monitoring and trend analysis for Zaif Trade Bot.

This module provides historical performance tracking, trend analysis,
and predictive monitoring for system resources and trading performance.
"""

import json
import logging
import os
import time
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, cast

import psutil

from ztb.utils.logging_utils import get_logger

logger = get_logger(__name__)


@dataclass
class PerformanceSnapshot:
    """Snapshot of system performance metrics."""

    timestamp: datetime
    cpu_percent: float
    memory_percent: float
    disk_usage_percent: float
    network_bytes_sent: int
    network_bytes_recv: int
    gpu_memory_used_mb: Optional[float] = None
    gpu_utilization_percent: Optional[float] = None


@dataclass
class PerformanceTrend:
    """Performance trend analysis result."""

    metric: str
    current_value: float
    average_24h: float
    average_7d: float
    trend_direction: str  # "increasing", "decreasing", "stable"
    trend_strength: float  # 0-1, strength of trend
    is_concerning: bool
    analysis: str


class PerformanceMonitor:
    """
    Monitors and analyzes system performance trends over time.

    Tracks historical performance data and provides trend analysis
    to detect performance degradation or resource issues.
    """

    def __init__(self, data_dir: str = "data/performance"):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.history_file = self.data_dir / "performance_history.json"
        self.max_history_days = 30

    def take_snapshot(self) -> PerformanceSnapshot:
        """Take a current performance snapshot."""
        try:
            # Basic system metrics
            cpu_percent = psutil.cpu_percent(interval=0.1)
            memory = psutil.virtual_memory()
            disk = psutil.disk_usage('/')
            network = psutil.net_io_counters()

            snapshot = PerformanceSnapshot(
                timestamp=datetime.now(),
                cpu_percent=cpu_percent,
                memory_percent=memory.percent,
                disk_usage_percent=disk.percent,
                network_bytes_sent=network.bytes_sent,
                network_bytes_recv=network.bytes_recv
            )

            # Try to get GPU metrics if available
            try:
                # This would require nvidia-ml-py or similar
                # For now, we'll leave GPU metrics as None
                pass
            except ImportError:
                pass

            return snapshot

        except Exception as e:
            logger.error(f"Failed to take performance snapshot: {e}")
            # Return a minimal snapshot with current timestamp
            return PerformanceSnapshot(
                timestamp=datetime.now(),
                cpu_percent=0.0,
                memory_percent=0.0,
                disk_usage_percent=0.0,
                network_bytes_sent=0,
                network_bytes_recv=0
            )

    def save_snapshot(self, snapshot: PerformanceSnapshot) -> None:
        """Save a performance snapshot to history."""
        try:
            # Load existing history
            history = self._load_history()

            # Add new snapshot
            history.append({
                "timestamp": snapshot.timestamp.isoformat(),
                "cpu_percent": snapshot.cpu_percent,
                "memory_percent": snapshot.memory_percent,
                "disk_usage_percent": snapshot.disk_usage_percent,
                "network_bytes_sent": snapshot.network_bytes_sent,
                "network_bytes_recv": snapshot.network_bytes_recv,
                "gpu_memory_used_mb": snapshot.gpu_memory_used_mb,
                "gpu_utilization_percent": snapshot.gpu_utilization_percent
            })

            # Clean old data (keep only last 30 days)
            cutoff_date = datetime.now() - timedelta(days=self.max_history_days)
            history = [
                entry for entry in history
                if datetime.fromisoformat(entry["timestamp"]) > cutoff_date
            ]

            # Save back to file
            with open(self.history_file, 'w') as f:
                json.dump(history, f, indent=2)

        except Exception as e:
            logger.error(f"Failed to save performance snapshot: {e}")

    def _load_history(self) -> List[Dict[str, Any]]:
        """Load performance history from file."""
        if not self.history_file.exists():
            return []

        try:
            with open(self.history_file, 'r') as f:
                return cast(List[Dict[str, Any]], json.load(f))
        except Exception as e:
            logger.error(f"Failed to load performance history: {e}")
            return []

    def analyze_trends(self) -> List[PerformanceTrend]:
        """Analyze performance trends over different time periods."""
        history = self._load_history()
        if len(history) < 2:
            return []

        trends = []

        # Convert history to PerformanceSnapshot objects
        snapshots = []
        for entry in history:
            try:
                snapshots.append(PerformanceSnapshot(
                    timestamp=datetime.fromisoformat(entry["timestamp"]),
                    cpu_percent=entry["cpu_percent"],
                    memory_percent=entry["memory_percent"],
                    disk_usage_percent=entry["disk_usage_percent"],
                    network_bytes_sent=entry["network_bytes_sent"],
                    network_bytes_recv=entry["network_bytes_recv"],
                    gpu_memory_used_mb=entry.get("gpu_memory_used_mb"),
                    gpu_utilization_percent=entry.get("gpu_utilization_percent")
                ))
            except (KeyError, ValueError) as e:
                logger.warning(f"Skipping invalid history entry: {e}")
                continue

        # Analyze each metric
        metrics = [
            ("cpu_percent", "CPU Usage"),
            ("memory_percent", "Memory Usage"),
            ("disk_usage_percent", "Disk Usage")
        ]

        for metric_attr, metric_name in metrics:
            trend = self._analyze_metric_trend(snapshots, metric_attr, metric_name)
            if trend:
                trends.append(trend)

        return trends

    def _analyze_metric_trend(
        self,
        snapshots: List[PerformanceSnapshot],
        metric_attr: str,
        metric_name: str
    ) -> Optional[PerformanceTrend]:
        """Analyze trend for a specific metric."""
        if len(snapshots) < 2:
            return None

        now = datetime.now()

        # Get data for different periods
        last_24h = [s for s in snapshots if (now - s.timestamp).total_seconds() <= 86400]
        last_7d = [s for s in snapshots if (now - s.timestamp).total_seconds() <= 604800]

        if not last_24h:
            return None

        # Current value (most recent)
        current_value = getattr(last_24h[-1], metric_attr)

        # Calculate averages
        avg_24h = sum(getattr(s, metric_attr) for s in last_24h) / len(last_24h)
        avg_7d = sum(getattr(s, metric_attr) for s in last_7d) / len(last_7d) if last_7d else avg_24h

        # Calculate trend direction and strength
        if len(last_24h) >= 2:
            # Simple linear trend over last 24h
            values = [getattr(s, metric_attr) for s in last_24h]
            trend_slope = self._calculate_trend_slope(values)

            if trend_slope > 0.1:
                trend_direction = "increasing"
                trend_strength = min(abs(trend_slope) / 10, 1.0)  # Normalize
            elif trend_slope < -0.1:
                trend_direction = "decreasing"
                trend_strength = min(abs(trend_slope) / 10, 1.0)
            else:
                trend_direction = "stable"
                trend_strength = 0.0
        else:
            trend_direction = "stable"
            trend_strength = 0.0

        # Determine if concerning
        is_concerning = False
        analysis = ""

        if metric_attr == "cpu_percent":
            if current_value > 80:
                is_concerning = True
                analysis = f"High CPU usage ({current_value:.1f}%) may impact trading performance"
            elif trend_direction == "increasing" and trend_strength > 0.3:
                is_concerning = True
                analysis = f"CPU usage trending upward, monitor for performance impact"

        elif metric_attr == "memory_percent":
            if current_value > 85:
                is_concerning = True
                analysis = f"High memory usage ({current_value:.1f}%) may cause system instability"
            elif trend_direction == "increasing" and trend_strength > 0.3:
                is_concerning = True
                analysis = f"Memory usage trending upward, monitor for memory pressure"

        elif metric_attr == "disk_usage_percent":
            if current_value > 90:
                is_concerning = True
                analysis = f"Critical disk usage ({current_value:.1f}%), storage nearly full"
            elif trend_direction == "increasing" and trend_strength > 0.2:
                is_concerning = True
                analysis = f"Disk usage trending upward, plan for storage expansion"

        if not is_concerning:
            analysis = f"{metric_name} is within normal parameters"

        return PerformanceTrend(
            metric=metric_name,
            current_value=current_value,
            average_24h=avg_24h,
            average_7d=avg_7d,
            trend_direction=trend_direction,
            trend_strength=trend_strength,
            is_concerning=is_concerning,
            analysis=analysis
        )

    def _calculate_trend_slope(self, values: List[float]) -> float:
        """Calculate the slope of a trend line."""
        if len(values) < 2:
            return 0.0

        n = len(values)
        x = list(range(n))
        y = values

        # Calculate slope using linear regression
        sum_x = sum(x)
        sum_y = sum(y)
        sum_xy = sum(xi * yi for xi, yi in zip(x, y))
        sum_xx = sum(xi * xi for xi in x)

        slope = (n * sum_xy - sum_x * sum_y) / (n * sum_xx - sum_x * sum_x)
        return slope

    def get_performance_report(self) -> Dict[str, Any]:
        """Generate a comprehensive performance report."""
        trends = self.analyze_trends()
        current_snapshot = self.take_snapshot()

        # Save current snapshot
        self.save_snapshot(current_snapshot)

        return {
            "current_snapshot": {
                "timestamp": current_snapshot.timestamp.isoformat(),
                "cpu_percent": current_snapshot.cpu_percent,
                "memory_percent": current_snapshot.memory_percent,
                "disk_usage_percent": current_snapshot.disk_usage_percent,
                "network_bytes_sent": current_snapshot.network_bytes_sent,
                "network_bytes_recv": current_snapshot.network_bytes_recv
            },
            "trends": [
                {
                    "metric": trend.metric,
                    "current_value": trend.current_value,
                    "average_24h": trend.average_24h,
                    "average_7d": trend.average_7d,
                    "trend_direction": trend.trend_direction,
                    "trend_strength": trend.trend_strength,
                    "is_concerning": trend.is_concerning,
                    "analysis": trend.analysis
                }
                for trend in trends
            ],
            "summary": {
                "total_trends_analyzed": len(trends),
                "concerning_trends": sum(1 for t in trends if t.is_concerning),
                "data_points": len(self._load_history())
            }
        }


# Global performance monitor instance
_performance_monitor = None


def get_performance_monitor() -> PerformanceMonitor:
    """Get the global performance monitor instance."""
    global _performance_monitor
    if _performance_monitor is None:
        _performance_monitor = PerformanceMonitor()
    return _performance_monitor


def run_performance_check() -> Dict[str, Any]:
    """Run a performance check and return results."""
    monitor = get_performance_monitor()
    return monitor.get_performance_report()