#!/usr/bin/env python3
"""
Real-Time Monitoring System.

This module provides real-time monitoring capabilities for training and system metrics,
including live dashboards, streaming updates, and performance monitoring.
"""

import asyncio
import json
import logging
import queue
import threading
import time
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any, Callable

import websockets

from ..performance.memory_optimizer import LRUCache, MemoryMonitor, WeakRefRegistry
from .metrics_collector import MetricsCollector, get_global_metrics_collector

@dataclass
class MonitorConfig:
    """Configuration for real-time monitoring."""

    update_interval: float = 1.0  # seconds
    buffer_size: int = 1000
    enable_websocket: bool = True
    websocket_port: int = 8765
    websocket_host: str = "localhost"
    enable_http_server: bool = False
    http_port: int = 8080
    retention_minutes: int = 60
    alert_check_interval: float = 5.0

@dataclass
class MonitorAlert:
    """Alert configuration and state."""

    name: str
    condition: Callable[[dict[str, Any]], bool]
    message: str
    severity: str = "warning"  # info, warning, error, critical
    cooldown_minutes: int = 5
    enabled: bool = True
    last_triggered: datetime | None = None
    trigger_count: int = 0

    def should_trigger(self, metrics: dict[str, Any]) -> bool:
        """Check if alert should trigger."""
        if not self.enabled:
            return False

        # Check cooldown
        if self.last_triggered:
            cooldown_end = self.last_triggered + timedelta(
                minutes=self.cooldown_minutes
            )
            if datetime.now() < cooldown_end:
                return False

        return self.condition(metrics)

    def trigger(self) -> str:
        """Trigger the alert and return message."""
        self.last_triggered = datetime.now()
        self.trigger_count += 1
        return f"[{self.severity.upper()}] {self.message} (count: {self.trigger_count})"

class RealTimeMonitor:
    """
    Real-time monitoring system with live updates and alerting.

    Features:
    - Live metrics streaming via WebSocket
    - Configurable alerts and notifications
    - Performance monitoring and anomaly detection
    - Historical data buffering
    - REST API for external integration
    """

    def __init__(
        self,
        config: MonitorConfig | None = None,
        metrics_collector: MetricsCollector | None = None,
    ):
        self.config = config or MonitorConfig()
        self.metrics_collector = metrics_collector or get_global_metrics_collector()
        self.logger = logging.getLogger(__name__)

        # Core components
        self.alerts: dict[str, MonitorAlert] = {}
        self.active_alerts: list[str] = []
        self.monitoring_data: dict[str, Any] = {}

        # Memory optimization components
        self.data_cache = LRUCache(max_size=5000)  # Cache for monitoring data
        self.memory_monitor = MemoryMonitor()
        self.weak_ref_registry = WeakRefRegistry()

        # Threading and synchronization
        self._lock = threading.RLock()
        self._running = False
        self._monitor_thread: threading.Thread | None = None
        self._websocket_thread: threading.Thread | None = None
        self._alert_thread: threading.Thread | None = None

        # WebSocket components
        self.websocket_server = None
        self.connected_clients: set = set()
        self.update_queue = queue.Queue(maxsize=self.config.buffer_size)

        # HTTP server (future enhancement)
        self.http_server = None

        # Performance tracking
        self.performance_stats = {
            "updates_sent": 0,
            "alerts_triggered": 0,
            "websocket_connections": 0,
            "errors": 0,
            "cache_hits": 0,
            "cache_misses": 0,
            "memory_cleanups": 0,
        }

    def start_monitoring(self) -> None:
        """Start the real-time monitoring system."""
        if self._running:
            self.logger.warning("Monitoring already running")
            return

        self._running = True
        self.logger.info("Starting real-time monitoring system")

        # Start memory monitoring
        self.memory_monitor.start_monitoring()

        # Start monitoring thread
        self._monitor_thread = threading.Thread(
            target=self._monitoring_loop, name="realtime-monitor", daemon=True
        )
        self._monitor_thread.start()

        # Start alert checking thread
        self._alert_thread = threading.Thread(
            target=self._alert_loop, name="alert-monitor", daemon=True
        )
        self._alert_thread.start()

        # Start WebSocket server
        if self.config.enable_websocket:
            self._websocket_thread = threading.Thread(
                target=self._start_websocket_server,
                name="websocket-server",
                daemon=True,
            )
            self._websocket_thread.start()

        # Start HTTP server (if enabled)
        if self.config.enable_http_server:
            self._start_http_server()

    def stop_monitoring(self) -> None:
        """Stop the real-time monitoring system."""
        if not self._running:
            return

        self.logger.info("Stopping real-time monitoring system")
        self._running = False

        # Stop memory monitoring
        self.memory_monitor.stop_monitoring()

        # Stop WebSocket server
        if self.websocket_server:
            # WebSocket server will stop when _running becomes False
            pass

        # Wait for threads to finish
        threads = [self._monitor_thread, self._websocket_thread, self._alert_thread]
        for thread in threads:
            if thread and thread.is_alive():
                thread.join(timeout=5.0)

        self.logger.info("Real-time monitoring system stopped")

    def _monitoring_loop(self) -> None:
        """Main monitoring loop."""
        last_update = time.time()

        while self._running:
            try:
                current_time = time.time()

                # Update monitoring data
                self._update_monitoring_data()

                # Send updates if interval has passed
                if current_time - last_update >= self.config.update_interval:
                    self._send_updates()
                    last_update = current_time

                # Clean up old data and memory
                self._cleanup_old_data()
                self.weak_ref_registry.cleanup()  # Clean up weak references

                time.sleep(0.1)  # Small sleep to prevent busy waiting

            except Exception as e:
                self.logger.error(f"Error in monitoring loop: {e}")
                self.performance_stats["errors"] += 1
                time.sleep(1.0)  # Back off on errors

    def _update_monitoring_data(self) -> None:
        """Update monitoring data from metrics collector."""
        with self._lock:
            # Check cache first for metrics data
            cache_key = "latest_metrics"
            cached_metrics = self.data_cache.get(cache_key)

            if cached_metrics:
                all_metrics = cached_metrics
                self.performance_stats["cache_hits"] += 1
            else:
                all_metrics = self.metrics_collector.get_all_metrics()
                self.data_cache.put(cache_key, all_metrics)
                self.performance_stats["cache_misses"] += 1

            # Get memory statistics
            memory_stats = self.memory_monitor.get_memory_stats()

            # Check memory pressure and trigger cleanup if needed
            memory_pressure = memory_stats.get("memory_pressure", 0)
            if memory_pressure > 0.8:  # 80% memory usage
                self.logger.warning(
                    f"High memory pressure detected: {memory_pressure:.2%}"
                )
                self.memory_monitor.force_cleanup()
                self.performance_stats["memory_cleanups"] += 1

            # Add system status with memory information
            self.monitoring_data.update(
                {
                    "timestamp": datetime.now().isoformat(),
                    "metrics": all_metrics,
                    "active_alerts": self.active_alerts.copy(),
                    "performance_stats": self.performance_stats.copy(),
                    "system_status": self._get_system_status(),
                    "memory_stats": memory_stats,
                }
            )

    def _get_system_status(self) -> dict[str, Any]:
        """Get overall system status."""
        status = {
            "overall": "healthy",
            "issues": [],
            "last_update": datetime.now().isoformat(),
        }

        # Check for critical alerts
        critical_alerts = [
            alert
            for alert in self.active_alerts
            if self.alerts[alert].severity == "critical"
        ]
        if critical_alerts:
            status["overall"] = "critical"
            status["issues"].extend(critical_alerts)

        # Check for error alerts
        error_alerts = [
            alert
            for alert in self.active_alerts
            if self.alerts[alert].severity == "error"
        ]
        if error_alerts and status["overall"] == "healthy":
            status["overall"] = "error"
            status["issues"].extend(error_alerts)

        # Check for warning alerts
        warning_alerts = [
            alert
            for alert in self.active_alerts
            if self.alerts[alert].severity == "warning"
        ]
        if warning_alerts and status["overall"] == "healthy":
            status["overall"] = "warning"
            status["issues"].extend(warning_alerts)

        return status

    def _send_updates(self) -> None:
        """Send monitoring updates to connected clients."""
        try:
            update_data = json.dumps(self.monitoring_data)

            # Add to queue for WebSocket clients
            try:
                self.update_queue.put_nowait(update_data)
            except queue.Full:
                # Remove oldest item if queue is full
                try:
                    self.update_queue.get_nowait()
                    self.update_queue.put_nowait(update_data)
                except queue.Empty:
                    pass

            self.performance_stats["updates_sent"] += 1

        except Exception as e:
            self.logger.error(f"Error sending updates: {e}")
            self.performance_stats["errors"] += 1

    def _alert_loop(self) -> None:
        """Alert checking loop."""
        while self._running:
            try:
                self._check_alerts()
                time.sleep(self.config.alert_check_interval)
            except Exception as e:
                self.logger.error(f"Error in alert loop: {e}")
                time.sleep(5.0)

    def _check_alerts(self) -> None:
        """Check all alerts and trigger if necessary."""
        current_metrics = self.metrics_collector.get_all_metrics()

        with self._lock:
            for alert_name, alert in self.alerts.items():
                try:
                    if alert.should_trigger(current_metrics):
                        message = alert.trigger()
                        self.logger.warning(f"Alert triggered: {message}")

                        # Add to active alerts
                        if alert_name not in self.active_alerts:
                            self.active_alerts.append(alert_name)

                        self.performance_stats["alerts_triggered"] += 1

                        # Send alert notification
                        self._send_alert_notification(alert, message)

                except Exception as e:
                    self.logger.error(f"Error checking alert {alert_name}: {e}")

    def _send_alert_notification(self, alert: MonitorAlert, message: str) -> None:
        """Send alert notification to connected clients."""
        alert_data = {
            "type": "alert",
            "alert_name": alert.name,
            "severity": alert.severity,
            "message": message,
            "timestamp": datetime.now().isoformat(),
        }

        try:
            alert_json = json.dumps(alert_data)
            try:
                self.update_queue.put_nowait(alert_json)
            except queue.Full:
                pass
        except Exception as e:
            self.logger.error(f"Error sending alert notification: {e}")

    def _cleanup_old_data(self) -> None:
        """Clean up old monitoring data."""
        # This is handled by the metrics collector's retention policy
        # Additional cleanup can be added here if needed
        pass

    def add_alert(self, alert: MonitorAlert) -> None:
        """Add an alert to the monitoring system."""
        with self._lock:
            self.alerts[alert.name] = alert
            self.logger.info(f"Added alert: {alert.name}")

    def remove_alert(self, alert_name: str) -> bool:
        """Remove an alert from the monitoring system."""
        with self._lock:
            if alert_name in self.alerts:
                del self.alerts[alert_name]
                if alert_name in self.active_alerts:
                    self.active_alerts.remove(alert_name)
                self.logger.info(f"Removed alert: {alert_name}")
                return True
            return False

    def get_monitoring_snapshot(self) -> dict[str, Any]:
        """Get a snapshot of current monitoring data."""
        with self._lock:
            return self.monitoring_data.copy()

    def get_alert_status(self) -> dict[str, Any]:
        """Get status of all alerts."""
        with self._lock:
            return {
                "alerts": {
                    name: {
                        "severity": alert.severity,
                        "enabled": alert.enabled,
                        "last_triggered": alert.last_triggered.isoformat()
                        if alert.last_triggered
                        else None,
                        "trigger_count": alert.trigger_count,
                        "active": name in self.active_alerts,
                    }
                    for name, alert in self.alerts.items()
                },
                "active_alerts": self.active_alerts.copy(),
            }

    async def _websocket_handler(self, websocket, path):
        """Handle WebSocket connections."""
        self.connected_clients.add(websocket)
        self.performance_stats["websocket_connections"] += 1

        try:
            # Send initial data
            initial_data = json.dumps({"type": "initial", "data": self.monitoring_data})
            await websocket.send(initial_data)

            # Send queued updates
            while self._running:
                try:
                    # Wait for update with timeout
                    update_data = await asyncio.get_event_loop().run_in_executor(
                        None, self.update_queue.get, True, 1.0
                    )
                    await websocket.send(update_data)
                except queue.Empty:
                    # No data available, check if still running
                    continue
                except asyncio.TimeoutError:
                    continue

        except websockets.exceptions.ConnectionClosed:
            pass
        finally:
            self.connected_clients.discard(websocket)

    def _start_websocket_server(self) -> None:
        """Start the WebSocket server."""
        try:
            # Create new event loop for WebSocket server
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

            start_server = websockets.serve(
                self._websocket_handler,
                self.config.websocket_host,
                self.config.websocket_port,
            )

            self.websocket_server = loop.run_until_complete(start_server)

            self.logger.info(
                f"WebSocket server started on ws://{self.config.websocket_host}:{self.config.websocket_port}"
            )

            loop.run_forever()

        except Exception as e:
            self.logger.error(f"Error starting WebSocket server: {e}")

    def _start_http_server(self) -> None:
        """Start the HTTP server for REST API."""
        # Future enhancement - implement REST API
        self.logger.info("HTTP server not yet implemented")

    def get_performance_stats(self) -> dict[str, Any]:
        """Get performance statistics."""
        with self._lock:
            return self.performance_stats.copy()

# Predefined alert conditions
def create_high_cpu_alert(threshold: float = 90.0) -> MonitorAlert:
    """Create alert for high CPU usage."""

    def condition(metrics):
        cpu_metric = metrics.get("system.cpu.percent", {})
        return cpu_metric.get("latest", 0) > threshold

    return MonitorAlert(
        name="high_cpu_usage",
        condition=condition,
        message=f"CPU usage exceeded {threshold}%",
        severity="warning",
        cooldown_minutes=2,
    )

def create_high_memory_alert(threshold: float = 90.0) -> MonitorAlert:
    """Create alert for high memory usage."""

    def condition(metrics):
        mem_metric = metrics.get("system.memory.percent", {})
        return mem_metric.get("latest", 0) > threshold

    return MonitorAlert(
        name="high_memory_usage",
        condition=condition,
        message=f"Memory usage exceeded {threshold}%",
        severity="warning",
        cooldown_minutes=5,
    )

def create_training_stuck_alert(timeout_minutes: int = 30) -> MonitorAlert:
    """Create alert for training appearing stuck."""
    [datetime.now()]

    def condition(metrics):
        # Check if training metrics haven't updated recently
        training_metrics = [
            m
            for m in metrics.values()
            if "training" in m.get("definition", {}).get("name", "").lower()
        ]
        if not training_metrics:
            return False

        # Find latest training metric timestamp
        latest_timestamp = None
        for metric in training_metrics:
            if "latest_timestamp" in metric.get("current", {}):
                ts_str = metric["current"]["latest_timestamp"]
                ts = datetime.fromisoformat(ts_str)
                if latest_timestamp is None or ts > latest_timestamp:
                    latest_timestamp = ts

        if latest_timestamp is None:
            return False

        time_since_update = datetime.now() - latest_timestamp
        return time_since_update > timedelta(minutes=timeout_minutes)

    return MonitorAlert(
        name="training_stuck",
        condition=condition,
        message=f"No training progress for {timeout_minutes} minutes",
        severity="error",
        cooldown_minutes=10,
    )

    def get_current_metrics(self) -> dict[str, Any]:
        """Get current monitoring metrics with caching."""
        with self._lock:
            # Try cache first
            cache_key = "current_metrics"
            cached_data = self.data_cache.get(cache_key)

            if cached_data:
                self.performance_stats["cache_hits"] += 1
                return cached_data

            # Generate fresh data
            metrics = {
                "monitoring_data": self.monitoring_data.copy(),
                "memory_stats": self.memory_monitor.get_memory_stats(),
                "cache_stats": {
                    "size": len(self.data_cache.cache),
                    "max_size": self.data_cache.max_size,
                    "hit_rate": self._calculate_cache_hit_rate(),
                },
                "timestamp": datetime.now().isoformat(),
            }

            # Cache the result
            self.data_cache.put(cache_key, metrics)
            self.performance_stats["cache_misses"] += 1

            return metrics

    def _calculate_cache_hit_rate(self) -> float:
        """Calculate cache hit rate."""
        total_requests = (
            self.performance_stats["cache_hits"]
            + self.performance_stats["cache_misses"]
        )
        if total_requests == 0:
            return 0.0
        return self.performance_stats["cache_hits"] / total_requests

# Global monitor instance
_global_monitor = None

def get_global_monitor() -> RealTimeMonitor:
    """Get the global monitor instance."""
    global _global_monitor
    if _global_monitor is None:
        _global_monitor = RealTimeMonitor()
    return _global_monitor
