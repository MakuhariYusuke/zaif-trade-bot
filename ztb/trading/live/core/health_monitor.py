"""
Health monitoring for live trading bot.
"""
import logging
from typing import Any, Dict, Optional
import gc
import time

logger = logging.getLogger(__name__)


class HealthMonitor:
    """Monitors system health and provides status information."""

    def __init__(self) -> None:
        self._cleanup_counter = 0
        self._cleanup_interval = 100
        self._last_health_check = time.time()

    def get_health_status(self) -> Dict[str, Any]:
        """Get comprehensive health status."""
        import psutil  # type: ignore[import-untyped]

        try:
            # System resources
            cpu_percent = psutil.cpu_percent(interval=1)
            memory = psutil.virtual_memory()
            disk = psutil.disk_usage('/')

            # Process info
            process = psutil.Process()
            process_memory = process.memory_info()
            process_cpu = process.cpu_percent()

            return {
                "status": "healthy",
                "timestamp": time.time(),
                "system": {
                    "cpu_percent": cpu_percent,
                    "memory_percent": memory.percent,
                    "memory_used_gb": memory.used / (1024**3),
                    "memory_total_gb": memory.total / (1024**3),
                    "disk_percent": disk.percent,
                    "disk_free_gb": disk.free / (1024**3),
                },
                "process": {
                    "cpu_percent": process_cpu,
                    "memory_rss_mb": process_memory.rss / (1024**2),
                    "memory_vms_mb": process_memory.vms / (1024**2),
                    "threads": process.num_threads(),
                },
                "uptime": time.time() - self._last_health_check,
            }
        except Exception as e:
            logger.warning(f"Failed to get health status: {e}")
            return {
                "status": "degraded",
                "error": str(e),
                "timestamp": time.time(),
            }

    def periodic_cleanup(self) -> None:
        """Perform periodic memory cleanup."""
        self._cleanup_counter += 1
        if self._cleanup_counter >= self._cleanup_interval:
            # Force garbage collection periodically
            gc.collect()
            self._cleanup_counter = 0
            logger.debug("Periodic cleanup completed")

    def setup_metrics(self) -> Optional[Dict[str, Any]]:
        """Setup Prometheus metrics if available."""
        try:
            from prometheus_client import Counter, Gauge, Histogram  # type: ignore[import-untyped]

            metrics = {
                "price_fetches": Counter("price_fetches_total", "Total price fetches", ["success"]),
                "price_fetch_duration": Histogram("price_fetch_duration_seconds", "Price fetch duration"),
                "price_current": Gauge("price_current", "Current price"),
                "trades_executed": Counter("trades_executed_total", "Total trades executed", ["action"]),
                "pnl_current": Gauge("pnl_current", "Current PnL"),
                "position_size": Gauge("position_size", "Current position size"),
            }

            logger.info("Prometheus metrics initialized")
            return metrics
        except ImportError:
            logger.info("Prometheus not available, metrics disabled")
            return None

    def update_metrics(
        self,
        metrics: Optional[Dict[str, Any]],
        price: float,
        pnl: float,
        position: float,
        trade_success: bool = False,
        action: Optional[str] = None
    ) -> None:
        """Update metrics with current values."""
        if not metrics:
            return

        try:
            metrics["price_current"].set(price)
            metrics["pnl_current"].set(pnl)
            metrics["position_size"].set(position)

            if trade_success and action:
                metrics["trades_executed"].labels(action=action).inc()
        except Exception as e:
            logger.warning(f"Failed to update metrics: {e}")