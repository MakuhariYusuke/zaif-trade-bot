"""
Health monitoring for the 24/7 trading service.
"""

import logging
import time
from pathlib import Path
from typing import Any, Dict

import psutil

from ztb.utils.errors import safe_operation


class HealthMonitor:
    """Monitor the health of the trading service."""

    def __init__(self, service_name: str = "trading_service"):
        self.service_name = service_name
        self.logger = logging.getLogger(__name__)
        self.start_time = time.time()
        self.last_health_check = 0.0
        self.health_check_interval = 60  # seconds

    def check_overall_health(self) -> Dict[str, Any]:
        """
        Perform comprehensive health check.

        Returns:
            Dict containing health status and metrics
        """
        return safe_operation(
            logger=self.logger,
            operation=self._check_overall_health_impl,
            context="health_check",
            default_result={
                "timestamp": time.time(),
                "service": self.service_name,
                "status": "unhealthy",
                "error": "Health check failed",
            },
        )

    def _check_overall_health_impl(self) -> Dict[str, Any]:
        """Implementation of overall health check."""
        health_status: Dict[str, Any] = {
            "timestamp": time.time(),
            "service": self.service_name,
            "status": "healthy",
            "checks": {},
            "metrics": {},
        }

        # System resource checks
        health_status["checks"]["memory"] = self._check_memory_usage()
        health_status["checks"]["cpu"] = self._check_cpu_usage()
        health_status["checks"]["disk"] = self._check_disk_space()
        health_status["checks"]["uptime"] = self._check_uptime()

        # Application-specific checks
        health_status["checks"]["logs"] = self._check_log_files()
        health_status["checks"]["config"] = self._check_configuration()

        # Aggregate status
        failed_checks = [
            k
            for k, v in health_status["checks"].items()
            if not v.get("healthy", False)
        ]
        if failed_checks:
            health_status["status"] = "degraded"
            health_status["failed_checks"] = failed_checks

        # Performance metrics
        health_status["metrics"] = self._collect_metrics()

        self.last_health_check = time.time()
        return health_status

    def _check_memory_usage(self) -> Dict[str, Any]:
        """Check memory usage."""
        try:
            process = psutil.Process()
            memory_info = process.memory_info()
            memory_percent = process.memory_percent()

            return {
                "healthy": memory_percent < 80,  # Less than 80% memory usage
                "used_mb": memory_info.rss / 1024 / 1024,
                "used_percent": memory_percent,
                "limit_mb": 1024,  # 1GB limit
            }
        except Exception as e:
            return {"healthy": False, "error": str(e)}

    def _check_cpu_usage(self) -> Dict[str, Any]:
        """Check CPU usage."""
        try:
            process = psutil.Process()
            cpu_percent = process.cpu_percent(interval=1)

            return {
                "healthy": cpu_percent < 70,  # Less than 70% CPU usage
                "used_percent": cpu_percent,
            }
        except Exception as e:
            return {"healthy": False, "error": str(e)}

    def _check_disk_space(self) -> Dict[str, Any]:
        """Check disk space availability."""
        try:
            disk_usage = psutil.disk_usage("/")
            free_gb = disk_usage.free / 1024 / 1024 / 1024

            return {
                "healthy": free_gb > 1,  # At least 1GB free
                "free_gb": free_gb,
                "total_gb": disk_usage.total / 1024 / 1024 / 1024,
            }
        except Exception as e:
            return {"healthy": False, "error": str(e)}

    def _check_uptime(self) -> Dict[str, Any]:
        """Check service uptime."""
        try:
            uptime_seconds = time.time() - self.start_time
            uptime_hours = uptime_seconds / 3600

            return {
                "healthy": True,
                "uptime_seconds": uptime_seconds,
                "uptime_hours": uptime_hours,
            }
        except Exception as e:
            return {"healthy": False, "error": str(e)}

    def _check_log_files(self) -> Dict[str, Any]:
        """Check log file health."""
        try:
            log_file = Path("trading_service.log")
            if log_file.exists():
                size_mb = log_file.stat().st_size / 1024 / 1024
                # Check if log file is not too large (>100MB)
                healthy = size_mb < 100
                return {"healthy": healthy, "size_mb": size_mb, "exists": True}
            else:
                return {
                    "healthy": False,
                    "error": "Log file does not exist",
                    "exists": False,
                }
        except Exception as e:
            return {"healthy": False, "error": str(e)}

    def _check_configuration(self) -> Dict[str, Any]:
        """Check configuration file accessibility."""
        try:
            config_files = [
                "config/production.yaml",
                "config/default.yaml",
                "trade-config.json",
            ]
            existing_configs = []

            for config_file in config_files:
                if Path(config_file).exists():
                    existing_configs.append(config_file)

            return {
                "healthy": len(existing_configs) > 0,
                "config_files": existing_configs,
            }
        except Exception as e:
            return {"healthy": False, "error": str(e)}

    def _collect_metrics(self) -> Dict[str, Any]:
        """Collect performance metrics."""
        try:
            process = psutil.Process()
            return {
                "process_id": process.pid,
                "threads": process.num_threads(),
                "open_files": len(process.open_files()),
                "connections": len(process.connections()),
            }
        except Exception as e:
            return {"error": str(e)}

    def should_restart(self, health_status: Dict[str, Any]) -> bool:
        """
        Determine if service should restart based on health status.

        Args:
            health_status: Health check results

        Returns:
            True if restart is recommended
        """
        if health_status.get("status") == "unhealthy":
            return True

        # Check for critical failures
        checks = health_status.get("checks", {})
        critical_checks = ["memory", "disk"]

        for check_name in critical_checks:
            check_result = checks.get(check_name, {})
            if not check_result.get("healthy", True):
                self.logger.warning(f"Critical health check failed: {check_name}")
                return True

        return False
