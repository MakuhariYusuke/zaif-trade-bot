"""
Health monitoring utilities for ZTB system.

Provides comprehensive health checks for system components,
performance monitoring, and alerting capabilities.
"""

import logging
import time
from dataclasses import dataclass
from enum import Enum
from typing import Any, Callable, Dict, Optional

import psutil

from ztb.utils.circuit_breaker import CircuitBreaker, CircuitBreakerConfig, CircuitState
from ztb.utils.config import ZTBConfig
from ztb.utils.memory_monitor import MemoryMonitor

logger = logging.getLogger(__name__)


class HealthStatus(Enum):
    """Health status levels."""

    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    UNKNOWN = "unknown"


@dataclass
class HealthCheckResult:
    """Result of a health check."""

    name: str
    status: HealthStatus
    message: str
    details: Dict[str, Any]
    timestamp: float
    duration: float


@dataclass
class SystemMetrics:
    """System performance metrics."""

    cpu_percent: float
    memory_percent: float
    memory_mb: float
    disk_usage_percent: float
    network_connections: int
    timestamp: float


class HealthChecker:
    """Health checker for system components."""

    def __init__(self, config: Optional[ZTBConfig] = None):
        self.config = config or ZTBConfig()
        self.checks: Dict[str, Callable[[], HealthCheckResult]] = {}
        self.circuit_breakers: Dict[str, CircuitBreaker] = {}
        self.last_metrics: Optional[SystemMetrics] = None
        self.memory_monitor = MemoryMonitor(self.config)

    def register_check(
        self, name: str, check_func: Callable[[], HealthCheckResult]
    ) -> None:
        """
        Register a health check function.

        Args:
            name: Name of the health check
            check_func: Function that performs the health check
        """
        self.checks[name] = check_func

        # Create circuit breaker for this check
        cb_config = CircuitBreakerConfig(
            failure_threshold=self.config.get_int("ZTB_HEALTH_FAILURE_THRESHOLD", 3),
            recovery_timeout=self.config.get_float("ZTB_HEALTH_RECOVERY_TIMEOUT", 30.0),
            success_threshold=self.config.get_int("ZTB_HEALTH_SUCCESS_THRESHOLD", 2),
            timeout=self.config.get_float("ZTB_HEALTH_CHECK_TIMEOUT", 5.0),
        )
        self.circuit_breakers[name] = CircuitBreaker(name, cb_config)

    def unregister_check(self, name: str) -> None:
        """Unregister a health check."""
        self.checks.pop(name, None)
        self.circuit_breakers.pop(name, None)

    def run_check(self, name: str) -> HealthCheckResult:
        """
        Run a specific health check with circuit breaker protection.

        Args:
            name: Name of the health check to run

        Returns:
            Health check result
        """
        if name not in self.checks:
            return HealthCheckResult(
                name=name,
                status=HealthStatus.UNKNOWN,
                message=f"Health check '{name}' not registered",
                details={},
                timestamp=time.time(),
                duration=0.0,
            )

        circuit_breaker = self.circuit_breakers[name]

        try:
            # Check if circuit breaker allows the call
            if circuit_breaker.state == CircuitState.OPEN:
                return HealthCheckResult(
                    name=name,
                    status=HealthStatus.UNHEALTHY,
                    message=f"Circuit breaker is OPEN for '{name}'",
                    details={"circuit_state": "open"},
                    timestamp=time.time(),
                    duration=0.0,
                )

            # Run the health check
            start_time = time.time()
            result = self.checks[name]()
            duration = time.time() - start_time

            # Update circuit breaker
            if result.status == HealthStatus.HEALTHY:
                circuit_breaker.record_success()
            else:
                circuit_breaker.record_failure()

            # Update result with timing
            result.duration = duration
            return result

        except Exception as e:
            logger.error(f"Health check '{name}' failed with exception: {e}")
            circuit_breaker.record_failure()
            return HealthCheckResult(
                name=name,
                status=HealthStatus.UNHEALTHY,
                message=f"Health check failed: {str(e)}",
                details={"exception": str(e)},
                timestamp=time.time(),
                duration=time.time() - time.time(),  # Approximate
            )

    def run_all_checks(self) -> Dict[str, HealthCheckResult]:
        """
        Run all registered health checks.

        Returns:
            Dictionary of health check results
        """
        results = {}
        for name in self.checks:
            results[name] = self.run_check(name)
        return results

    def get_overall_health(self) -> HealthStatus:
        """
        Get overall system health status.

        Returns:
            Overall health status
        """
        results = self.run_all_checks()

        if not results:
            return HealthStatus.UNKNOWN

        # Count statuses
        status_counts = {}
        for result in results.values():
            status_counts[result.status] = status_counts.get(result.status, 0) + 1

        # Determine overall status
        if status_counts.get(HealthStatus.UNHEALTHY, 0) > 0:
            return HealthStatus.UNHEALTHY
        elif status_counts.get(HealthStatus.DEGRADED, 0) > 0:
            return HealthStatus.DEGRADED
        elif status_counts.get(HealthStatus.HEALTHY, 0) == len(results):
            return HealthStatus.HEALTHY
        else:
            return HealthStatus.UNKNOWN

    def collect_system_metrics(self) -> SystemMetrics:
        """
        Collect current system performance metrics.

        Returns:
            System metrics
        """
        try:
            cpu_percent = psutil.cpu_percent(interval=1)
            memory = psutil.virtual_memory()
            disk = psutil.disk_usage("/")
            network = len(psutil.net_connections())

            metrics = SystemMetrics(
                cpu_percent=cpu_percent,
                memory_percent=memory.percent,
                memory_mb=memory.used / 1024 / 1024,
                disk_usage_percent=disk.percent,
                network_connections=network,
                timestamp=time.time(),
            )

            self.last_metrics = metrics
            return metrics

        except Exception as e:
            logger.error(f"Failed to collect system metrics: {e}")
            # Return dummy metrics
            return SystemMetrics(
                cpu_percent=0.0,
                memory_percent=0.0,
                memory_mb=0.0,
                disk_usage_percent=0.0,
                network_connections=0,
                timestamp=time.time(),
            )

    def check_system_health(self) -> HealthCheckResult:
        """
        Check system-level health (CPU, memory, disk).

        Returns:
            Health check result
        """
        start_time = time.time()
        metrics = self.collect_system_metrics()

        issues = []

        # Check CPU usage
        cpu_threshold = self.config.get_float("ZTB_HEALTH_CPU_THRESHOLD", 90.0)
        if metrics.cpu_percent > cpu_threshold:
            issues.append(f"High CPU usage: {metrics.cpu_percent:.1f}%")

        # Check memory usage
        memory_threshold = self.config.get_float("ZTB_HEALTH_MEMORY_THRESHOLD", 90.0)
        if metrics.memory_percent > memory_threshold:
            issues.append(f"High memory usage: {metrics.memory_percent:.1f}%")

        # Check disk usage
        disk_threshold = self.config.get_float("ZTB_HEALTH_DISK_THRESHOLD", 95.0)
        if metrics.disk_usage_percent > disk_threshold:
            issues.append(f"High disk usage: {metrics.disk_usage_percent:.1f}%")

        # Determine status
        if issues:
            status = (
                HealthStatus.DEGRADED if len(issues) == 1 else HealthStatus.UNHEALTHY
            )
            message = f"System health issues: {', '.join(issues)}"
        else:
            status = HealthStatus.HEALTHY
            message = "System operating normally"

        return HealthCheckResult(
            name="system_health",
            status=status,
            message=message,
            details={
                "cpu_percent": metrics.cpu_percent,
                "memory_percent": metrics.memory_percent,
                "memory_mb": metrics.memory_mb,
                "disk_usage_percent": metrics.disk_usage_percent,
                "network_connections": metrics.network_connections,
                "issues": issues,
            },
            timestamp=metrics.timestamp,
            duration=time.time() - start_time,
        )

    def check_database_connectivity(self) -> HealthCheckResult:
        """
        Check database connectivity (placeholder for actual implementation).

        Returns:
            Health check result
        """
        start_time = time.time()

        # Placeholder - in real implementation, this would test actual DB connection
        try:
            # Simulate database check
            time.sleep(0.1)  # Simulate network latency
            status = HealthStatus.HEALTHY
            message = "Database connection healthy"
            details = {"connection_time_ms": 100}
        except Exception as e:
            status = HealthStatus.UNHEALTHY
            message = f"Database connection failed: {str(e)}"
            details = {"error": str(e)}

        return HealthCheckResult(
            name="database_connectivity",
            status=status,
            message=message,
            details=details,
            timestamp=time.time(),
            duration=time.time() - start_time,
        )

    def check_external_api_health(self) -> HealthCheckResult:
        """
        Check external API health (placeholder for actual implementation).

        Returns:
            Health check result
        """
        start_time = time.time()

        # Placeholder - in real implementation, this would test actual API endpoints
        try:
            # Simulate API check
            time.sleep(0.05)  # Simulate network latency
            status = HealthStatus.HEALTHY
            message = "External APIs responding normally"
            details = {
                "response_time_ms": 50,
                "apis_checked": ["exchange_api", "market_data_api"],
            }
        except Exception as e:
            status = HealthStatus.UNHEALTHY
            message = f"External API check failed: {str(e)}"
            details = {"error": str(e)}

        return HealthCheckResult(
            name="external_api_health",
            status=status,
            message=message,
            details=details,
            timestamp=time.time(),
            duration=time.time() - start_time,
        )

    def check_memory_health(self) -> HealthCheckResult:
        """
        Check memory usage health.

        Returns:
            Health check result
        """
        start_time = time.time()
        memory_stats = self.memory_monitor.get_memory_stats()

        issues = []
        current_mb = memory_stats["current_mb"]

        # Check memory thresholds
        critical_threshold = self.config.get_int(
            "ZTB_MEMORY_CRITICAL_THRESHOLD_MB", 2000
        )
        warning_threshold = self.config.get_int("ZTB_MEMORY_WARNING_THRESHOLD_MB", 1000)

        if current_mb > critical_threshold:
            issues.append(
                f"Critical memory usage: {current_mb:.1f}MB > {critical_threshold}MB"
            )
        elif current_mb > warning_threshold:
            issues.append(
                f"High memory usage: {current_mb:.1f}MB > {warning_threshold}MB"
            )

        # Check memory trend
        trend = self.memory_monitor.get_memory_trend()
        if trend == "increasing":
            issues.append("Memory usage is increasing")

        # Determine status
        if any("Critical" in issue for issue in issues):
            status = HealthStatus.UNHEALTHY
        elif issues:
            status = HealthStatus.DEGRADED
        else:
            status = HealthStatus.HEALTHY

        message = (
            "Memory usage normal"
            if not issues
            else f"Memory issues: {', '.join(issues)}"
        )

        return HealthCheckResult(
            name="memory_health",
            status=status,
            message=message,
            details={
                "current_mb": memory_stats["current_mb"],
                "average_mb": memory_stats["average_mb"],
                "peak_mb": memory_stats["peak_mb"],
                "samples": memory_stats["samples"],
                "trend": trend,
                "issues": issues,
            },
            timestamp=time.time(),
            duration=time.time() - start_time,
        )

    def setup_default_checks(self) -> None:
        """Set up default health checks."""
        self.register_check("system_health", self.check_system_health)
        self.register_check("memory_health", self.check_memory_health)
        self.register_check("database_connectivity", self.check_database_connectivity)
        self.register_check("external_api_health", self.check_external_api_health)

    def get_health_summary(self) -> Dict[str, Any]:
        """
        Get a summary of all health checks and system status.

        Returns:
            Health summary dictionary
        """
        results = self.run_all_checks()
        overall_status = self.get_overall_health()
        metrics = self.collect_system_metrics()
        memory_stats = self.memory_monitor.get_memory_stats()

        return {
            "overall_status": overall_status.value,
            "timestamp": time.time(),
            "checks": {
                name: {
                    "status": result.status.value,
                    "message": result.message,
                    "duration": result.duration,
                    "details": result.details,
                }
                for name, result in results.items()
            },
            "system_metrics": {
                "cpu_percent": metrics.cpu_percent,
                "memory_percent": metrics.memory_percent,
                "memory_mb": metrics.memory_mb,
                "disk_usage_percent": metrics.disk_usage_percent,
                "network_connections": metrics.network_connections,
            },
            "memory_stats": {
                "current_mb": memory_stats["current_mb"],
                "average_mb": memory_stats["average_mb"],
                "peak_mb": memory_stats["peak_mb"],
                "samples": memory_stats["samples"],
                "trend": self.memory_monitor.get_memory_trend(),
            },
            "circuit_breakers": {
                name: cb.state.value for name, cb in self.circuit_breakers.items()
            },
        }
