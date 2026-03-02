"""
System health monitoring for Zaif Trade Bot.

This module provides comprehensive health checks for system resources,
dependencies, and trading bot components.
"""

import os
import sys
from dataclasses import dataclass
from typing import Any

import psutil

from ztb.utils.logging_utils import get_logger

logger = get_logger(__name__)

@dataclass
class HealthCheckResult:
    """Result of a health check."""

    name: str
    status: str  # "healthy", "warning", "critical"
    message: str
    details: dict[str, Any] | None = None

class SystemHealthChecker:
    """
    Comprehensive system health checker for trading bot operations.

    Checks system resources, dependencies, and trading-specific components.
    """

    def __init__(self) -> None:
        self.checks: list[HealthCheckResult] = []

    async def run_all_checks_async(self) -> list[HealthCheckResult]:
        """
        Run all health checks (async version).

        Returns:
            list of health check results
        """
        self.checks = []

        # System resource checks
        self._check_cpu_usage()
        self._check_memory_usage()
        self._check_disk_space()
        self._check_network_connectivity()

        # Python environment checks
        self._check_python_version()
        self._check_dependencies()

        # Trading-specific checks
        self._check_data_access()
        self._check_model_access()

        # Venue connectivity check (async) - skip if websockets not available
        try:
            await self._check_venue_connectivity_async()
        except Exception as e:
            logger.warning(f"Venue connectivity check failed: {e}")
            self.checks.append(
                HealthCheckResult(
                    name="venue_connectivity",
                    status="warning",
                    message="Venue connectivity check skipped due to dependency issues",
                    details={"error": str(e), "skipped": True},
                )
            )

        return self.checks

    def _check_cpu_usage(self) -> None:
        """Check CPU usage."""
        try:
            cpu_percent = psutil.cpu_percent(interval=1)
            if cpu_percent > 90:
                status = "critical"
                message = f"CPU usage is critically high: {cpu_percent}%"
            elif cpu_percent > 70:
                status = "warning"
                message = f"CPU usage is high: {cpu_percent}%"
            else:
                status = "healthy"
                message = f"CPU usage is normal: {cpu_percent}%"

            self.checks.append(
                HealthCheckResult(
                    name="cpu_usage",
                    status=status,
                    message=message,
                    details={"cpu_percent": cpu_percent},
                )
            )
        except Exception as e:
            self.checks.append(
                HealthCheckResult(
                    name="cpu_usage",
                    status="critical",
                    message=f"Failed to check CPU usage: {e}",
                    details={"error": str(e)},
                )
            )

    def _check_memory_usage(self) -> None:
        """Check memory usage."""
        try:
            memory = psutil.virtual_memory()
            memory_percent = memory.percent

            if memory_percent > 90:
                status = "critical"
                message = f"Memory usage is critically high: {memory_percent}%"
            elif memory_percent > 80:
                status = "warning"
                message = f"Memory usage is high: {memory_percent}%"
            else:
                status = "healthy"
                message = f"Memory usage is normal: {memory_percent}%"

            self.checks.append(
                HealthCheckResult(
                    name="memory_usage",
                    status=status,
                    message=message,
                    details={
                        "memory_percent": memory_percent,
                        "total_gb": memory.total / (1024**3),
                        "available_gb": memory.available / (1024**3),
                    },
                )
            )
        except Exception as e:
            self.checks.append(
                HealthCheckResult(
                    name="memory_usage",
                    status="critical",
                    message=f"Failed to check memory usage: {e}",
                    details={"error": str(e)},
                )
            )

    def _check_disk_space(self) -> None:
        """Check disk space for data and model directories."""
        try:
            # Check current directory disk space
            disk = psutil.disk_usage(".")
            disk_percent = disk.percent

            if disk_percent > 95:
                status = "critical"
                message = f"Disk space is critically low: {disk_percent}% used"
            elif disk_percent > 85:
                status = "warning"
                message = f"Disk space is low: {disk_percent}% used"
            else:
                status = "healthy"
                message = f"Disk space is adequate: {disk_percent}% used"

            self.checks.append(
                HealthCheckResult(
                    name="disk_space",
                    status=status,
                    message=message,
                    details={
                        "disk_percent": disk_percent,
                        "total_gb": disk.total / (1024**3),
                        "free_gb": disk.free / (1024**3),
                    },
                )
            )
        except Exception as e:
            self.checks.append(
                HealthCheckResult(
                    name="disk_space",
                    status="critical",
                    message=f"Failed to check disk space: {e}",
                    details={"error": str(e)},
                )
            )

    def _check_network_connectivity(self) -> None:
        """Check network connectivity."""
        try:
            import socket

            # Try to connect to a reliable host
            socket.create_connection(("8.8.8.8", 53), timeout=5)
            self.checks.append(
                HealthCheckResult(
                    name="network_connectivity",
                    status="healthy",
                    message="Network connectivity is available",
                    details={"connectivity": True},
                )
            )
        except Exception as e:
            self.checks.append(
                HealthCheckResult(
                    name="network_connectivity",
                    status="warning",
                    message=f"Network connectivity check failed: {e}",
                    details={"connectivity": False, "error": str(e)},
                )
            )

    def _check_python_version(self) -> None:
        """Check Python version compatibility."""
        try:
            version = sys.version_info
            min_version = (3, 11)

            if version >= min_version:
                status = "healthy"
                message = (
                    f"Python version {version.major}.{version.minor} is compatible"
                )
            else:
                status = "critical"
                message = f"Python version {version.major}.{version.minor} is too old. Minimum required: {min_version[0]}.{min_version[1]}"

            self.checks.append(
                HealthCheckResult(
                    name="python_version",
                    status=status,
                    message=message,
                    details={
                        "current_version": f"{version.major}.{version.minor}.{version.micro}",
                        "required_version": f"{min_version[0]}.{min_version[1]}",
                    },
                )
            )
        except Exception as e:
            self.checks.append(
                HealthCheckResult(
                    name="python_version",
                    status="critical",
                    message=f"Failed to check Python version: {e}",
                    details={"error": str(e)},
                )
            )

    def _check_dependencies(self) -> None:
        """Check critical dependencies."""
        critical_deps = [
            ("numpy", "Core numerical computing"),
            ("pandas", "Data manipulation"),
            ("psutil", "System monitoring"),
            ("pytest", "Testing framework"),
        ]

        for dep_name, description in critical_deps:
            try:
                __import__(dep_name)
                self.checks.append(
                    HealthCheckResult(
                        name=f"dependency_{dep_name}",
                        status="healthy",
                        message=f"{description} ({dep_name}) is available",
                        details={"available": True},
                    )
                )
            except ImportError:
                self.checks.append(
                    HealthCheckResult(
                        name=f"dependency_{dep_name}",
                        status="critical",
                        message=f"{description} ({dep_name}) is not available",
                        details={"available": False},
                    )
                )

    def _check_data_access(self) -> None:
        """Check data directory access."""
        try:
            data_paths = ["data", "config", "models"]
            missing_paths = []

            for path in data_paths:
                if not os.path.exists(path):
                    missing_paths.append(path)

            if missing_paths:
                self.checks.append(
                    HealthCheckResult(
                        name="data_access",
                        status="warning",
                        message=f"Some data directories are missing: {', '.join(missing_paths)}",
                        details={"missing_paths": missing_paths},
                    )
                )
            else:
                self.checks.append(
                    HealthCheckResult(
                        name="data_access",
                        status="healthy",
                        message="All data directories are accessible",
                        details={"all_paths_exist": True},
                    )
                )
        except Exception as e:
            self.checks.append(
                HealthCheckResult(
                    name="data_access",
                    status="warning",
                    message=f"Failed to check data access: {e}",
                    details={"error": str(e)},
                )
            )

    def _check_model_access(self) -> None:
        """Check model directory access and basic functionality."""
        try:
            if os.path.exists("models"):
                # Check if we can write to models directory
                test_file = os.path.join("models", ".health_check")
                try:
                    with open(test_file, "w") as f:
                        f.write("health_check")
                    os.remove(test_file)

                    self.checks.append(
                        HealthCheckResult(
                            name="model_access",
                            status="healthy",
                            message="Model directory is writable",
                            details={"writable": True},
                        )
                    )
                except Exception as e:
                    self.checks.append(
                        HealthCheckResult(
                            name="model_access",
                            status="warning",
                            message=f"Model directory is not writable: {e}",
                            details={"writable": False, "error": str(e)},
                        )
                    )
            else:
                self.checks.append(
                    HealthCheckResult(
                        name="model_access",
                        status="warning",
                        message="Model directory does not exist",
                        details={"exists": False},
                    )
                )
        except Exception as e:
            self.checks.append(
                HealthCheckResult(
                    name="model_access",
                    status="warning",
                    message=f"Failed to check model access: {e}",
                    details={"error": str(e)},
                )
            )

    async def _check_venue_connectivity_async(self) -> None:
        """Check trading venue connectivity (async version)."""
        try:
            # Check if websockets is available
            try:
                import websockets
            except ImportError:
                self.checks.append(
                    HealthCheckResult(
                        name="venue_connectivity",
                        status="warning",
                        message="Venue connectivity check requires websockets package",
                        details={
                            "available": False,
                            "missing_dependency": "websockets",
                        },
                    )
                )
                return

            # Try to import venue health checker
            try:
                from .check_venue_health import VenueHealthChecker
            except ImportError:
                self.checks.append(
                    HealthCheckResult(
                        name="venue_connectivity",
                        status="warning",
                        message="Venue health check module not available",
                        details={"available": False},
                    )
                )
                return

            # Check primary venue (coincheck)
            checker = VenueHealthChecker("coincheck", "btc_jpy", timeout=5)
            result = await checker.run_checks()

            # Determine status based on connectivity results
            connectivity = result.get("connectivity", {})
            api_available = connectivity.get("rest_api", False)
            ws_available = connectivity.get("websocket", False)

            if api_available or ws_available:
                status = "healthy" if api_available and ws_available else "warning"
                message = f"Primary venue (Coincheck) connectivity is {'healthy' if status == 'healthy' else 'degraded'}"
            else:
                status = "warning"  # Change from critical to warning for venue issues
                message = "Primary venue (Coincheck) connectivity check completed but services unavailable"

            self.checks.append(
                HealthCheckResult(
                    name="venue_connectivity",
                    status=status,
                    message=message,
                    details={
                        "venue": "coincheck",
                        "symbol": "btc_jpy",
                        "venue_status": result.get("status"),
                        "latency_ms": result.get("latency", {}).get("rest_ms"),
                        "api_available": api_available,
                        "ws_available": ws_available,
                        "internet_available": connectivity.get("internet", False),
                        "errors": result.get("errors", []),
                    },
                )
            )
        except Exception as e:
            self.checks.append(
                HealthCheckResult(
                    name="venue_connectivity",
                    status="warning",
                    message=f"Venue connectivity check failed: {e}",
                    details={"error": str(e), "skipped": True},
                )
            )

    def get_summary(self) -> dict[str, Any]:
        """
        Get a summary of all health checks.

        Returns:
            Summary dictionary with counts and overall status
        """
        if not self.checks:
            return {"status": "unknown", "message": "No checks performed"}

        status_counts = {"healthy": 0, "warning": 0, "critical": 0}

        for check in self.checks:
            status_counts[check.status] = status_counts.get(check.status, 0) + 1

        # Determine overall status
        if status_counts["critical"] > 0:
            overall_status = "critical"
        elif status_counts["warning"] > 0:
            overall_status = "warning"
        else:
            overall_status = "healthy"

        return {
            "status": overall_status,
            "total_checks": len(self.checks),
            "healthy": status_counts["healthy"],
            "warning": status_counts["warning"],
            "critical": status_counts["critical"],
            "checks": [check.__dict__ for check in self.checks],
        }

async def run_health_check_async() -> dict[str, Any]:
    """
    Run a complete health check and return results (async version).

    Returns:
        Health check summary
    """
    checker = SystemHealthChecker()
    await checker.run_all_checks_async()

    # Get performance monitoring results
    from .performance_monitor import run_performance_check

    performance_report = run_performance_check()

    summary = checker.get_summary()
    summary["performance"] = performance_report

    return summary

def run_health_check() -> dict[str, Any]:
    """
    Run a complete health check and return results.

    Returns:
        Health check summary
    """
    import asyncio

    return asyncio.run(run_health_check_async())

if __name__ == "__main__":
    # Command-line interface
    import json

    result = run_health_check()
    print(json.dumps(result, indent=2, ensure_ascii=False))
