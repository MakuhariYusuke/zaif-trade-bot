"""
Unit tests for health_monitor.py
"""

import time
import unittest
from unittest.mock import Mock, patch

from ztb.utils.config import ZTBConfig
from ztb.utils.health_monitor import (
    HealthChecker,
    HealthCheckResult,
    HealthStatus,
    SystemMetrics,
)


class TestHealthMonitor(unittest.TestCase):
    """Test cases for health monitoring functionality."""

    def setUp(self):
        """Set up test fixtures."""
        self.config = ZTBConfig()
        self.health_checker = HealthChecker(self.config)

    def test_health_status_enum(self):
        """Test HealthStatus enum values."""
        self.assertEqual(HealthStatus.HEALTHY.value, "healthy")
        self.assertEqual(HealthStatus.DEGRADED.value, "degraded")
        self.assertEqual(HealthStatus.UNHEALTHY.value, "unhealthy")
        self.assertEqual(HealthStatus.UNKNOWN.value, "unknown")

    def test_health_check_result_creation(self):
        """Test HealthCheckResult creation."""
        result = HealthCheckResult(
            name="test_check",
            status=HealthStatus.HEALTHY,
            message="Test passed",
            details={"key": "value"},
            timestamp=1234567890.0,
            duration=1.5,
        )

        self.assertEqual(result.name, "test_check")
        self.assertEqual(result.status, HealthStatus.HEALTHY)
        self.assertEqual(result.message, "Test passed")
        self.assertEqual(result.details, {"key": "value"})
        self.assertEqual(result.timestamp, 1234567890.0)
        self.assertEqual(result.duration, 1.5)

    @patch("psutil.cpu_percent")
    @patch("psutil.virtual_memory")
    @patch("psutil.disk_usage")
    @patch("psutil.net_connections")
    def test_collect_system_metrics(self, mock_net, mock_disk, mock_memory, mock_cpu):
        """Test system metrics collection."""
        # Mock system calls
        mock_cpu.return_value = 45.5
        memory_mock = Mock()
        memory_mock.percent = 67.8
        memory_mock.used = 1024 * 1024 * 1024  # 1GB in bytes
        mock_memory.return_value = memory_mock

        disk_mock = Mock()
        disk_mock.percent = 23.4
        mock_disk.return_value = disk_mock

        mock_net.return_value = [Mock()] * 5  # 5 connections

        metrics = self.health_checker.collect_system_metrics()

        self.assertIsInstance(metrics, SystemMetrics)
        self.assertEqual(metrics.cpu_percent, 45.5)
        self.assertEqual(metrics.memory_percent, 67.8)
        self.assertAlmostEqual(metrics.memory_mb, 1024.0, places=1)  # ~1GB
        self.assertEqual(metrics.disk_usage_percent, 23.4)
        self.assertEqual(metrics.network_connections, 5)
        self.assertIsInstance(metrics.timestamp, float)

    def test_register_and_run_check(self):
        """Test registering and running health checks."""

        def mock_check():
            return HealthCheckResult(
                name="mock_check",
                status=HealthStatus.HEALTHY,
                message="Mock check passed",
                details={},
                timestamp=time.time(),
                duration=0.1,
            )

        self.health_checker.register_check("mock_check", mock_check)
        result = self.health_checker.run_check("mock_check")

        self.assertEqual(result.name, "mock_check")
        self.assertEqual(result.status, HealthStatus.HEALTHY)
        self.assertEqual(result.message, "Mock check passed")

    def test_run_check_with_exception(self):
        """Test health check that raises an exception."""

        def failing_check():
            raise ValueError("Test error")

        self.health_checker.register_check("failing_check", failing_check)
        result = self.health_checker.run_check("failing_check")

        self.assertEqual(result.name, "failing_check")
        self.assertEqual(result.status, HealthStatus.UNHEALTHY)
        self.assertIn("Test error", result.message)

    def test_run_unknown_check(self):
        """Test running a check that doesn't exist."""
        result = self.health_checker.run_check("unknown_check")

        self.assertEqual(result.name, "unknown_check")
        self.assertEqual(result.status, HealthStatus.UNKNOWN)
        self.assertIn("not registered", result.message)

    def test_overall_health_calculation(self):
        """Test overall health status calculation."""

        # Register some checks
        def healthy_check():
            return HealthCheckResult(
                name="healthy",
                status=HealthStatus.HEALTHY,
                message="OK",
                details={},
                timestamp=time.time(),
                duration=0.1,
            )

        def degraded_check():
            return HealthCheckResult(
                name="degraded",
                status=HealthStatus.DEGRADED,
                message="Warning",
                details={},
                timestamp=time.time(),
                duration=0.1,
            )

        def unhealthy_check():
            return HealthCheckResult(
                name="unhealthy",
                status=HealthStatus.UNHEALTHY,
                message="Error",
                details={},
                timestamp=time.time(),
                duration=0.1,
            )

        self.health_checker.register_check("healthy", healthy_check)
        self.health_checker.register_check("degraded", degraded_check)
        self.health_checker.register_check("unhealthy", unhealthy_check)

        # Test overall health
        overall = self.health_checker.get_overall_health()
        self.assertEqual(overall, HealthStatus.UNHEALTHY)  # Unhealthy takes precedence

        # Remove unhealthy check
        self.health_checker.unregister_check("unhealthy")
        overall = self.health_checker.get_overall_health()
        self.assertEqual(overall, HealthStatus.DEGRADED)  # Degraded is next

        # Remove degraded check
        self.health_checker.unregister_check("degraded")
        overall = self.health_checker.get_overall_health()
        self.assertEqual(overall, HealthStatus.HEALTHY)  # All healthy

    def test_setup_default_checks(self):
        """Test setting up default health checks."""
        self.health_checker.setup_default_checks()

        # Check that default checks are registered
        expected_checks = [
            "system_health",
            "memory_health",
            "database_connectivity",
            "external_api_health",
        ]

        for check_name in expected_checks:
            self.assertIn(check_name, self.health_checker.checks)

    @patch("psutil.cpu_percent")
    @patch("psutil.virtual_memory")
    @patch("psutil.disk_usage")
    def test_system_health_check(self, mock_disk, mock_memory, mock_cpu):
        """Test system health check."""
        # Mock normal system state
        mock_cpu.return_value = 50.0
        mock_memory.return_value = Mock(percent=60.0)
        mock_disk.return_value = Mock(percent=40.0)

        result = self.health_checker.check_system_health()

        self.assertEqual(result.name, "system_health")
        self.assertEqual(result.status, HealthStatus.HEALTHY)
        self.assertIn("operating normally", result.message)

    @patch("psutil.cpu_percent")
    @patch("psutil.virtual_memory")
    @patch("psutil.disk_usage")
    def test_system_health_check_high_usage(self, mock_disk, mock_memory, mock_cpu):
        """Test system health check with high resource usage."""
        # Mock high usage
        mock_cpu.return_value = 95.0
        memory_mock = Mock()
        memory_mock.percent = 95.0
        memory_mock.used = 2048 * 1024 * 1024  # 2GB in bytes
        mock_memory.return_value = memory_mock

        disk_mock = Mock()
        disk_mock.percent = 96.0
        mock_disk.return_value = disk_mock

        result = self.health_checker.check_system_health()

        self.assertEqual(result.name, "system_health")
        self.assertEqual(result.status, HealthStatus.UNHEALTHY)
        self.assertIn("High CPU usage", result.message)
        self.assertIn("High memory usage", result.message)
        self.assertIn("High disk usage", result.message)

    def test_memory_health_check(self):
        """Test memory health check."""
        # Start memory monitoring
        self.health_checker.memory_monitor.start_monitoring(interval_seconds=0.1)

        # Record some memory usage
        for _ in range(5):
            self.health_checker.memory_monitor.record_memory_usage()
            time.sleep(0.01)

        result = self.health_checker.check_memory_health()

        self.assertEqual(result.name, "memory_health")
        self.assertIn(result.status, [HealthStatus.HEALTHY, HealthStatus.DEGRADED])
        self.assertIn("current_mb", result.details)
        self.assertIn("trend", result.details)

        # Stop monitoring
        self.health_checker.memory_monitor.stop_monitoring()

    def test_get_health_summary(self):
        """Test getting comprehensive health summary."""
        self.health_checker.setup_default_checks()

        summary = self.health_checker.get_health_summary()

        # Check structure
        self.assertIn("overall_status", summary)
        self.assertIn("timestamp", summary)
        self.assertIn("checks", summary)
        self.assertIn("system_metrics", summary)
        self.assertIn("memory_stats", summary)
        self.assertIn("circuit_breakers", summary)

        # Check that checks were run
        self.assertIn("system_health", summary["checks"])
        self.assertIn("memory_health", summary["checks"])


if __name__ == "__main__":
    unittest.main()
