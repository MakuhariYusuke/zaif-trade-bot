"""
Tests for system health monitoring functionality.
"""

import unittest
from unittest.mock import AsyncMock, MagicMock, patch

from ztb.ops.health.system_health import (
    HealthCheckResult,
    SystemHealthChecker,
    run_health_check_async,
)


def _fake_health_check(name: str, status: str = "healthy") -> HealthCheckResult:
    return HealthCheckResult(
        name=name,
        status=status,
        message=f"{name} check {status}",
        details={"stubbed": True},
    )


def _install_stubbed_checks(checker: SystemHealthChecker) -> None:
    checker._check_cpu_usage = MagicMock(  # type: ignore[method-assign]
        side_effect=lambda: checker.checks.append(_fake_health_check("cpu_usage"))
    )
    checker._check_memory_usage = MagicMock(  # type: ignore[method-assign]
        side_effect=lambda: checker.checks.append(_fake_health_check("memory_usage"))
    )
    checker._check_disk_space = MagicMock(  # type: ignore[method-assign]
        side_effect=lambda: checker.checks.append(_fake_health_check("disk_space"))
    )
    checker._check_network_connectivity = MagicMock(  # type: ignore[method-assign]
        side_effect=lambda: checker.checks.append(
            _fake_health_check("network_connectivity")
        )
    )
    checker._check_python_version = MagicMock(  # type: ignore[method-assign]
        side_effect=lambda: checker.checks.append(_fake_health_check("python_version"))
    )
    checker._check_dependencies = MagicMock(  # type: ignore[method-assign]
        side_effect=lambda: checker.checks.append(_fake_health_check("dependency_numpy"))
    )
    checker._check_data_access = MagicMock(  # type: ignore[method-assign]
        side_effect=lambda: checker.checks.append(_fake_health_check("data_access"))
    )
    checker._check_model_access = MagicMock(  # type: ignore[method-assign]
        side_effect=lambda: checker.checks.append(_fake_health_check("model_access"))
    )

    async def _fake_venue_check() -> None:
        checker.checks.append(_fake_health_check("venue_connectivity"))

    checker._check_venue_connectivity_async = AsyncMock(  # type: ignore[method-assign]
        side_effect=_fake_venue_check
    )


class TestHealthCheckResult(unittest.TestCase):
    """Test HealthCheckResult dataclass."""

    def test_health_check_result_creation(self):
        """Test creating a HealthCheckResult."""
        result = HealthCheckResult(
            name="test_check",
            status="healthy",
            message="Test passed",
            details={"value": 100},
        )

        self.assertEqual(result.name, "test_check")
        self.assertEqual(result.status, "healthy")
        self.assertEqual(result.message, "Test passed")
        self.assertEqual(result.details, {"value": 100})


class TestSystemHealthChecker(unittest.IsolatedAsyncioTestCase):
    """Test SystemHealthChecker class."""

    def setUp(self):
        """Set up test fixtures."""
        self.checker = SystemHealthChecker()

    async def test_run_all_checks_populates_checks_list(self):
        """Test that run_all_checks populates the checks list."""
        _install_stubbed_checks(self.checker)
        initial_length = len(self.checker.checks)
        await self.checker.run_all_checks_async()

        # Should have more checks after running
        self.assertGreater(len(self.checker.checks), initial_length)

    async def test_get_summary_returns_dict(self):
        """Test that get_summary returns a dictionary."""
        _install_stubbed_checks(self.checker)
        await self.checker.run_all_checks_async()
        summary = self.checker.get_summary()

        self.assertIsInstance(summary, dict)
        self.assertIn("status", summary)
        self.assertIn("total_checks", summary)
        self.assertIn("healthy", summary)
        self.assertIn("warning", summary)
        self.assertIn("critical", summary)
        self.assertIn("checks", summary)

    @patch("psutil.cpu_percent")
    def test_cpu_usage_check_normal(self, mock_cpu_percent):
        """Test CPU usage check with normal usage."""
        mock_cpu_percent.return_value = 50.0

        self.checker._check_cpu_usage()

        self.assertEqual(len(self.checker.checks), 1)
        check = self.checker.checks[0]
        self.assertEqual(check.name, "cpu_usage")
        self.assertEqual(check.status, "healthy")
        self.assertIn("normal", check.message)

    @patch("psutil.cpu_percent")
    def test_cpu_usage_check_high(self, mock_cpu_percent):
        """Test CPU usage check with high usage."""
        mock_cpu_percent.return_value = 85.0

        self.checker._check_cpu_usage()

        self.assertEqual(len(self.checker.checks), 1)
        check = self.checker.checks[0]
        self.assertEqual(check.name, "cpu_usage")
        self.assertEqual(check.status, "warning")
        self.assertIn("high", check.message)

    @patch("psutil.cpu_percent")
    def test_cpu_usage_check_critical(self, mock_cpu_percent):
        """Test CPU usage check with critical usage."""
        mock_cpu_percent.return_value = 95.0

        self.checker._check_cpu_usage()

        self.assertEqual(len(self.checker.checks), 1)
        check = self.checker.checks[0]
        self.assertEqual(check.name, "cpu_usage")
        self.assertEqual(check.status, "critical")
        self.assertIn("critically high", check.message)

    @patch("psutil.virtual_memory")
    def test_memory_usage_check_normal(self, mock_virtual_memory):
        """Test memory usage check with normal usage."""
        mock_memory = MagicMock()
        mock_memory.percent = 60.0
        mock_memory.total = 16 * 1024**3  # 16 GB
        mock_memory.available = 6.4 * 1024**3  # 6.4 GB
        mock_virtual_memory.return_value = mock_memory

        self.checker._check_memory_usage()

        self.assertEqual(len(self.checker.checks), 1)
        check = self.checker.checks[0]
        self.assertEqual(check.name, "memory_usage")
        self.assertEqual(check.status, "healthy")
        self.assertIn("normal", check.message)

    @patch("psutil.virtual_memory")
    def test_memory_usage_check_warning(self, mock_virtual_memory):
        """Test memory usage check with warning usage."""
        mock_memory = MagicMock()
        mock_memory.percent = 85.0
        mock_memory.total = 16 * 1024**3
        mock_memory.available = 2.4 * 1024**3
        mock_virtual_memory.return_value = mock_memory

        self.checker._check_memory_usage()

        self.assertEqual(len(self.checker.checks), 1)
        check = self.checker.checks[0]
        self.assertEqual(check.name, "memory_usage")
        self.assertEqual(check.status, "warning")
        self.assertIn("high", check.message)

    @patch("psutil.disk_usage")
    def test_disk_space_check_healthy(self, mock_disk_usage):
        """Test disk space check with healthy usage."""
        mock_disk = MagicMock()
        mock_disk.percent = 50.0
        mock_disk.total = 500 * 1024**3
        mock_disk.free = 250 * 1024**3
        mock_disk_usage.return_value = mock_disk

        self.checker._check_disk_space()

        self.assertEqual(len(self.checker.checks), 1)
        check = self.checker.checks[0]
        self.assertEqual(check.name, "disk_space")
        self.assertEqual(check.status, "healthy")
        self.assertIn("adequate", check.message)

    @patch("socket.create_connection")
    def test_network_connectivity_check_success(self, mock_create_connection):
        """Test network connectivity check success."""
        mock_create_connection.return_value = MagicMock()

        self.checker._check_network_connectivity()

        self.assertEqual(len(self.checker.checks), 1)
        check = self.checker.checks[0]
        self.assertEqual(check.name, "network_connectivity")
        self.assertEqual(check.status, "healthy")
        self.assertIn("available", check.message)

    @patch("socket.create_connection")
    def test_network_connectivity_check_failure(self, mock_create_connection):
        """Test network connectivity check failure."""
        mock_create_connection.side_effect = Exception("Connection failed")

        self.checker._check_network_connectivity()

        self.assertEqual(len(self.checker.checks), 1)
        check = self.checker.checks[0]
        self.assertEqual(check.name, "network_connectivity")
        self.assertEqual(check.status, "warning")
        self.assertIn("failed", check.message)

    def test_python_version_check_compatible(self):
        """Test Python version check with compatible version."""
        # This should pass with Python 3.14
        self.checker._check_python_version()

        self.assertEqual(len(self.checker.checks), 1)
        check = self.checker.checks[0]
        self.assertEqual(check.name, "python_version")
        self.assertEqual(check.status, "healthy")
        self.assertIn("compatible", check.message)

    @patch("builtins.__import__")
    def test_dependency_check_available(self, mock_import):
        """Test dependency check with available package."""
        mock_import.return_value = MagicMock()

        # Mock the critical dependencies check
        self.checker._check_dependencies()

        # Should have 4 dependency checks
        dependency_checks = [
            c for c in self.checker.checks if c.name.startswith("dependency_")
        ]
        self.assertEqual(len(dependency_checks), 4)

        # All should be healthy
        for check in dependency_checks:
            self.assertEqual(check.status, "healthy")
            self.assertIn("available", check.message)

    @patch("os.path.exists")
    def test_data_access_check_all_exist(self, mock_exists):
        """Test data access check when all directories exist."""
        mock_exists.return_value = True

        self.checker._check_data_access()

        self.assertEqual(len(self.checker.checks), 1)
        check = self.checker.checks[0]
        self.assertEqual(check.name, "data_access")
        self.assertEqual(check.status, "healthy")
        self.assertIn("accessible", check.message)

    @patch("os.path.exists")
    def test_data_access_check_missing(self, mock_exists):
        """Test data access check when directories are missing."""
        mock_exists.return_value = False

        self.checker._check_data_access()

        self.assertEqual(len(self.checker.checks), 1)
        check = self.checker.checks[0]
        self.assertEqual(check.name, "data_access")
        self.assertEqual(check.status, "warning")
        self.assertIn("missing", check.message)


class TestRunHealthCheck(unittest.IsolatedAsyncioTestCase):
    """Test the run_health_check function."""

    async def test_run_health_check_async_returns_dict(self):
        """Test that run_health_check_async returns a dictionary."""
        async def fake_run_all_checks(self):
            self.checks = [
                _fake_health_check("cpu_usage"),
                _fake_health_check("memory_usage"),
                _fake_health_check("venue_connectivity"),
            ]
            return self.checks

        with patch.object(SystemHealthChecker, "run_all_checks_async", fake_run_all_checks):
            result = await run_health_check_async()

        self.assertIsInstance(result, dict)
        self.assertIn("status", result)
        self.assertIn("total_checks", result)
        self.assertIn("checks", result)

    async def test_run_health_check_async_has_expected_structure(self):
        """Test that run_health_check_async returns expected structure."""
        async def fake_run_all_checks(self):
            self.checks = [
                _fake_health_check("cpu_usage"),
                _fake_health_check("memory_usage"),
                _fake_health_check("venue_connectivity"),
            ]
            return self.checks

        with patch.object(SystemHealthChecker, "run_all_checks_async", fake_run_all_checks):
            result = await run_health_check_async()

        # Should have all expected keys
        expected_keys = [
            "status",
            "total_checks",
            "healthy",
            "warning",
            "critical",
            "checks",
        ]
        for key in expected_keys:
            self.assertIn(key, result)

        # Checks should be a list
        self.assertIsInstance(result["checks"], list)

        # Each check should have expected structure
        for check in result["checks"]:
            self.assertIn("name", check)
            self.assertIn("status", check)
            self.assertIn("message", check)
