"""
Unit tests for health_monitor.py
"""

import time
import unittest
from unittest.mock import MagicMock, patch

from ztb.ops.monitoring.health_monitor import HealthMonitor


class TestHealthMonitor(unittest.TestCase):
    def setUp(self):
        """Set up test fixtures."""
        self.monitor = HealthMonitor("test_service")
        # Reset start time for consistent testing
        self.monitor.start_time = time.time() - 1

    def test_initialization(self):
        """Test monitor initialization."""
        self.assertEqual(self.monitor.service_name, "test_service")
        self.assertIsNotNone(self.monitor.logger)
        self.assertGreater(self.monitor.start_time, 0)

    @patch("ztb.scripts.health_monitor.psutil.Process")
    def test_check_memory_usage_success(self, mock_process_class):
        """Test successful memory check."""
        mock_process = MagicMock()
        mock_process.memory_info.return_value.rss = 100 * 1024 * 1024  # 100MB
        mock_process.memory_percent.return_value = 50.0
        mock_process_class.return_value = mock_process

        result = self.monitor._check_memory_usage()
        self.assertTrue(result["healthy"])
        self.assertEqual(result["used_mb"], 100.0)
        self.assertEqual(result["used_percent"], 50.0)

    @patch("ztb.scripts.health_monitor.psutil.Process")
    def test_check_memory_usage_high_usage(self, mock_process_class):
        """Test memory check with high usage."""
        mock_process = MagicMock()
        mock_process.memory_info.return_value.rss = 900 * 1024 * 1024  # 900MB
        mock_process.memory_percent.return_value = 85.0
        mock_process_class.return_value = mock_process

        result = self.monitor._check_memory_usage()
        self.assertFalse(result["healthy"])
        self.assertEqual(result["used_percent"], 85.0)

    @patch("ztb.scripts.health_monitor.psutil.Process")
    def test_check_cpu_usage_success(self, mock_process_class):
        """Test successful CPU check."""
        mock_process = MagicMock()
        mock_process.cpu_percent.return_value = 30.0
        mock_process_class.return_value = mock_process

        result = self.monitor._check_cpu_usage()
        self.assertTrue(result["healthy"])
        self.assertEqual(result["used_percent"], 30.0)

    @patch("ztb.scripts.health_monitor.psutil.disk_usage")
    def test_check_disk_space_success(self, mock_disk_usage):
        """Test successful disk space check."""
        mock_disk_usage.return_value.free = 2 * 1024 * 1024 * 1024  # 2GB
        mock_disk_usage.return_value.total = 100 * 1024 * 1024 * 1024  # 100GB

        result = self.monitor._check_disk_space()
        self.assertTrue(result["healthy"])
        self.assertEqual(result["free_gb"], 2.0)
        self.assertEqual(result["total_gb"], 100.0)

    @patch("ztb.scripts.health_monitor.psutil.disk_usage")
    def test_check_disk_space_low_space(self, mock_disk_usage):
        """Test disk space check with low space."""
        mock_disk_usage.return_value.free = 500 * 1024 * 1024  # 500MB
        mock_disk_usage.return_value.total = 50 * 1024 * 1024 * 1024  # 50GB

        result = self.monitor._check_disk_space()
        self.assertFalse(result["healthy"])
        self.assertLess(result["free_gb"], 1)

    def test_check_uptime(self):
        """Test uptime check."""
        result = self.monitor._check_uptime()
        self.assertTrue(result["healthy"])
        self.assertGreater(result["uptime_seconds"], 0)
        self.assertGreater(result["uptime_hours"], 0)

    @patch("ztb.scripts.health_monitor.Path")
    def test_check_log_files_exists(self, mock_path_class):
        """Test log file check when file exists."""
        mock_path = MagicMock()
        mock_path.exists.return_value = True
        mock_stat = MagicMock()
        mock_stat.st_size = 50 * 1024 * 1024  # 50MB
        mock_path.stat.return_value = mock_stat
        mock_path_class.return_value = mock_path

        result = self.monitor._check_log_files()
        self.assertTrue(result["healthy"])
        self.assertEqual(result["size_mb"], 50.0)
        self.assertTrue(result["exists"])

    @patch("ztb.scripts.health_monitor.Path")
    def test_check_log_files_missing(self, mock_path_class):
        """Test log file check when file is missing."""
        mock_path = MagicMock()
        mock_path.exists.return_value = False
        mock_path_class.return_value = mock_path

        result = self.monitor._check_log_files()
        self.assertFalse(result["healthy"])
        self.assertFalse(result["exists"])

    @patch("ztb.scripts.health_monitor.Path")
    def test_check_configuration_success(self, mock_path_class):
        """Test configuration check with existing files."""
        # Mock Path constructor to return different mocks for different paths
        mock_paths = {}

        def path_constructor(path_str):
            if path_str not in mock_paths:
                mock_path = MagicMock()
                mock_path.exists.return_value = path_str in [
                    "config/production.yaml",
                    "config/default.yaml",
                ]
                mock_paths[path_str] = mock_path
            return mock_paths[path_str]

        mock_path_class.side_effect = path_constructor

        result = self.monitor._check_configuration()
        self.assertTrue(result["healthy"])
        self.assertIn("config/production.yaml", result["config_files"])
        self.assertIn("config/default.yaml", result["config_files"])

    @patch("ztb.scripts.health_monitor.Path")
    def test_check_configuration_no_files(self, mock_path_class):
        """Test configuration check with no config files."""
        mock_path = MagicMock()
        mock_path.exists.return_value = False
        mock_path_class.return_value = mock_path

        result = self.monitor._check_configuration()
        self.assertFalse(result["healthy"])
        self.assertEqual(len(result["config_files"]), 0)

    @patch("ztb.scripts.health_monitor.psutil.Process")
    def test_collect_metrics(self, mock_process_class):
        """Test metrics collection."""
        mock_process = MagicMock()
        mock_process.pid = 12345
        mock_process.num_threads.return_value = 4
        mock_process.open_files.return_value = ["file1", "file2"]
        mock_process.connections.return_value = ["conn1"]
        mock_process_class.return_value = mock_process

        result = self.monitor._collect_metrics()
        self.assertEqual(result["process_id"], 12345)
        self.assertEqual(result["threads"], 4)
        self.assertEqual(result["open_files"], 2)
        self.assertEqual(result["connections"], 1)

    def test_should_restart_healthy(self):
        """Test restart decision for healthy status."""
        health_status = {"status": "healthy", "checks": {}}
        result = self.monitor.should_restart(health_status)
        self.assertFalse(result)

    def test_should_restart_unhealthy(self):
        """Test restart decision for unhealthy status."""
        health_status = {"status": "unhealthy", "checks": {}}
        result = self.monitor.should_restart(health_status)
        self.assertTrue(result)

    def test_should_restart_critical_failure(self):
        """Test restart decision for critical check failure."""
        health_status = {
            "status": "degraded",
            "checks": {"memory": {"healthy": False}, "cpu": {"healthy": True}},
        }
        result = self.monitor.should_restart(health_status)
        self.assertTrue(result)


if __name__ == "__main__":
    unittest.main()
