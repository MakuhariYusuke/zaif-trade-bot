"""
Unit tests for memory_monitor.py
"""

import time
import unittest
from unittest.mock import Mock, patch

from ztb.utils.config import ZTBConfig
from ztb.utils.memory_monitor import (
    MemoryMonitor,
    get_memory_monitor,
    check_memory_usage,
    get_memory_usage,
    log_memory_usage,
)


class TestMemoryMonitor(unittest.TestCase):
    """Test cases for memory monitoring functionality."""

    def setUp(self):
        """Set up test fixtures."""
        self.config = ZTBConfig()
        self.monitor = MemoryMonitor(self.config)

    def tearDown(self):
        """Clean up after tests."""
        self.monitor.stop_monitoring()

    @patch('psutil.Process')
    def test_get_memory_usage(self, mock_process):
        """Test getting memory usage."""
        mock_process.return_value.memory_info.return_value.rss = 512 * 1024 * 1024  # 512MB

        usage = get_memory_usage()
        self.assertAlmostEqual(usage, 512.0, places=1)

    @patch('psutil.Process')
    def test_check_memory_usage_warning(self, mock_process):
        """Test memory usage warning."""
        # Mock high memory usage
        mock_process.return_value.memory_info.return_value.rss = 1500 * 1024 * 1024  # 1.5GB

        with patch('ztb.utils.memory_monitor.ZTBConfig') as mock_config_class:
            mock_config = Mock()
            mock_config.get_bool.return_value = True
            mock_config_class.return_value = mock_config

            with patch('builtins.print') as mock_print:
                check_memory_usage(threshold_mb=1000)
                mock_print.assert_called_once()
                call_args = mock_print.call_args[0][0]
                self.assertIn("WARNING", call_args)
                self.assertIn("1500.0MB", call_args)

    @patch('psutil.Process')
    def test_check_memory_usage_no_warning(self, mock_process):
        """Test memory usage with no warning."""
        # Mock normal memory usage
        mock_process.return_value.memory_info.return_value.rss = 500 * 1024 * 1024  # 500MB

        with patch('builtins.print') as mock_print:
            check_memory_usage(threshold_mb=1000)
            mock_print.assert_not_called()

    def test_log_memory_usage(self):
        """Test logging memory usage."""
        with patch('builtins.print') as mock_print:
            log_memory_usage("test_label")
            mock_print.assert_called_once()
            call_args = mock_print.call_args[0][0]
            self.assertIn("Memory usage", call_args)
            self.assertIn("[test_label]", call_args)

    @patch('psutil.Process')
    def test_memory_monitor_record_usage(self, mock_process):
        """Test recording memory usage in monitor."""
        mock_process.return_value.memory_info.return_value.rss = 256 * 1024 * 1024  # 256MB

        usage = self.monitor.record_memory_usage()
        self.assertAlmostEqual(usage, 256.0, places=1)

        # Check history
        self.assertEqual(len(self.monitor.memory_history), 1)
        entry = self.monitor.memory_history[0]
        self.assertAlmostEqual(entry['memory_mb'], 256.0, places=1)
        self.assertIsInstance(entry['timestamp'], float)

    def test_memory_monitor_stats_empty(self):
        """Test memory stats with no data."""
        stats = self.monitor.get_memory_stats()

        self.assertEqual(stats['current_mb'], 0.0)
        self.assertEqual(stats['average_mb'], 0.0)
        self.assertEqual(stats['peak_mb'], 0.0)
        self.assertEqual(stats['samples'], 0)

    @patch('psutil.Process')
    def test_memory_monitor_stats_with_data(self, mock_process):
        """Test memory stats with recorded data."""
        # Record different memory values
        values = [100, 200, 150, 300, 250]
        for i, val in enumerate(values):
            mock_process.return_value.memory_info.return_value.rss = val * 1024 * 1024
            self.monitor.record_memory_usage()

        stats = self.monitor.get_memory_stats()

        self.assertEqual(stats['current_mb'], 250.0)  # Last recorded
        self.assertAlmostEqual(stats['average_mb'], 200.0, places=1)  # Average
        self.assertEqual(stats['peak_mb'], 300.0)  # Maximum
        self.assertEqual(stats['samples'], 5)

    def test_memory_trend_insufficient_data(self):
        """Test memory trend with insufficient data."""
        trend = self.monitor.get_memory_trend()
        self.assertEqual(trend, "insufficient_data")

    @patch('psutil.Process')
    def test_memory_trend_increasing(self, mock_process):
        """Test memory trend increasing."""
        # Record increasing values
        for val in [100, 110, 120, 130, 140, 150, 160, 170, 180, 190]:
            mock_process.return_value.memory_info.return_value.rss = val * 1024 * 1024
            self.monitor.record_memory_usage()

        trend = self.monitor.get_memory_trend()
        self.assertEqual(trend, "increasing")

    @patch('psutil.Process')
    def test_memory_trend_decreasing(self, mock_process):
        """Test memory trend decreasing."""
        # Record decreasing values
        for val in [200, 190, 180, 170, 160, 150, 140, 130, 120, 110]:
            mock_process.return_value.memory_info.return_value.rss = val * 1024 * 1024
            self.monitor.record_memory_usage()

        trend = self.monitor.get_memory_trend()
        self.assertEqual(trend, "decreasing")

    @patch('psutil.Process')
    def test_memory_trend_stable(self, mock_process):
        """Test memory trend stable."""
        # Record stable values
        for val in [150, 152, 148, 151, 149, 153, 147, 150, 152, 148]:
            mock_process.return_value.memory_info.return_value.rss = val * 1024 * 1024
            self.monitor.record_memory_usage()

        trend = self.monitor.get_memory_trend()
        self.assertEqual(trend, "stable")

    @patch('psutil.Process')
    def test_memory_monitor_alerts(self, mock_process):
        """Test memory monitoring alerts."""
        # Set high memory usage
        mock_process.return_value.memory_info.return_value.rss = 2500 * 1024 * 1024  # 2.5GB

        with patch('ztb.utils.memory_monitor.logger') as mock_logger:
            self.monitor.record_memory_usage()
            self.monitor._check_alerts()

            # Should trigger critical alert
            mock_logger.error.assert_called_once()
            call_args = mock_logger.error.call_args[0][0]
            self.assertIn("CRITICAL", call_args)
            self.assertIn("2500.0MB", call_args)

    def test_memory_monitor_start_stop(self):
        """Test starting and stopping memory monitoring."""
        # Start monitoring
        self.monitor.start_monitoring(interval_seconds=0.1)
        self.assertTrue(self.monitor.monitoring_active)

        # Stop monitoring
        self.monitor.stop_monitoring()
        self.assertFalse(self.monitor.monitoring_active)

    def test_get_memory_monitor_singleton(self):
        """Test getting memory monitor singleton."""
        monitor1 = get_memory_monitor()
        monitor2 = get_memory_monitor()

        self.assertIs(monitor1, monitor2)
        self.assertIsInstance(monitor1, MemoryMonitor)


if __name__ == '__main__':
    unittest.main()