"""
Unit tests for memory_monitor.py
"""

import unittest
from unittest.mock import Mock, patch

from ztb.trading.environment.constants import BYTES_PER_GB, BYTES_PER_MB
from ztb.utils.config import ZTBConfig
from ztb.utils.memory_monitor import (
    BackgroundMemoryMonitor,
    build_post_cycle_memory_status,
    check_memory_usage,
    get_memory_monitor,
    get_memory_usage,
    log_memory_usage,
)


class TestBackgroundMemoryMonitor(unittest.TestCase):
    """Test cases for memory monitoring functionality."""

    def setUp(self):
        """Set up test fixtures."""
        self.config = ZTBConfig()
        self.monitor = BackgroundMemoryMonitor(self.config)

    def tearDown(self):
        """Clean up after tests."""
        self.monitor.stop_monitoring()

    @patch("psutil.Process")
    def test_get_memory_usage(self, mock_process):
        """Test getting memory usage."""
        mock_process.return_value.memory_info.return_value.rss = (
            512 * BYTES_PER_MB
        )  # 512MB

        usage = get_memory_usage()
        self.assertAlmostEqual(usage, 512.0, places=1)

    @patch("psutil.Process")
    def test_check_memory_usage_warning(self, mock_process):
        """Test memory usage warning."""
        # Mock high memory usage
        mock_process.return_value.memory_info.return_value.rss = (
            1.5 * BYTES_PER_GB
        )  # 1.5GB

        with patch("ztb.utils.memory_monitor.ZTBConfig") as mock_config_class:
            mock_config = Mock()
            mock_config.get_bool.return_value = True
            mock_config_class.return_value = mock_config

            with patch("builtins.print") as mock_print:
                check_memory_usage(threshold_mb=1000)
                mock_print.assert_called_once()
                call_args = mock_print.call_args[0][0]
                self.assertIn("WARNING", call_args)
                self.assertIn("1536.0MB", call_args)

    @patch("psutil.Process")
    def test_check_memory_usage_no_warning(self, mock_process):
        """Test memory usage with no warning."""
        # Mock normal memory usage
        mock_process.return_value.memory_info.return_value.rss = (
            500 * BYTES_PER_MB
        )  # 500MB

        with patch("builtins.print") as mock_print:
            check_memory_usage(threshold_mb=1000)
            mock_print.assert_not_called()

    def test_log_memory_usage(self):
        """Test logging memory usage."""
        with patch("builtins.print") as mock_print:
            log_memory_usage("test_label")
            mock_print.assert_called_once()
            call_args = mock_print.call_args[0][0]
            self.assertIn("Memory usage", call_args)
            self.assertIn("[test_label]", call_args)

    @patch("psutil.Process")
    def test_memory_monitor_record_usage(self, mock_process):
        """Test recording memory usage in monitor."""
        mock_process.return_value.memory_info.return_value.rss = (
            256 * BYTES_PER_MB
        )  # 256MB

        usage = self.monitor.record_memory_usage()
        self.assertAlmostEqual(usage, 256.0, places=1)

        # Check history
        self.assertEqual(len(self.monitor.memory_history), 1)
        entry = self.monitor.memory_history[0]
        self.assertAlmostEqual(entry["memory_mb"], 256.0, places=1)
        self.assertIsInstance(entry["timestamp"], float)

    def test_memory_monitor_stats_empty(self):
        """Test memory stats with no data."""
        stats = self.monitor.get_memory_stats()

        self.assertEqual(stats["current_mb"], 0.0)
        self.assertEqual(stats["average_mb"], 0.0)
        self.assertEqual(stats["peak_mb"], 0.0)
        self.assertEqual(stats["samples"], 0)

    @patch("psutil.Process")
    def test_memory_monitor_stats_with_data(self, mock_process):
        """Test memory stats with recorded data."""
        # Record different memory values
        values = [100, 200, 150, 300, 250]
        for i, val in enumerate(values):
            mock_process.return_value.memory_info.return_value.rss = val * BYTES_PER_MB
            self.monitor.record_memory_usage()

        stats = self.monitor.get_memory_stats()

        self.assertEqual(stats["current_mb"], 250.0)  # Last recorded
        self.assertAlmostEqual(stats["average_mb"], 200.0, places=1)  # Average
        self.assertEqual(stats["peak_mb"], 300.0)  # Maximum
        self.assertEqual(stats["samples"], 5)

    def test_memory_trend_insufficient_data(self):
        """Test memory trend with insufficient data."""
        trend = self.monitor.get_memory_trend()
        self.assertEqual(trend, "insufficient_data")

    @patch("psutil.Process")
    def test_memory_trend_increasing(self, mock_process):
        """Test memory trend increasing."""
        # Record increasing values
        for val in [100, 110, 120, 130, 140, 150, 160, 170, 180, 190]:
            mock_process.return_value.memory_info.return_value.rss = val * BYTES_PER_MB
            self.monitor.record_memory_usage()

        trend = self.monitor.get_memory_trend()
        self.assertEqual(trend, "increasing")

    @patch("psutil.Process")
    def test_memory_trend_decreasing(self, mock_process):
        """Test memory trend decreasing."""
        # Record decreasing values
        for val in [200, 190, 180, 170, 160, 150, 140, 130, 120, 110]:
            mock_process.return_value.memory_info.return_value.rss = val * BYTES_PER_MB
            self.monitor.record_memory_usage()

        trend = self.monitor.get_memory_trend()
        self.assertEqual(trend, "decreasing")

    @patch("psutil.Process")
    def test_memory_trend_stable(self, mock_process):
        """Test memory trend stable."""
        # Record stable values
        for val in [150, 152, 148, 151, 149, 153, 147, 150, 152, 148]:
            mock_process.return_value.memory_info.return_value.rss = val * BYTES_PER_MB
            self.monitor.record_memory_usage()

        trend = self.monitor.get_memory_trend()
        self.assertEqual(trend, "stable")

    @patch("psutil.Process")
    def test_memory_monitor_alerts(self, mock_process):
        """Test memory monitoring alerts."""
        # Set high memory usage
        mock_process.return_value.memory_info.return_value.rss = (
            2.5 * BYTES_PER_GB
        )  # 2.5GB

        with patch("ztb.utils.memory_monitor.logger") as mock_logger:
            self.monitor.record_memory_usage()
            self.monitor._check_alerts()

            # Should trigger critical alert
            mock_logger.error.assert_called_once()
            call_args = mock_logger.error.call_args[0][0]
            self.assertIn("CRITICAL", call_args)
            self.assertIn("2560.0MB", call_args)

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
        self.assertIsInstance(monitor1, BackgroundMemoryMonitor)


class TestPostCycleMemoryStatus(unittest.TestCase):
    @patch("ztb.utils.memory_monitor.get_memory_snapshot")
    def test_marks_leak_and_threshold(self, mock_get_memory_snapshot):
        mock_get_memory_snapshot.return_value = {
            "rss": 320.0,
            "cache_total_entries": 7.0,
        }

        status = build_post_cycle_memory_status(
            150.0,
            rss_warning_mb=256.0,
        )

        self.assertEqual(status["rss_mb"], 320.0)
        self.assertEqual(status["rss_delta_mb"], 170.0)
        self.assertEqual(status["cache_total_entries"], 7.0)
        self.assertTrue(status["leak_warning"])
        self.assertTrue(status["rss_warning"])

    @patch("ztb.utils.memory_monitor.get_memory_snapshot")
    def test_skips_leak_on_first_cycle(self, mock_get_memory_snapshot):
        mock_get_memory_snapshot.return_value = {
            "rss": 80.0,
            "cache_total_entries": 0.0,
        }

        status = build_post_cycle_memory_status(
            0.0,
            rss_warning_mb=256.0,
        )

        self.assertEqual(status["rss_delta_mb"], 0.0)
        self.assertFalse(status["leak_warning"])
        self.assertFalse(status["rss_warning"])


class TestBackgroundMemoryMonitorThresholdAdjustment(unittest.TestCase):
    """Test memory monitor threshold adjustments for DEBUG log enhancements."""

    def setUp(self):
        """Set up test fixtures."""
        self.config = ZTBConfig()

    def test_warning_threshold_increased_to_1000mb(self):
        """Test that warning threshold has been increased from 500MB to 1000MB."""
        monitor = BackgroundMemoryMonitor(self.config)

        # The warning threshold should now be 1000MB (increased from 500MB)
        expected_threshold = 1000
        self.assertEqual(
            monitor.warning_threshold_mb,
            expected_threshold,
            f"Warning threshold should be {expected_threshold}MB, got {monitor.warning_threshold_mb}MB",
        )

    def test_config_default_warning_threshold(self):
        """Test that config default for warning threshold is 1000MB."""
        # Test with default config (no explicit setting)
        monitor = BackgroundMemoryMonitor(self.config)

        # Should use the new default of 1000MB
        self.assertEqual(monitor.warning_threshold_mb, 1000)

    @patch("psutil.Process")
    def test_warning_not_triggered_at_800mb(self, mock_process):
        """Test that warning is not triggered at 800MB (below new 1000MB threshold)."""
        # Mock memory usage at 800MB
        mock_process.return_value.memory_info.return_value.rss = (
            800 * BYTES_PER_MB
        )  # 800MB

        monitor = BackgroundMemoryMonitor(self.config)

        # Should not trigger warning at 800MB
        with patch("ztb.utils.memory_monitor.logger") as mock_logger:
            monitor.record_memory_usage()
            monitor._check_alerts()
            # Warning should not be called
            mock_logger.warning.assert_not_called()

    @patch("psutil.Process")
    def test_warning_triggered_at_1200mb(self, mock_process):
        """Test that warning is triggered at 1200MB (above new 1000MB threshold)."""
        # Mock memory usage at 1200MB
        mock_process.return_value.memory_info.return_value.rss = (
            1200 * BYTES_PER_MB
        )  # 1200MB

        monitor = BackgroundMemoryMonitor(self.config)

        # Should trigger warning at 1200MB
        with patch("ztb.utils.memory_monitor.logger") as mock_logger:
            monitor.record_memory_usage()
            monitor._check_alerts()
            # Warning should be called
            mock_logger.warning.assert_called_once()
            warning_call = mock_logger.warning.call_args[0][0]
            self.assertIn("1200.0MB", warning_call)
            self.assertIn("1000MB", warning_call)

    def test_threshold_values_are_reasonable(self):
        """Test that threshold values are set to reasonable levels."""
        monitor = BackgroundMemoryMonitor(self.config)

        # Warning threshold should be reasonable (not too low or high)
        self.assertGreater(
            monitor.warning_threshold_mb, 100, "Warning threshold should be > 100MB"
        )
        self.assertLess(
            monitor.warning_threshold_mb, 5000, "Warning threshold should be < 5000MB"
        )

        # Alert threshold should be higher than warning threshold
        self.assertGreater(
            monitor.alert_threshold_mb,
            monitor.warning_threshold_mb,
            "Alert threshold should be higher than warning threshold",
        )


if __name__ == "__main__":
    unittest.main()
