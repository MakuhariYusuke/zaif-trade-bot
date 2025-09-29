#!/usr/bin/env python3
"""
test_memory_monitor.py
Unit tests for memory monitoring utilities
"""

import os
from unittest.mock import MagicMock, patch

from ztb.utils.memory_monitor import (
    check_memory_usage,
    get_memory_usage,
    log_memory_usage,
)


class TestMemoryMonitor:
    """Test memory monitoring utilities"""

    def setup_method(self):
        """Clean up environment before each test"""
        if "ZTB_DEV_MEMORY_WARN" in os.environ:
            del os.environ["ZTB_DEV_MEMORY_WARN"]

    def teardown_method(self):
        """Clean up environment after each test"""
        self.setup_method()

    def test_get_memory_usage(self):
        """Test getting current memory usage"""
        with patch("psutil.Process") as mock_process:
            mock_proc_instance = MagicMock()
            mock_proc_instance.memory_info.return_value.rss = 50 * 1024 * 1024  # 50MB
            mock_process.return_value = mock_proc_instance

            result = get_memory_usage()
            assert result == 50.0

    def test_check_memory_usage_disabled_by_default(self):
        """Test that memory check is disabled by default"""
        with patch("psutil.Process") as mock_process:
            with patch("builtins.print") as mock_print:
                mock_proc_instance = MagicMock()
                mock_proc_instance.memory_info.return_value.rss = (
                    2000 * 1024 * 1024
                )  # 2000MB
                mock_process.return_value = mock_proc_instance

                check_memory_usage(1000)
                # Should not print anything when ZTB_DEV_MEMORY_WARN is not set
                mock_print.assert_not_called()

    def test_check_memory_usage_enabled_below_threshold(self):
        """Test memory check enabled but below threshold"""
        with patch.dict(os.environ, {"ZTB_DEV_MEMORY_WARN": "1"}):
            with patch("psutil.Process") as mock_process:
                with patch("builtins.print") as mock_print:
                    mock_proc_instance = MagicMock()
                    mock_proc_instance.memory_info.return_value.rss = (
                        500 * 1024 * 1024
                    )  # 500MB
                    mock_process.return_value = mock_proc_instance

                    check_memory_usage(1000)
                    # Should not print warning when below threshold
                    mock_print.assert_not_called()

    def test_check_memory_usage_enabled_above_threshold(self):
        """Test memory check enabled and above threshold triggers warning"""
        with patch.dict(os.environ, {"ZTB_DEV_MEMORY_WARN": "1"}):
            with patch("psutil.Process") as mock_process:
                with patch("builtins.print") as mock_print:
                    mock_proc_instance = MagicMock()
                    mock_proc_instance.memory_info.return_value.rss = (
                        1500 * 1024 * 1024
                    )  # 1500MB
                    mock_process.return_value = mock_proc_instance

                    check_memory_usage(1000)
                    # Should print warning when above threshold
                    mock_print.assert_called_once_with(
                        "WARNING: High memory usage: 1500.0MB (threshold: 1000MB)"
                    )

    def test_check_memory_usage_custom_threshold(self):
        """Test memory check with custom threshold"""
        with patch.dict(os.environ, {"ZTB_DEV_MEMORY_WARN": "1"}):
            with patch("psutil.Process") as mock_process:
                with patch("builtins.print") as mock_print:
                    mock_proc_instance = MagicMock()
                    mock_proc_instance.memory_info.return_value.rss = (
                        2500 * 1024 * 1024
                    )  # 2500MB
                    mock_process.return_value = mock_proc_instance

                    check_memory_usage(2000)
                    # Should print warning when above custom threshold
                    mock_print.assert_called_once_with(
                        "WARNING: High memory usage: 2500.0MB (threshold: 2000MB)"
                    )

    def test_check_memory_usage_at_exact_threshold(self):
        """Test memory check at exact threshold (should not warn)"""
        with patch.dict(os.environ, {"ZTB_DEV_MEMORY_WARN": "1"}):
            with patch("psutil.Process") as mock_process:
                with patch("builtins.print") as mock_print:
                    mock_proc_instance = MagicMock()
                    mock_proc_instance.memory_info.return_value.rss = (
                        1000 * 1024 * 1024
                    )  # 1000MB
                    mock_process.return_value = mock_proc_instance

                    check_memory_usage(1000)
                    # Should not print warning when at exact threshold
                    mock_print.assert_not_called()

    def test_log_memory_usage_without_label(self):
        """Test logging memory usage without label"""
        with patch("psutil.Process") as mock_process:
            with patch("builtins.print") as mock_print:
                mock_proc_instance = MagicMock()
                mock_proc_instance.memory_info.return_value.rss = (
                    75.5 * 1024 * 1024
                )  # 75.5MB
                mock_process.return_value = mock_proc_instance

                log_memory_usage()
                mock_print.assert_called_once_with("Memory usage: 75.5MB")

    def test_log_memory_usage_with_label(self):
        """Test logging memory usage with label"""
        with patch("psutil.Process") as mock_process:
            with patch("builtins.print") as mock_print:
                mock_proc_instance = MagicMock()
                mock_proc_instance.memory_info.return_value.rss = (
                    200 * 1024 * 1024
                )  # 200MB
                mock_process.return_value = mock_proc_instance

                log_memory_usage("after training")
                mock_print.assert_called_once_with(
                    "Memory usage [after training]: 200.0MB"
                )

    def test_log_memory_usage_empty_label(self):
        """Test logging memory usage with empty string label"""
        with patch("psutil.Process") as mock_process:
            with patch("builtins.print") as mock_print:
                mock_proc_instance = MagicMock()
                mock_proc_instance.memory_info.return_value.rss = (
                    100 * 1024 * 1024
                )  # 100MB
                mock_process.return_value = mock_proc_instance

                log_memory_usage("")
                mock_print.assert_called_once_with("Memory usage: 100.0MB")
