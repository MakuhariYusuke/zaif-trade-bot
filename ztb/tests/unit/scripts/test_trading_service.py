"""
Unit tests for trading_service.py
"""

import signal
import unittest
from unittest.mock import patch

from ztb.scripts.trading_service import TradingService


class TestTradingService(unittest.TestCase):
    def setUp(self):
        """Set up test fixtures."""
        self.service = TradingService(log_level="DEBUG")

    def test_initialization(self):
        """Test service initialization."""
        self.assertIsNotNone(self.service.health_monitor)
        self.assertEqual(self.service.log_level, "DEBUG")
        self.assertFalse(self.service.running)
        self.assertEqual(self.service.restart_count, 0)

    @patch("ztb.scripts.trading_service.time.sleep")
    def test_run_trading_cycle_success(self, mock_sleep):
        """Test successful trading cycle."""
        # Mock successful cycle
        with patch.object(self.service, "_simulate_trading_cycle"):
            with patch.object(self.service, "_check_health", return_value=True):
                result = self.service._run_trading_cycle()
                self.assertTrue(result)

    @patch("ztb.scripts.trading_service.time.sleep")
    def test_run_trading_cycle_failure(self, mock_sleep):
        """Test failed trading cycle."""
        # Mock failed cycle
        with patch.object(
            self.service, "_simulate_trading_cycle", side_effect=Exception("Test error")
        ):
            result = self.service._run_trading_cycle()
            self.assertFalse(result)

    def test_check_health_success(self):
        """Test successful health check."""
        with patch.object(
            self.service.health_monitor,
            "check_overall_health",
            return_value={
                "status": "healthy",
                "checks": {"memory": {"healthy": True}, "cpu": {"healthy": True}},
            },
        ):
            with patch.object(
                self.service.health_monitor, "should_restart", return_value=False
            ):
                result = self.service._check_health()
                self.assertTrue(result)

    def test_check_health_failure(self):
        """Test failed health check."""
        with patch.object(
            self.service.health_monitor,
            "check_overall_health",
            return_value={
                "status": "unhealthy",
                "checks": {"memory": {"healthy": False}},
            },
        ):
            result = self.service._check_health()
            self.assertFalse(result)

    def test_should_restart_success(self):
        """Test restart decision for successful cycle."""
        result = self.service._should_restart(True)
        self.assertTrue(result)
        self.assertEqual(self.service.restart_count, 0)

    def test_should_restart_failure_within_limit(self):
        """Test restart decision for failed cycle within limit."""
        self.service.restart_count = 5
        self.service.max_restarts = 10
        result = self.service._should_restart(False)
        self.assertTrue(result)
        self.assertEqual(self.service.restart_count, 6)

    def test_should_restart_failure_over_limit(self):
        """Test restart decision for failed cycle over limit."""
        self.service.restart_count = 9
        self.service.max_restarts = 10
        result = self.service._should_restart(False)
        self.assertFalse(result)
        self.assertEqual(self.service.restart_count, 10)

    def test_signal_handler(self):
        """Test signal handler."""
        self.service.running = True
        self.service._signal_handler(signal.SIGTERM, None)
        self.assertFalse(self.service.running)

    @patch("ztb.scripts.trading_service.time.sleep")
    def test_run_service_success(self, mock_sleep):
        """Test successful service run."""
        with patch.object(self.service, "_run_trading_cycle", return_value=True):
            with patch.object(self.service, "_should_restart", return_value=False):
                # Run service briefly
                self.service.running = True

                # Simulate stopping after one cycle
                def stop_after_cycle(cycle_success):
                    self.service.running = False
                    return False

                with patch.object(
                    self.service, "_should_restart", side_effect=stop_after_cycle
                ):
                    self.service.run()
                    self.assertFalse(self.service.running)


if __name__ == "__main__":
    unittest.main()
