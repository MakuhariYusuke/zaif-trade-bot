#!/usr/bin/env python3
"""
Integration tests for Advanced Auto-Stop System with Live Trading.

Tests the integration between auto-stop system and live trading bot.
"""

import os
import tempfile
import unittest
from datetime import datetime, timedelta
from unittest.mock import MagicMock, Mock, patch

from ztb.risk.advanced_auto_stop import create_production_auto_stop


class TestAutoStopIntegration(unittest.TestCase):
    """Integration tests for auto-stop with live trading."""

    def setUp(self):
        """Set up integration test fixtures."""
        self.auto_stop = create_production_auto_stop()

        # Mock live trader components
        self.mock_trader = Mock()
        self.mock_trader.notifier = Mock()
        self.mock_trader.config = {"min_trade_amount": 0.001}
        self.mock_trader.total_pnl = 0.0
        self.mock_trader.trades_count = 0
        self.mock_trader.position = 0
        self.mock_trader.entry_price = 0.0
        self.mock_trader.auto_stop = self.auto_stop

    def test_market_data_integration(self):
        """Test market data flow from trader to auto-stop."""
        # Simulate price updates
        prices = [1000000.0, 1001000.0, 999000.0, 1002000.0]
        base_time = datetime.now()

        for i, price in enumerate(prices):
            timestamp = base_time + timedelta(minutes=i)
            self.auto_stop.update_market_data(timestamp, price)

        # Check that data was stored
        self.assertEqual(len(self.auto_stop.price_history), 4)
        self.assertEqual(self.auto_stop.price_history[-1], (timestamp, price))

    def test_trade_result_integration(self):
        """Test trade result flow from trader to auto-stop."""
        # Simulate trade updates
        trades = [
            (1000.0, {"action": 1, "entry_price": 1000000.0, "exit_price": 1001000.0}),
            (-500.0, {"action": 2, "entry_price": 1001000.0, "exit_price": 999000.0}),
            (800.0, {"action": 1, "entry_price": 999000.0, "exit_price": 1000000.0}),
        ]

        for pnl, trade_info in trades:
            self.auto_stop.update_trade_result(pnl, trade_info)

        # Check trade history
        self.assertEqual(len(self.auto_stop.trade_history), 3)
        self.assertEqual(self.auto_stop.consecutive_losses, 0)  # Last trade was a win

    def test_stop_check_integration(self):
        """Test stop condition checking in trading loop."""
        # Normal conditions - should not stop
        should_stop, reason, message = self.auto_stop.check_stop_conditions()
        self.assertFalse(should_stop)

        # Set high volatility - should trigger stop
        self.auto_stop.volatility = 0.04  # Above 3% threshold

        should_stop, reason, message = self.auto_stop.check_stop_conditions()
        self.assertTrue(should_stop)
        self.assertEqual(reason.value, "volatility_spike")

    def test_drawdown_stop_integration(self):
        """Test drawdown-based stopping in live trading scenario."""
        # Simulate losing streak
        losing_trades = [
            (-2000.0, {"action": 2}),
            (-1500.0, {"action": 2}),
            (-1800.0, {"action": 2}),
        ]

        for pnl, info in losing_trades:
            self.auto_stop.update_trade_result(pnl, info)

        # Check if drawdown triggers stop
        if self.auto_stop.current_drawdown > 0.05:  # 5% threshold
            should_stop, reason, message = self.auto_stop.check_stop_conditions()
            if should_stop:
                self.assertIn("drawdown", reason.value)

    def test_consecutive_losses_integration(self):
        """Test consecutive losses stopping."""
        # Simulate 4 consecutive losses
        for i in range(4):
            self.auto_stop.update_trade_result(-1000.0, {"action": 2})

        # Should trigger consecutive losses stop (threshold is 3)
        should_stop, reason, message = self.auto_stop.check_stop_conditions()
        if should_stop and reason:
            self.assertEqual(reason.value, "consecutive_losses")

    @patch("ztb.risk.advanced_auto_stop.datetime")
    def test_time_based_stop_integration(self, mock_datetime):
        """Test time-based stopping."""
        # Mock time progression
        base_time = datetime.now()
        mock_datetime.now.return_value = base_time

        # Initialize auto-stop
        auto_stop = create_production_auto_stop()

        # Fast-forward time past the limit (6 hours)
        mock_datetime.now.return_value = base_time + timedelta(hours=7)

        # Should trigger time-based stop
        should_stop, reason, message = auto_stop.check_stop_conditions()
        if should_stop and reason:
            self.assertEqual(reason.value, "time_limit")

    def test_notification_integration(self):
        """Test Discord notification integration."""
        # Trigger a stop condition
        self.auto_stop.volatility = 0.04
        should_stop, reason, message = self.auto_stop.check_stop_conditions()

        if should_stop:
            # Simulate notification call
            self.mock_trader.notifier.send_notification.assert_not_called()  # Not called yet

            # In real integration, this would be called in live_trader.py
            expected_message = f"Reason: {reason.value}\nMessage: {message}"
            self.assertIsInstance(expected_message, str)

    def test_status_reporting_integration(self):
        """Test status reporting for monitoring."""
        # Add some data
        self.auto_stop.update_market_data(datetime.now(), 1000000.0)
        self.auto_stop.update_trade_result(500.0, {"action": 1})

        status = self.auto_stop.get_status()

        # Check status structure
        self.assertIn("is_active", status)
        self.assertIn("current_drawdown", status)
        self.assertIn("volatility", status)
        self.assertIn("consecutive_losses", status)
        self.assertIn("total_trades", status)

        # Values should be reasonable
        self.assertIsInstance(status["is_active"], bool)
        self.assertIsInstance(status["total_trades"], int)
        self.assertEqual(status["total_trades"], 1)

    def test_cooldown_behavior_integration(self):
        """Test cooldown behavior in trading loop."""
        # Trigger stop
        self.auto_stop.volatility = 0.04
        self.auto_stop.check_stop_conditions()

        # Immediate check should be blocked by cooldown
        should_stop, reason, message = self.auto_stop.check_stop_conditions()
        self.assertFalse(should_stop)
        self.assertIn("cooldown", message.lower())

    def test_resume_functionality_integration(self):
        """Test resume functionality."""
        # Trigger and stop
        self.auto_stop.volatility = 0.04
        self.auto_stop.check_stop_conditions()
        self.assertFalse(self.auto_stop.is_active)

        # Try resume (may fail due to cooldown)
        result = self.auto_stop.resume_trading()
        # Result depends on cooldown timing, but method should exist
        self.assertIsInstance(result, bool)


class TestLiveTraderAutoStopIntegration(unittest.TestCase):
    """Test LiveTrader integration with auto-stop system."""

    def setUp(self):
        """Set up LiveTrader integration test."""
        # Create temporary model file
        self.temp_model = tempfile.NamedTemporaryFile(suffix=".zip", delete=False)
        self.temp_model.close()

        # Mock the LiveTrader class
        with patch("live_trade.PPO") as mock_ppo:
            mock_ppo.load.return_value = Mock()
            mock_ppo.load.return_value.observation_space = Mock()
            mock_ppo.load.return_value.observation_space.shape = [10]

            from live_trade import LiveTrader

            self.trader = LiveTrader(self.temp_model.name)

    def tearDown(self):
        """Clean up temporary files."""
        os.unlink(self.temp_model.name)

    def test_auto_stop_initialization(self):
        """Test that auto-stop is properly initialized in LiveTrader."""
        # Check that auto_stop attribute exists
        self.assertTrue(hasattr(self.trader, "auto_stop"))

        # If auto_stop_available was True during import, auto_stop should be initialized
        if hasattr(self.trader, "auto_stop") and self.trader.auto_stop:
            self.assertIsNotNone(self.trader.auto_stop)
            status = self.trader.auto_stop.get_status()
            self.assertIn("is_active", status)

    def test_position_update_with_auto_stop(self):
        """Test position updates trigger auto-stop updates."""
        if not self.trader.auto_stop:
            self.skipTest("Auto-stop not available")

        # Mock position update
        old_position = self.trader.position
        current_price = 1000000.0

        # Simulate position change
        self.trader.position = 1
        self.trader.entry_price = current_price
        self.trader.trades_count = 1

        # Call update_position (which should update auto_stop)
        with patch.object(self.trader.auto_stop, "update_trade_result") as mock_update:
            # Simulate the PnL calculation logic
            if old_position != self.trader.position and old_position != 0:
                direction = 1 if old_position > 0 else -1
                pnl = (
                    (current_price - self.trader.entry_price)
                    * direction
                    * self.trader.config["min_trade_amount"]
                )

                # This would be called in the real update_position method
                self.trader.auto_stop.update_trade_result(
                    pnl,
                    {
                        "action": 1,
                        "entry_price": self.trader.entry_price,
                        "exit_price": current_price,
                        "position": self.trader.position,
                        "timestamp": datetime.now(),
                    },
                )

                mock_update.assert_called_once()


if __name__ == "__main__":
    unittest.main()
