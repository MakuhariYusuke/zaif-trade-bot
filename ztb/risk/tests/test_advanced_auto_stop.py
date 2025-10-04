#!/usr/bin/env python3
"""
Unit tests for Advanced Auto-Stop System.

Tests the risk management and automatic stop functionality for live trading.
"""

import unittest
from datetime import datetime, timedelta
from unittest.mock import Mock, patch

import numpy as np

from ztb.risk.advanced_auto_stop import (
    AdvancedAutoStop,
    StopReason,
    create_production_auto_stop,
)


class TestAdvancedAutoStop(unittest.TestCase):
    """Test cases for AdvancedAutoStop class."""

    def setUp(self):
        """Set up test fixtures."""
        self.config = {
            "volatility_stop": {
                "enabled": True,
                "threshold": 0.05,
                "window_size": 60,
                "cooldown_period": 300,
                "severity": "warning",
            },
            "drawdown_stop": {
                "enabled": True,
                "threshold": 0.10,
                "window_size": 1440,
                "cooldown_period": 3600,
                "severity": "critical",
            },
        }
        self.auto_stop = AdvancedAutoStop(self.config)

    def test_initialization(self):
        """Test auto-stop system initialization."""
        self.assertTrue(self.auto_stop.is_active)
        self.assertIsNone(self.auto_stop.stop_reason)
        self.assertIsNone(self.auto_stop.cooldown_until)
        self.assertEqual(self.auto_stop.current_drawdown, 0.0)
        self.assertEqual(self.auto_stop.volatility, 0.0)
        self.assertEqual(self.auto_stop.consecutive_losses, 0)

    def test_update_market_data(self):
        """Test market data updates."""
        timestamp = datetime.now()
        price = 1000000.0

        self.auto_stop.update_market_data(timestamp, price)

        self.assertEqual(len(self.auto_stop.price_history), 1)
        self.assertEqual(self.auto_stop.price_history[0], (timestamp, price))

    def test_volatility_calculation(self):
        """Test volatility calculation with price history."""
        base_time = datetime.now()
        prices = [1000000.0, 1001000.0, 999000.0, 1002000.0, 998000.0]

        for i, price in enumerate(prices):
            timestamp = base_time + timedelta(minutes=i)
            self.auto_stop.update_market_data(timestamp, price)

        # Should have calculated volatility
        self.assertGreaterEqual(self.auto_stop.volatility, 0.0)

    def test_update_trade_result(self):
        """Test trade result updates."""
        # Winning trade
        self.auto_stop.update_trade_result(1000.0, {"action": 1})
        self.assertEqual(self.auto_stop.consecutive_losses, 0)

        # Losing trade
        self.auto_stop.update_trade_result(-500.0, {"action": 2})
        self.assertEqual(self.auto_stop.consecutive_losses, 1)

        # Another losing trade
        self.auto_stop.update_trade_result(-300.0, {"action": 2})
        self.assertEqual(self.auto_stop.consecutive_losses, 2)

    def test_drawdown_calculation(self):
        """Test drawdown calculation."""
        # Simulate trades with running P&L
        trades = [
            (1000.0, {"action": 1}),
            (500.0, {"action": 1}),
            (-2000.0, {"action": 2}),  # Large loss
            (300.0, {"action": 1}),
        ]

        for pnl, info in trades:
            self.auto_stop.update_trade_result(pnl, info)

        # Should have calculated drawdown
        self.assertIsInstance(self.auto_stop.current_drawdown, float)

    def test_volatility_stop_condition(self):
        """Test volatility-based stop condition."""
        # Set high volatility
        self.auto_stop.volatility = 0.06  # Above 5% threshold

        should_stop, reason, message = self.auto_stop._check_volatility_stop(
            self.auto_stop.stop_conditions["volatility_stop"]
        )

        self.assertTrue(should_stop)
        self.assertEqual(reason, StopReason.VOLATILITY_SPIKE)
        self.assertIn("volatility", message.lower())

    def test_drawdown_stop_condition(self):
        """Test drawdown-based stop condition."""
        # Set high drawdown
        self.auto_stop.current_drawdown = 0.12  # Above 10% threshold

        should_stop, reason, message = self.auto_stop._check_drawdown_stop(
            self.auto_stop.stop_conditions["drawdown_stop"]
        )

        self.assertTrue(should_stop)
        self.assertEqual(reason, StopReason.DRAWDOWN_LIMIT)
        self.assertIn("drawdown", message.lower())

    def test_consecutive_losses_stop(self):
        """Test consecutive losses stop condition."""
        # Set consecutive losses
        self.auto_stop.consecutive_losses = 6  # Above threshold

        should_stop, reason, message = self.auto_stop._check_consecutive_losses_stop(
            self.auto_stop.stop_conditions.get(
                "consecutive_losses_stop", Mock(threshold=5, enabled=True)
            )
        )

        self.assertTrue(should_stop)
        self.assertEqual(reason, StopReason.CONSECUTIVE_LOSSES)
        self.assertIn("consecutive", message.lower())

    def test_stop_triggering(self):
        """Test stop condition triggering."""
        # Set high volatility to trigger stop
        self.auto_stop.volatility = 0.06

        should_stop, reason, message = self.auto_stop.check_stop_conditions()

        self.assertTrue(should_stop)
        self.assertEqual(reason, StopReason.VOLATILITY_SPIKE)
        self.assertFalse(self.auto_stop.is_active)
        self.assertIsNotNone(self.auto_stop.cooldown_until)

    def test_cooldown_period(self):
        """Test cooldown period functionality."""
        # Trigger a stop
        self.auto_stop.volatility = 0.06
        self.auto_stop.check_stop_conditions()

        # Should be in cooldown
        should_stop, reason, message = self.auto_stop.check_stop_conditions()
        self.assertFalse(should_stop)
        self.assertIn("cooldown", message.lower())

    def test_resume_trading(self):
        """Test manual resume functionality."""
        # Trigger stop
        self.auto_stop.volatility = 0.06
        self.auto_stop.check_stop_conditions()

        # Try to resume immediately (should fail due to cooldown)
        result = self.auto_stop.resume_trading()
        self.assertFalse(result)
        self.assertFalse(self.auto_stop.is_active)

        # Mock cooldown expiry
        self.auto_stop.cooldown_until = datetime.now() - timedelta(seconds=1)
        result = self.auto_stop.resume_trading()
        self.assertTrue(result)
        self.assertTrue(self.auto_stop.is_active)

    def test_get_status(self):
        """Test status reporting."""
        status = self.auto_stop.get_status()

        required_keys = [
            "is_active",
            "stop_reason",
            "cooldown_until",
            "current_drawdown",
            "volatility",
            "consecutive_losses",
            "total_trades",
            "active_conditions",
        ]

        for key in required_keys:
            self.assertIn(key, status)

    def test_production_config(self):
        """Test production configuration."""
        prod_auto_stop = create_production_auto_stop()

        # Should have conservative settings
        volatility_condition = prod_auto_stop.stop_conditions["volatility_stop"]
        self.assertEqual(volatility_condition.threshold, 0.03)  # Conservative threshold

        drawdown_condition = prod_auto_stop.stop_conditions["drawdown_stop"]
        self.assertEqual(drawdown_condition.threshold, 0.05)  # Conservative threshold

    def test_disabled_conditions(self):
        """Test disabled stop conditions."""
        config = self.config.copy()
        config["volatility_stop"]["enabled"] = False

        auto_stop = AdvancedAutoStop(config)

        # Set high volatility but condition is disabled
        auto_stop.volatility = 0.06

        should_stop, reason, message = auto_stop.check_stop_conditions()
        self.assertFalse(should_stop)  # Should not stop due to disabled condition


if __name__ == "__main__":
    unittest.main()
