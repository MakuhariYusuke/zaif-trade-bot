#!/usr/bin/env python3
"""
Unit tests for reward_calculator.py logging improvements.

Tests dynamic log level control and structured logging functionality.
"""

import logging
import unittest
from unittest.mock import patch
import numpy as np

from ztb.trading.environment.components.reward_calculator import RewardCalculator
from ztb.trading.environment.utils.config import EnvironmentConfig, RewardSettings


class TestRewardCalculatorLogging(unittest.TestCase):
    """Test cases for RewardCalculator logging improvements."""

    def setUp(self):
        """Set up test fixtures."""
        self.config = EnvironmentConfig.from_dict({
            "curriculum_stage": "forced_balance",
            "logging": {
                "reward_calculator_level": "WARNING",
                "dynamic_level_control": True,
                "level_change_threshold": 1000,
            }
        })
        self.reward_settings = RewardSettings.from_dict({
            "behavior_optimization": {
                "action_balance_target": 0.8,
                "entropy_regularization": 0.01,
            }
        })
        self.initial_portfolio_value = 100000.0

        self.calculator = RewardCalculator(
            config=self.config,
            reward_settings=self.reward_settings,
            initial_portfolio_value=self.initial_portfolio_value
        )

    def test_structured_logger_initialization(self):
        """Test that structured logger is properly initialized."""
        self.assertIsNotNone(self.calculator.structured_logger)
        self.assertTrue(hasattr(self.calculator.structured_logger, 'info'))
        self.assertTrue(hasattr(self.calculator.structured_logger, 'warning'))

    def test_dynamic_log_level_control(self):
        """Test dynamic log level changes."""
        # Initial level should be WARNING
        reward_logger = logging.getLogger("ztb.trading.environment.reward")
        initial_level = reward_logger.level
        self.assertEqual(initial_level, logging.WARNING)

        # Test manual level change
        self.calculator.set_log_level("ERROR")
        self.assertEqual(reward_logger.level, logging.ERROR)

        # Test invalid level (should not change)
        self.calculator.set_log_level("INVALID")
        self.assertEqual(reward_logger.level, logging.ERROR)

    def test_dynamic_logging_evaluation(self):
        """Test dynamic logging evaluation based on training progress."""
        # Early stage - should keep WARNING level
        self.calculator._evaluate_dynamic_logging(1000)
        reward_logger = logging.getLogger("ztb.trading.environment.reward")
        self.assertEqual(reward_logger.level, logging.WARNING)

        # Later stage - should reduce logging
        self.calculator._evaluate_dynamic_logging(60000)
        # Note: This test may need adjustment based on actual threshold logic

    def test_log_evaluation_counter(self):
        """Test that log evaluation counter works properly."""
        initial_counter = self.calculator._log_evaluation_counter

        # Call evaluation multiple times
        for i in range(5):
            self.calculator._evaluate_dynamic_logging(1000 + i * 100)

        # Counter should have increased
        self.assertGreater(self.calculator._log_evaluation_counter, initial_counter)

    @patch('time.time')
    def test_structured_logging_format(self, mock_time):
        """Test structured logging output format."""
        mock_time.return_value = 1234567890.0

        with patch('builtins.print') as mock_print:
            # This would normally output JSON, but we're testing the structure
            self.calculator.structured_logger.info(
                "Test message",
                extra={"test_key": "test_value", "step": 1000}
            )

            # Verify the logger was called (actual output format testing would require more setup)
            # The structured logger should handle JSON formatting internally

    def test_reward_calculation_with_dynamic_logging(self):
        """Test that reward calculation triggers dynamic logging evaluation."""
        # Mock observation
        observation = np.array([1.0, 2.0, 3.0])

        # Calculate reward - this should trigger dynamic logging evaluation
        initial_counter = self.calculator._log_evaluation_counter

        reward = self.calculator.calculate_reward(
            action=1,
            current_price=100.0,
            position=0.5,
            portfolio_value=100000.0,
            atr=1.0,
            transaction_cost=0.0,
            reward_scaling=1.0,
            pnl=100.0,
            old_position=0.0,
            step=1000,
            observation=observation,
            reward_history=[0.0, 0.0],
            portfolio_value_history=[100000.0, 100000.0]
        )

        # Counter should have increased due to dynamic logging evaluation
        self.assertGreater(self.calculator._log_evaluation_counter, initial_counter)
        self.assertIsInstance(reward, float)

    def test_log_level_bounds_checking(self):
        """Test that log level setting handles edge cases."""
        reward_logger = logging.getLogger("ztb.trading.environment.reward")

        # Test valid levels
        valid_levels = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
        for level in valid_levels:
            self.calculator.set_log_level(level)
            expected_level = getattr(logging, level)
            self.assertEqual(reward_logger.level, expected_level)

        # Test case insensitive
        self.calculator.set_log_level("debug")
        self.assertEqual(reward_logger.level, logging.DEBUG)

    def test_dynamic_logging_disabled(self):
        """Test behavior when dynamic logging is disabled."""
        # Create calculator with dynamic logging disabled
        config = EnvironmentConfig.from_dict({
            "curriculum_stage": "forced_balance",
            "logging": {
                "reward_calculator_level": "WARNING",
                "dynamic_level_control": False,
            }
        })

        calculator = RewardCalculator(
            config=config,
            reward_settings=self.reward_settings,
            initial_portfolio_value=self.initial_portfolio_value
        )

        # Dynamic evaluation should not change level
        initial_level = logging.getLogger("ztb.trading.environment.reward").level
        calculator._evaluate_dynamic_logging(100000)
        final_level = logging.getLogger("ztb.trading.environment.reward").level

        self.assertEqual(initial_level, final_level)


class TestRewardCalculatorLoggingConfiguration(unittest.TestCase):
    """Test cases for RewardCalculator logging configuration."""

    def test_logging_configuration_defaults(self):
        """Test default logging configuration."""
        config = EnvironmentConfig.from_dict({
            "curriculum_stage": "forced_balance"
        })

        calculator = RewardCalculator(
            config=config,
            reward_settings=RewardSettings.from_dict({}),
            initial_portfolio_value=100000.0
        )

        # Should have default values
        self.assertTrue(calculator._dynamic_logging_enabled)
        self.assertEqual(calculator._log_level_change_threshold, 1000)
        self.assertEqual(calculator._current_log_level, logging.WARNING)

    def test_custom_logging_configuration(self):
        """Test custom logging configuration."""
        config = EnvironmentConfig.from_dict({
            "curriculum_stage": "forced_balance",
            "logging": {
                "reward_calculator_level": "INFO",
                "dynamic_level_control": False,
                "level_change_threshold": 500,
            }
        })

        calculator = RewardCalculator(
            config=config,
            reward_settings=RewardSettings.from_dict({}),
            initial_portfolio_value=100000.0
        )

        # Should use custom values
        self.assertFalse(calculator._dynamic_logging_enabled)
        self.assertEqual(calculator._log_level_change_threshold, 500)
        reward_logger = logging.getLogger("ztb.trading.environment.reward")
        self.assertEqual(reward_logger.level, logging.INFO)


if __name__ == "__main__":
    unittest.main()
