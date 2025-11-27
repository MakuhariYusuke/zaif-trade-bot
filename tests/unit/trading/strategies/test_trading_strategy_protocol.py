"""
Tests for TradingStrategy Protocol implementation.

This module tests the TradingStrategy Protocol and its implementations
to ensure type safety and correct interface compliance.
"""

import unittest
from typing import Dict, Union
from unittest.mock import Mock

import pandas as pd

# Import the protocol directly to avoid torch dependency issues
from ztb.trading.backtest.unified_backtest.strategy_base import (
    TradingStrategy,
    BaseTradingStrategy,
    MLTradingStrategy,
    SignalBasedStrategy,
    validate_trading_strategy
)


class MockTradingStrategy:
    """Mock implementation of TradingStrategy Protocol for testing."""

    @property
    def name(self) -> str:
        return "MockStrategy"

    def generate_signal(
        self,
        data: pd.DataFrame,
        current_position: int
    ) -> Dict[str, Union[str, int, float, bool]]:
        return {
            "action": "BUY",
            "confidence": 0.8,
            "quantity": 1.0
        }

    def update_hyperparameters(self, hyperparameters: Dict[str, float]) -> None:
        pass


class TestTradingStrategyProtocol(unittest.TestCase):
    """Test TradingStrategy Protocol compliance."""

    def setUp(self):
        """Set up test fixtures."""
        self.sample_data = pd.DataFrame({
            'open': [100.0, 101.0, 102.0],
            'high': [105.0, 106.0, 107.0],
            'low': [95.0, 96.0, 97.0],
            'close': [103.0, 104.0, 105.0],
            'volume': [1000, 1100, 1200]
        })

    def test_protocol_compliance_mock_strategy(self):
        """Test that MockTradingStrategy implements the protocol correctly."""
        strategy = MockTradingStrategy()

        # Test name property
        self.assertEqual(strategy.name, "MockStrategy")

        # Test generate_signal method
        signal = strategy.generate_signal(self.sample_data, 0)
        self.assertIsInstance(signal, dict)
        self.assertIn('action', signal)

        # Test update_hyperparameters method
        strategy.update_hyperparameters({'param1': 1.0})

    def test_protocol_typing_compliance(self):
        """Test that protocol methods have correct type annotations."""
        import inspect

        strategy = MockTradingStrategy()

        # Check name property
        name_prop = getattr(type(strategy), 'name', None)
        self.assertIsNotNone(name_prop)

        # Check generate_signal method signature
        generate_signal = getattr(strategy, 'generate_signal')
        sig = inspect.signature(generate_signal)
        params = list(sig.parameters.keys())
        self.assertIn('data', params)
        self.assertIn('current_position', params)

        # Check update_hyperparameters method signature
        update_hyperparams = getattr(strategy, 'update_hyperparameters')
        sig = inspect.signature(update_hyperparams)
        params = list(sig.parameters.keys())
        self.assertIn('hyperparameters', params)

    def test_base_trading_strategy_implementation(self):
        """Test BaseTradingStrategy abstract class."""
        # BaseTradingStrategy should be abstract and not instantiable directly
        with self.assertRaises(TypeError):
            # This should fail because it's abstract and requires name parameter
            BaseTradingStrategy()  # type: ignore

    def test_ml_trading_strategy_structure(self):
        """Test MLTradingStrategy class structure."""
        # This is a placeholder test - actual implementation would need concrete subclass
        self.assertTrue(hasattr(MLTradingStrategy, 'name'))
        self.assertTrue(hasattr(MLTradingStrategy, 'generate_signal'))
        self.assertTrue(hasattr(MLTradingStrategy, 'update_hyperparameters'))

    def test_signal_based_strategy_structure(self):
        """Test SignalBasedStrategy class structure."""
        # This is a placeholder test - actual implementation would need concrete subclass
        self.assertTrue(hasattr(SignalBasedStrategy, 'name'))
        self.assertTrue(hasattr(SignalBasedStrategy, 'generate_signal'))
        self.assertTrue(hasattr(SignalBasedStrategy, 'update_hyperparameters'))

    def test_signal_format_validation(self):
        """Test that generated signals have required format."""
        strategy = MockTradingStrategy()
        signal = strategy.generate_signal(self.sample_data, 0)

        # Required fields
        required_fields = ['action']
        for field in required_fields:
            self.assertIn(field, signal, f"Signal missing required field: {field}")

        # Action should be a valid string
        self.assertIsInstance(signal['action'], str)
        valid_actions = ['BUY', 'SELL', 'HOLD', 'CLOSE']
        self.assertIn(signal['action'], valid_actions,
                     f"Invalid action: {signal['action']}")

    def test_protocol_validation_function(self):
        """Test the validate_trading_strategy function."""
        # Valid strategy should pass validation
        valid_strategy = MockTradingStrategy()
        self.assertTrue(validate_trading_strategy(valid_strategy))

        # Invalid strategy (missing method) should fail
        invalid_strategy = Mock()
        invalid_strategy.name = "Invalid"
        # Missing generate_signal method
        self.assertFalse(validate_trading_strategy(invalid_strategy))

        # Strategy with wrong method signature should fail
        class WrongSignatureStrategy:
            @property
            def name(self):
                return "Wrong"

            def generate_signal(self, wrong_param):
                return {}

            def update_hyperparameters(self, params):
                pass

        wrong_strategy = WrongSignatureStrategy()
        self.assertFalse(validate_trading_strategy(wrong_strategy))


if __name__ == '__main__':
    unittest.main()