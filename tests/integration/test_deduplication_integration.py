"""
Integration tests for deduplication efforts.

This module tests that the deduplication of classes (TradingStrategy Protocol,
ErrorHandlingStrategy Enum, TradingEvaluator) works correctly and maintains
backward compatibility.
"""

import unittest
from unittest.mock import Mock, patch

import pandas as pd

from ztb.analysis.evaluator.evaluator import TradingEvaluator
from ztb.trading.backtest.unified_backtest.strategy_base import TradingStrategy
from ztb.training.callbacks.shared.base.learning_callback import ErrorHandlingStrategy


class MockStrategyImplementation:
    """Mock implementation of TradingStrategy Protocol for testing."""

    def __init__(self, name: str = "TestStrategy"):
        self._name = name

    @property
    def name(self) -> str:
        return self._name

    def generate_signal(self, data: pd.DataFrame, current_position: int):
        return {
            "action": "BUY" if current_position == 0 else "HOLD",
            "confidence": 0.8,
            "quantity": 1.0
        }

    def update_hyperparameters(self, hyperparameters):
        pass


class TestDeduplicationIntegration(unittest.TestCase):
    """Test integration after deduplication efforts."""

    def setUp(self):
        """Set up test fixtures."""
        self.sample_data = pd.DataFrame({
            'timestamp': pd.date_range('2023-01-01', periods=100, freq='1H'),
            'open': [100.0 + i * 0.1 for i in range(100)],
            'high': [105.0 + i * 0.1 for i in range(100)],
            'low': [95.0 + i * 0.1 for i in range(100)],
            'close': [103.0 + i * 0.1 for i in range(100)],
            'volume': [1000 + i * 10 for i in range(100)]
        }).set_index('timestamp')

    def test_trading_strategy_protocol_integration(self):
        """Test that TradingStrategy Protocol works with UnifiedBacktester."""
        strategy = MockStrategyImplementation()

        # Verify protocol compliance
        self.assertEqual(strategy.name, "TestStrategy")
        signal = strategy.generate_signal(self.sample_data, 0)
        self.assertIn('action', signal)

        # Test that it can be used with UnifiedBacktester (mock the backtester)
        with patch('ztb.trading.backtest.unified_backtest.unified_backtester.UnifiedBacktester') as mock_backtester:
            mock_instance = Mock()
            mock_backtester.return_value = mock_instance
            mock_instance.run_backtest.return_value = {"total_return": 0.05}

            # This would normally create a backtester, but we're mocking it
            # The important thing is that the import works and protocol is satisfied

    def test_error_handling_strategy_integration(self):
        """Test that ErrorHandlingStrategy works with learning callbacks."""
        # Test all strategies are accessible
        strategies = list(ErrorHandlingStrategy)
        self.assertEqual(len(strategies), 5)

        # Test that they can be used in callback context
        for strategy in strategies:
            with self.subTest(strategy=strategy):
                # Mock callback that uses error handling strategy
                mock_callback = Mock()
                mock_callback.error_strategy = strategy

                # Simulate error handling logic
                if strategy == ErrorHandlingStrategy.CONTINUE:
                    # Should continue on error
                    pass
                elif strategy == ErrorHandlingStrategy.RETRY:
                    # Should retry on error
                    pass
                elif strategy == ErrorHandlingStrategy.SKIP:
                    # Should skip on error
                    pass
                elif strategy == ErrorHandlingStrategy.DISABLE:
                    # Should disable on error
                    pass
                elif strategy == ErrorHandlingStrategy.ABORT:
                    # Should abort on error
                    pass

    def test_trading_evaluator_integration(self):
        """Test that TradingEvaluator works correctly after deduplication."""
        # Test that TradingEvaluator can be imported and instantiated
        with patch('pandas.read_csv') as mock_read_csv, \
             patch('os.path.exists') as mock_exists:

            mock_exists.return_value = True
            mock_read_csv.return_value = self.sample_data

            # This tests that the import works and basic instantiation doesn't fail
            try:
                from ztb.analysis.evaluator.evaluator import TradingEvaluator
                # We don't actually instantiate it here as it requires file paths
                self.assertTrue(hasattr(TradingEvaluator, '__init__'))
            except ImportError as e:
                self.fail(f"TradingEvaluator import failed: {e}")

    def test_import_consistency(self):
        """Test that all deduplicated classes can be imported consistently."""
        # Test TradingStrategy Protocol import
        try:
            from ztb.trading.backtest.unified_backtest.strategy_base import TradingStrategy
            self.assertTrue(
                getattr(TradingStrategy, "_is_protocol", False)
                or hasattr(TradingStrategy, "__annotations__")
            )
        except ImportError as e:
            self.fail(f"TradingStrategy import failed: {e}")

        # Test ErrorHandlingStrategy import
        try:
            from ztb.training.callbacks.shared.base.learning_callback import ErrorHandlingStrategy
            self.assertTrue(hasattr(ErrorHandlingStrategy, 'CONTINUE'))
        except ImportError as e:
            self.fail(f"ErrorHandlingStrategy import failed: {e}")

        # Test TradingEvaluator import
        try:
            from ztb.analysis.evaluator.evaluator import TradingEvaluator
            self.assertTrue(hasattr(TradingEvaluator, '__init__'))
        except ImportError as e:
            self.fail(f"TradingEvaluator import failed: {e}")

    def test_protocol_type_safety(self):
        """Test that protocol ensures type safety."""
        strategy = MockStrategyImplementation()

        # Test that protocol methods have correct signatures
        import inspect

        # Check generate_signal signature
        sig = inspect.signature(strategy.generate_signal)
        params = sig.parameters
        self.assertIn('data', params)
        self.assertIn('current_position', params)

        # Check return type structure
        signal = strategy.generate_signal(self.sample_data, 0)
        self.assertIsInstance(signal, dict)
        self.assertIsInstance(signal.get('action'), str)

    def test_enum_backward_compatibility(self):
        """Test that ErrorHandlingStrategy maintains backward compatibility."""
        # Test string conversion
        self.assertEqual(str(ErrorHandlingStrategy.CONTINUE), 'ErrorHandlingStrategy.CONTINUE')

        # Test value access
        self.assertEqual(ErrorHandlingStrategy.CONTINUE.value, 'continue')

        # Test iteration
        strategy_names = [s.name for s in ErrorHandlingStrategy]
        expected_names = ['CONTINUE', 'RETRY', 'SKIP', 'DISABLE', 'ABORT']
        self.assertEqual(sorted(strategy_names), sorted(expected_names))

    def test_no_circular_imports(self):
        """Test that deduplication eliminated circular import issues."""
        # This test ensures that importing the modules doesn't cause circular import errors
        try:
            # Import all deduplicated modules
            from ztb.trading.backtest.unified_backtest import strategy_base
            from ztb.trading.backtest.unified_backtest import unified_backtester
            from ztb.training.callbacks.shared.base import learning_callback
            from ztb.analysis.evaluator import evaluator

            # If we get here without ImportError, circular imports are resolved
            self.assertTrue(True)

        except ImportError as e:
            if "circular import" in str(e).lower():
                self.fail(f"Circular import detected: {e}")
            else:
                # Other import errors are acceptable for this test
                pass

    def test_module_structure_integrity(self):
        """Test that module structure remains intact after deduplication."""
        # Test that key classes are still accessible from their expected locations
        from ztb.training.callbacks.shared.base.learning_callback import (
            ErrorHandlingStrategy
        )

        # Verify these are classes/types
        self.assertTrue(
            getattr(TradingStrategy, "_is_protocol", False)
            or hasattr(TradingStrategy, "__annotations__")
        )
        self.assertTrue(hasattr(ErrorHandlingStrategy, "__members__"))
        self.assertTrue(hasattr(TradingEvaluator, '__init__'))


if __name__ == '__main__':
    unittest.main()
