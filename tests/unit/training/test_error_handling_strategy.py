"""
Tests for ErrorHandlingStrategy Enum.

This module tests the ErrorHandlingStrategy Enum and its usage
in learning callbacks to ensure proper error handling behavior.
"""

import unittest

from ztb.training.callbacks.shared.base.learning_callback import ErrorHandlingStrategy


class TestErrorHandlingStrategy(unittest.TestCase):
    """Test ErrorHandlingStrategy Enum functionality."""

    def test_enum_values(self):
        """Test that all expected enum values are defined."""
        expected_values = ['continue', 'retry', 'skip', 'disable', 'abort']

        actual_values = [strategy.value for strategy in ErrorHandlingStrategy]

        for expected in expected_values:
            self.assertIn(expected, actual_values,
                         f"Missing enum value: {expected}")

    def test_enum_members(self):
        """Test that all expected enum members exist."""
        self.assertTrue(hasattr(ErrorHandlingStrategy, 'CONTINUE'))
        self.assertTrue(hasattr(ErrorHandlingStrategy, 'RETRY'))
        self.assertTrue(hasattr(ErrorHandlingStrategy, 'SKIP'))
        self.assertTrue(hasattr(ErrorHandlingStrategy, 'DISABLE'))
        self.assertTrue(hasattr(ErrorHandlingStrategy, 'ABORT'))

    def test_enum_value_mapping(self):
        """Test that enum members map to correct string values."""
        self.assertEqual(ErrorHandlingStrategy.CONTINUE.value, 'continue')
        self.assertEqual(ErrorHandlingStrategy.RETRY.value, 'retry')
        self.assertEqual(ErrorHandlingStrategy.SKIP.value, 'skip')
        self.assertEqual(ErrorHandlingStrategy.DISABLE.value, 'disable')
        self.assertEqual(ErrorHandlingStrategy.ABORT.value, 'abort')

    def test_enum_uniqueness(self):
        """Test that all enum values are unique."""
        values = [strategy.value for strategy in ErrorHandlingStrategy]
        self.assertEqual(len(values), len(set(values)),
                        "Enum values are not unique")

    def test_enum_iteration(self):
        """Test that enum can be iterated over."""
        strategies = list(ErrorHandlingStrategy)
        self.assertEqual(len(strategies), 5)

        # Test that iteration returns enum members
        for strategy in strategies:
            self.assertIsInstance(strategy, ErrorHandlingStrategy)

    def test_enum_string_representation(self):
        """Test string representation of enum members."""
        self.assertEqual(str(ErrorHandlingStrategy.CONTINUE), 'ErrorHandlingStrategy.CONTINUE')
        self.assertEqual(
            repr(ErrorHandlingStrategy.RETRY),
            "<ErrorHandlingStrategy.RETRY: 'retry'>",
        )

    def test_enum_comparison(self):
        """Test enum member comparison."""
        self.assertEqual(ErrorHandlingStrategy.CONTINUE, ErrorHandlingStrategy.CONTINUE)
        self.assertNotEqual(ErrorHandlingStrategy.CONTINUE, ErrorHandlingStrategy.RETRY)

        self.assertTrue(ErrorHandlingStrategy.CONTINUE != ErrorHandlingStrategy.RETRY)
        self.assertFalse(ErrorHandlingStrategy.CONTINUE == ErrorHandlingStrategy.RETRY)

    def test_enum_hashable(self):
        """Test that enum members are hashable."""
        strategy_set = {ErrorHandlingStrategy.CONTINUE, ErrorHandlingStrategy.RETRY}
        self.assertEqual(len(strategy_set), 2)

        strategy_dict = {ErrorHandlingStrategy.CONTINUE: 'continue_value'}
        self.assertEqual(strategy_dict[ErrorHandlingStrategy.CONTINUE], 'continue_value')

    def test_enum_from_value(self):
        """Test creating enum from value."""
        strategy = ErrorHandlingStrategy('continue')
        self.assertEqual(strategy, ErrorHandlingStrategy.CONTINUE)

        strategy = ErrorHandlingStrategy('retry')
        self.assertEqual(strategy, ErrorHandlingStrategy.RETRY)

    def test_enum_invalid_value(self):
        """Test that invalid values raise ValueError."""
        with self.assertRaises(ValueError):
            ErrorHandlingStrategy('invalid')

        with self.assertRaises(ValueError):
            ErrorHandlingStrategy('')

    def test_enum_case_sensitivity(self):
        """Test that enum values are case sensitive."""
        with self.assertRaises(ValueError):
            ErrorHandlingStrategy('CONTINUE')  # Should be lowercase

        with self.assertRaises(ValueError):
            ErrorHandlingStrategy('Continue')


class MockCallbackWithErrorHandling:
    """Mock callback class that uses ErrorHandlingStrategy."""

    def __init__(self, error_strategy: ErrorHandlingStrategy):
        self.error_strategy = error_strategy
        self.call_count = 0
        self.error_count = 0

    def execute_with_error_handling(self):
        """Simulate callback execution with error handling."""
        self.call_count += 1

        try:
            # Simulate some operation that might fail
            if self.call_count == 1:
                raise ValueError("Simulated error")
            return "success"
        except Exception as e:
            self.error_count += 1

            if self.error_strategy == ErrorHandlingStrategy.CONTINUE:
                return "continued"
            elif self.error_strategy == ErrorHandlingStrategy.RETRY:
                if self.call_count < 3:  # Retry up to 2 times
                    return self.execute_with_error_handling()
                else:
                    return "max_retries_exceeded"
            elif self.error_strategy == ErrorHandlingStrategy.SKIP:
                return "skipped"
            elif self.error_strategy == ErrorHandlingStrategy.DISABLE:
                return "disabled"
            elif self.error_strategy == ErrorHandlingStrategy.ABORT:
                raise e
            else:
                raise ValueError(f"Unknown strategy: {self.error_strategy}")


class TestErrorHandlingStrategyIntegration(unittest.TestCase):
    """Test ErrorHandlingStrategy integration with callback system."""

    def test_continue_strategy(self):
        """Test CONTINUE strategy."""
        callback = MockCallbackWithErrorHandling(ErrorHandlingStrategy.CONTINUE)

        result = callback.execute_with_error_handling()
        self.assertEqual(result, "continued")
        self.assertEqual(callback.error_count, 1)

    def test_retry_strategy(self):
        """Test RETRY strategy."""
        callback = MockCallbackWithErrorHandling(ErrorHandlingStrategy.RETRY)

        result = callback.execute_with_error_handling()
        self.assertEqual(result, "success")  # Should succeed on retry
        self.assertEqual(callback.error_count, 1)
        self.assertEqual(callback.call_count, 2)  # Called twice: initial + retry

    def test_skip_strategy(self):
        """Test SKIP strategy."""
        callback = MockCallbackWithErrorHandling(ErrorHandlingStrategy.SKIP)

        result = callback.execute_with_error_handling()
        self.assertEqual(result, "skipped")
        self.assertEqual(callback.error_count, 1)

    def test_disable_strategy(self):
        """Test DISABLE strategy."""
        callback = MockCallbackWithErrorHandling(ErrorHandlingStrategy.DISABLE)

        result = callback.execute_with_error_handling()
        self.assertEqual(result, "disabled")
        self.assertEqual(callback.error_count, 1)

    def test_abort_strategy(self):
        """Test ABORT strategy."""
        callback = MockCallbackWithErrorHandling(ErrorHandlingStrategy.ABORT)

        with self.assertRaises(ValueError):
            callback.execute_with_error_handling()

        self.assertEqual(callback.error_count, 1)


if __name__ == '__main__':
    unittest.main()
