"""
Unit tests for circuit_breaker.py
"""

import asyncio
import time
import unittest

from ztb.utils.circuit_breaker import (
    CircuitBreaker,
    CircuitBreakerConfig,
    CircuitBreakerOpenException,
    CircuitState,
    get_circuit_breaker,
    reset_all_circuit_breakers,
)


class TestCircuitBreaker(unittest.TestCase):
    """Test cases for circuit breaker functionality."""

    def setUp(self):
        """Set up test fixtures."""
        self.config = CircuitBreakerConfig(
            failure_threshold=3, recovery_timeout=1.0, success_threshold=2, timeout=0.5
        )
        self.breaker = CircuitBreaker("test_breaker", self.config)

    def test_circuit_breaker_initial_state(self):
        """Test circuit breaker initial state."""
        self.assertEqual(self.breaker.name, "test_breaker")
        self.assertEqual(self.breaker.state, CircuitState.CLOSED)
        self.assertEqual(self.breaker.failure_count, 0)
        self.assertEqual(self.breaker.success_count, 0)
        self.assertIsNone(self.breaker.last_failure_time)

    async def test_successful_call(self):
        """Test successful function call."""

        async def success_func():
            return "success"

        result = await self.breaker.call(success_func)
        self.assertEqual(result, "success")
        self.assertEqual(self.breaker.state, CircuitState.CLOSED)
        self.assertEqual(self.breaker.failure_count, 0)

    async def test_failed_call(self):
        """Test failed function call."""

        async def fail_func():
            raise ValueError("Test error")

        with self.assertRaises(ValueError):
            await self.breaker.call(fail_func)

        self.assertEqual(self.breaker.state, CircuitState.CLOSED)
        self.assertEqual(self.breaker.failure_count, 1)
        self.assertIsNotNone(self.breaker.last_failure_time)

    async def test_open_circuit_after_failures(self):
        """Test circuit opens after reaching failure threshold."""

        async def fail_func():
            raise ValueError("Test error")

        # Fail multiple times
        for _ in range(self.config.failure_threshold):
            with self.assertRaises(ValueError):
                await self.breaker.call(fail_func)

        self.assertEqual(self.breaker.state, CircuitState.OPEN)
        self.assertEqual(self.breaker.failure_count, self.config.failure_threshold)

    async def test_open_circuit_blocks_calls(self):
        """Test that open circuit blocks subsequent calls."""

        async def fail_func():
            raise ValueError("Test error")

        # Open the circuit
        for _ in range(self.config.failure_threshold):
            with self.assertRaises(ValueError):
                await self.breaker.call(fail_func)

        # Next call should be blocked
        with self.assertRaises(CircuitBreakerOpenException):
            await self.breaker.call(fail_func)

    async def test_half_open_after_timeout(self):
        """Test circuit enters half-open state after recovery timeout."""

        async def fail_func():
            raise ValueError("Test error")

        # Open the circuit
        for _ in range(self.config.failure_threshold):
            with self.assertRaises(ValueError):
                await self.breaker.call(fail_func)

        self.assertEqual(self.breaker.state, CircuitState.OPEN)

        # Wait for recovery timeout
        await asyncio.sleep(self.config.recovery_timeout + 0.1)

        # Next call should attempt recovery (half-open)
        async def success_func():
            return "recovered"

        result = await self.breaker.call(success_func)
        self.assertEqual(result, "recovered")
        self.assertEqual(self.breaker.state, CircuitState.HALF_OPEN)

    async def test_half_open_failure(self):
        """Test circuit reopens on failure in half-open state."""

        async def fail_func():
            raise ValueError("Test error")

        # Open the circuit
        for _ in range(self.config.failure_threshold):
            with self.assertRaises(ValueError):
                await self.breaker.call(fail_func)

        # Wait for recovery and fail again
        await asyncio.sleep(self.config.recovery_timeout + 0.1)

        with self.assertRaises(ValueError):
            await self.breaker.call(fail_func)

        self.assertEqual(self.breaker.state, CircuitState.OPEN)

    async def test_half_open_success_recovery(self):
        """Test circuit closes after successful calls in half-open state."""

        async def fail_func():
            raise ValueError("Test error")

        # Open the circuit
        for _ in range(self.config.failure_threshold):
            with self.assertRaises(ValueError):
                await self.breaker.call(fail_func)

        # Wait for recovery
        await asyncio.sleep(self.config.recovery_timeout + 0.1)

        # Succeed required number of times
        async def success_func():
            return "success"

        for _ in range(self.config.success_threshold):
            result = await self.breaker.call(success_func)
            self.assertEqual(result, "success")

        self.assertEqual(self.breaker.state, CircuitState.CLOSED)
        self.assertEqual(self.breaker.failure_count, 0)
        self.assertEqual(self.breaker.success_count, 0)

    async def test_timeout_handling(self):
        """Test timeout handling."""

        async def slow_func():
            await asyncio.sleep(1.0)  # Longer than timeout
            return "slow"

        with self.assertRaises(asyncio.TimeoutError):
            await self.breaker.call(slow_func)

        self.assertEqual(self.breaker.failure_count, 1)

    def test_get_state(self):
        """Test getting circuit breaker state."""
        self.assertEqual(self.breaker.get_state(), CircuitState.CLOSED)

    def test_reset(self):
        """Test resetting circuit breaker."""
        # Change state
        self.breaker.state = CircuitState.OPEN
        self.breaker.failure_count = 5
        self.breaker.success_count = 2
        self.breaker.last_failure_time = time.time()

        self.breaker.reset()

        self.assertEqual(self.breaker.state, CircuitState.CLOSED)
        self.assertEqual(self.breaker.failure_count, 0)
        self.assertEqual(self.breaker.success_count, 0)
        self.assertIsNone(self.breaker.last_failure_time)

    def test_record_success_synchronous(self):
        """Test synchronous success recording."""
        # Put in half-open state
        self.breaker.state = CircuitState.HALF_OPEN
        self.breaker.success_count = 0

        self.breaker.record_success()

        # Give async task time to complete
        time.sleep(0.1)

        self.assertEqual(self.breaker.success_count, 1)

    def test_record_failure_synchronous(self):
        """Test synchronous failure recording."""
        self.breaker.failure_count = 0

        self.breaker.record_failure()

        # Give async task time to complete
        time.sleep(0.1)

        self.assertEqual(self.breaker.failure_count, 1)

    def test_get_circuit_breaker_registry(self):
        """Test circuit breaker registry."""
        # Clear registry
        from ztb.utils.circuit_breaker import _circuit_breakers

        _circuit_breakers.clear()

        # Get new breaker
        breaker1 = get_circuit_breaker("test", self.config)
        breaker2 = get_circuit_breaker("test")

        self.assertIs(breaker1, breaker2)
        self.assertEqual(breaker1.name, "test")

        # Get without config creates with defaults
        breaker3 = get_circuit_breaker("new_breaker")
        self.assertEqual(breaker3.name, "new_breaker")
        self.assertEqual(breaker3.config.failure_threshold, 5)  # default

    def test_reset_all_circuit_breakers(self):
        """Test resetting all circuit breakers."""
        # Clear registry
        from ztb.utils.circuit_breaker import _circuit_breakers

        _circuit_breakers.clear()

        # Create breakers
        breaker1 = get_circuit_breaker("breaker1", self.config)
        breaker2 = get_circuit_breaker("breaker2", self.config)

        # Modify states
        breaker1.state = CircuitState.OPEN
        breaker2.state = CircuitState.HALF_OPEN

        reset_all_circuit_breakers()

        self.assertEqual(breaker1.state, CircuitState.CLOSED)
        self.assertEqual(breaker2.state, CircuitState.CLOSED)


if __name__ == "__main__":
    unittest.main()
