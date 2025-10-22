"""
Tests for circuit breakers and kill switches.

This module tests circuit breaker patterns and emergency shutdown mechanisms.
"""

import asyncio
import time

import pytest

from ztb.risk.circuit_breakers import (
    CircuitBreaker,
    CircuitBreakerConfig,
    CircuitBreakerOpenError,
    CircuitBreakerRegistry,
    CircuitBreakerState,
    KillSwitch,
    get_circuit_breaker_registry,
    get_global_kill_switch,
)


class TestCircuitBreaker:
    """Test cases for CircuitBreaker class."""

    def setup_method(self):
        """Set up test fixtures."""
        self.config = CircuitBreakerConfig(
            failure_threshold=3, recovery_timeout=1.0, success_threshold=2, timeout=5.0
        )
        self.cb = CircuitBreaker("test_circuit", self.config)

    def test_init_default_config(self):
        """Test initialization with default config."""
        cb = CircuitBreaker("test")
        assert cb.name == "test"
        assert cb.state == CircuitBreakerState.CLOSED
        assert cb.failure_count == 0
        assert cb.success_count == 0
        assert cb.last_failure_time == 0.0

    def test_init_custom_config(self):
        """Test initialization with custom config."""
        cb = CircuitBreaker("test", self.config)
        assert cb.name == "test"
        assert cb.config.failure_threshold == 3
        assert cb.config.recovery_timeout == 1.0

    def test_get_state(self):
        """Test getting circuit breaker state."""
        assert self.cb.get_state() == CircuitBreakerState.CLOSED

    def test_reset(self):
        """Test resetting circuit breaker."""
        # Simulate some failures
        self.cb.failure_count = 2
        self.cb.state = CircuitBreakerState.OPEN

        self.cb.reset()

        assert self.cb.state == CircuitBreakerState.CLOSED
        assert self.cb.failure_count == 0
        assert self.cb.success_count == 0

    @pytest.mark.asyncio
    async def test_call_success(self):
        """Test successful function call."""

        async def success_func():
            return "success"

        result = await self.cb.call(success_func)

        assert result == "success"
        assert self.cb.state == CircuitBreakerState.CLOSED
        assert self.cb.failure_count == 0

    @pytest.mark.asyncio
    async def test_call_failure(self):
        """Test failed function call."""

        async def failure_func():
            raise ValueError("Test error")

        with pytest.raises(ValueError):
            await self.cb.call(failure_func)

        assert self.cb.failure_count == 1
        assert self.cb.state == CircuitBreakerState.CLOSED

    @pytest.mark.asyncio
    async def test_call_circuit_open(self):
        """Test call when circuit is open."""
        # Force circuit open and set recent failure time
        self.cb.failure_count = self.config.failure_threshold
        self.cb.state = CircuitBreakerState.OPEN
        self.cb.last_failure_time = time.time()  # Recent failure

        async def success_func():
            return "success"

        # Should raise CircuitBreakerOpenError
        with pytest.raises(CircuitBreakerOpenError):
            await self.cb.call(success_func)

    @pytest.mark.asyncio
    async def test_call_timeout(self):
        """Test call timeout."""

        async def slow_func():
            await asyncio.sleep(2.0)  # Longer than timeout
            return "done"

        config = CircuitBreakerConfig(timeout=1.0)
        cb = CircuitBreaker("timeout_test", config)

        with pytest.raises(asyncio.TimeoutError):
            await cb.call(slow_func)

        assert cb.failure_count == 1


class TestKillSwitch:
    """Test cases for KillSwitch class."""

    def setup_method(self):
        """Set up test fixtures."""
        self.ks = KillSwitch("test_kill")

    def test_init(self):
        """Test initialization."""
        assert self.ks.name == "test_kill"
        assert not self.ks.is_killed()
        assert self.ks.get_reason() == ""
        assert self.ks.get_kill_time() == 0.0

    def test_kill(self):
        """Test killing the switch."""
        reason = "Test emergency shutdown"
        self.ks.kill(reason)

        assert self.ks.is_killed()
        assert self.ks.get_reason() == reason
        assert self.ks.get_kill_time() > 0

    def test_reset(self):
        """Test resetting the kill switch."""
        self.ks.kill("Test reason")
        assert self.ks.is_killed()

        self.ks.reset()

        assert not self.ks.is_killed()
        assert self.ks.get_reason() == ""
        assert self.ks.get_kill_time() == 0.0

    def test_add_callback(self):
        """Test adding callback."""
        callback_called = []

        def test_callback(reason):
            callback_called.append(reason)

        self.ks.add_callback(test_callback)
        self.ks.kill("Callback test")

        assert "Callback test" in callback_called

    @pytest.mark.asyncio
    async def test_check_and_raise(self):
        """Test check and raise when killed."""
        self.ks.kill("Test kill")

        with pytest.raises(Exception):  # KillSwitchActivatedError
            await self.ks.check_and_raise()


class TestCircuitBreakerRegistry:
    """Test cases for CircuitBreakerRegistry class."""

    def setup_method(self):
        """Set up test fixtures."""
        self.registry = CircuitBreakerRegistry()

    def test_init(self):
        """Test initialization."""
        assert isinstance(self.registry.breakers, dict)
        assert len(self.registry.breakers) == 0

    def test_get_or_create(self):
        """Test getting or creating circuit breaker."""
        cb1 = self.registry.get_or_create("test1")
        assert isinstance(cb1, CircuitBreaker)
        assert cb1.name == "test1"

        # Should return same instance
        cb2 = self.registry.get_or_create("test1")
        assert cb1 is cb2

    def test_get_or_create_with_config(self):
        """Test getting or creating with custom config."""
        config = CircuitBreakerConfig(failure_threshold=10)
        cb = self.registry.get_or_create("test_config", config)

        assert cb.config.failure_threshold == 10

    def test_get_all_states(self):
        """Test getting all circuit breaker states."""
        cb1 = self.registry.get_or_create("test1")
        cb2 = self.registry.get_or_create("test2")

        states = self.registry.get_all_states()

        assert isinstance(states, dict)
        assert "test1" in states
        assert "test2" in states
        assert all(isinstance(state, CircuitBreakerState) for state in states.values())

    def test_reset_all(self):
        """Test resetting all circuit breakers."""
        cb1 = self.registry.get_or_create("test1")
        cb2 = self.registry.get_or_create("test2")

        # Simulate failures
        cb1.failure_count = 2
        cb2.failure_count = 3

        self.registry.reset_all()

        assert cb1.failure_count == 0
        assert cb2.failure_count == 0


class TestGlobalFunctions:
    """Test cases for global functions."""

    def test_get_global_kill_switch(self):
        """Test getting global kill switch."""
        ks = get_global_kill_switch()
        assert isinstance(ks, KillSwitch)
        assert ks.name == "global"

    def test_get_circuit_breaker_registry(self):
        """Test getting circuit breaker registry."""
        registry = get_circuit_breaker_registry()
        assert isinstance(registry, CircuitBreakerRegistry)
