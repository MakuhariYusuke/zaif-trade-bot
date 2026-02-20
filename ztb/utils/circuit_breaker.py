"""
Circuit Breaker Pattern Implementation

Provides circuit breaker functionality to prevent cascading failures
and allow graceful degradation of trading operations.
"""

import asyncio
import logging
import time
from dataclasses import dataclass
from enum import Enum
from typing import Any, Callable, Optional

logger = logging.getLogger(__name__)


class CircuitState(Enum):
    """Circuit breaker states."""

    CLOSED = "closed"  # Normal operation
    OPEN = "open"  # Failing, requests blocked
    HALF_OPEN = "half_open"  # Testing recovery


@dataclass
class CircuitBreakerConfig:
    """Configuration for circuit breaker."""

    failure_threshold: int = 5  # Failures before opening
    recovery_timeout: float = 60.0  # Seconds to wait before half-open
    success_threshold: int = 3  # Successes needed to close
    timeout: float = 10.0  # Request timeout in seconds


class CircuitBreakerOpenException(Exception):
    """Exception raised when circuit breaker is open."""


class CircuitBreaker:
    """Circuit breaker implementation."""

    def __init__(self, name: str, config: Optional[CircuitBreakerConfig] = None) -> None:
        """
        Initialize circuit breaker.

        Args:
            name: Name identifier for this breaker
            config: Circuit breaker configuration (defaults to CircuitBreakerConfig())
        """
        self.name = name
        self.config = config or CircuitBreakerConfig()
        self.state = CircuitState.CLOSED
        self.failure_count = 0
        self.success_count = 0
        self.last_failure_time: Optional[float] = None
        self._lock = asyncio.Lock()

    async def call(self, func: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
        """
        Execute function with circuit breaker protection.

        Args:
            func: Function to execute
            *args: Positional arguments for function
            **kwargs: Keyword arguments for function

        Returns:
            Function result

        Raises:
            CircuitBreakerOpenException: If circuit is open
        """
        if not await self._can_proceed():
            raise CircuitBreakerOpenException(f"Circuit breaker '{self.name}' is open")

        try:
            # Execute with timeout
            result = await asyncio.wait_for(
                func(*args, **kwargs), timeout=self.config.timeout
            )
            await self._on_success()
            return result
        except Exception as e:
            await self._on_failure()
            raise e

    async def _can_proceed(self) -> bool:
        """Check if request can proceed."""
        async with self._lock:
            if self.state == CircuitState.CLOSED:
                return True
            elif self.state == CircuitState.OPEN:
                if self._should_attempt_reset():
                    self.state = CircuitState.HALF_OPEN
                    self.success_count = 0
                    logger.info(
                        f"Circuit breaker '{self.name}' entering half-open state"
                    )
                    return True
                return False
            elif self.state == CircuitState.HALF_OPEN:
                return True
            else:
                # This should never happen if CircuitState is properly defined
                assert False, f"Unknown circuit state: {self.state}"

    def _should_attempt_reset(self) -> bool:
        """Check if enough time has passed to attempt reset."""
        if self.last_failure_time is None:
            return True
        return time.time() - self.last_failure_time >= self.config.recovery_timeout

    async def _on_success(self) -> None:
        """Handle successful operation."""
        async with self._lock:
            self.failure_count = 0

            if self.state == CircuitState.HALF_OPEN:
                self.success_count += 1
                if self.success_count >= self.config.success_threshold:
                    self.state = CircuitState.CLOSED
                    logger.info(
                        f"Circuit breaker '{self.name}' closed after successful recovery"
                    )
            # CLOSED state: no action needed

    def _on_success_sync(self) -> None:
        """Handle successful operation (synchronous version)."""
        # Create a new event loop for synchronous execution
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            loop.run_until_complete(self._on_success())
        finally:
            loop.close()
            asyncio.set_event_loop(None)

    async def _on_failure(self) -> None:
        """Handle failed operation."""
        async with self._lock:
            self.failure_count += 1
            self.last_failure_time = time.time()

            if self.state == CircuitState.HALF_OPEN:
                self.state = CircuitState.OPEN
                logger.warning(
                    f"Circuit breaker '{self.name}' reopened after failure in half-open"
                )
            elif (
                self.state == CircuitState.CLOSED
                and self.failure_count >= self.config.failure_threshold
            ):
                self.state = CircuitState.OPEN
                logger.warning(
                    f"Circuit breaker '{self.name}' opened after {self.failure_count} failures"
                )

    def _on_failure_sync(self) -> None:
        """Handle failed operation (synchronous version)."""
        # Create a new event loop for synchronous execution
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            loop.run_until_complete(self._on_failure())
        finally:
            loop.close()
            asyncio.set_event_loop(None)

    def get_state(self) -> CircuitState:
        """Get current circuit breaker state."""
        return self.state

    def reset(self) -> None:
        """Manually reset circuit breaker to closed state."""
        self.state = CircuitState.CLOSED
        self.failure_count = 0
        self.success_count = 0
        self.last_failure_time = None
        logger.info(f"Circuit breaker '{self.name}' manually reset")

    async def async_on_success(self) -> None:
        """Record a successful operation (async, awaitable).

        124#: private _on_success への直接アクセスを回避する public API.
        """
        await self._on_success()

    async def async_on_failure(self) -> None:
        """Record a failed operation (async, awaitable).

        124#: private _on_failure への直接アクセスを回避する public API.
        """
        await self._on_failure()

    def should_attempt_reset(self) -> bool:
        """Check if enough time has passed to attempt reset (public API).

        124#: private _should_attempt_reset への直接アクセスを回避.
        """
        return self._should_attempt_reset()

    def record_success(self) -> None:
        """Record a successful operation (synchronous version)."""
        try:
            # Try to create task if event loop is running
            asyncio.create_task(self._on_success())
        except RuntimeError:
            # No event loop, call synchronously
            self._on_success_sync()

    def record_failure(self) -> None:
        """Record a failed operation (synchronous version)."""
        try:
            # Try to create task if event loop is running
            asyncio.create_task(self._on_failure())
        except RuntimeError:
            # No event loop, call synchronously
            self._on_failure_sync()


# Global registry
_circuit_breakers: dict[str, CircuitBreaker] = {}


def get_circuit_breaker(
    name: str, config: Optional[CircuitBreakerConfig] = None
) -> CircuitBreaker:
    """
    Get or create circuit breaker instance.

    Args:
        name: Circuit breaker name
        config: Configuration (required for new instances)

    Returns:
        Circuit breaker instance
    """
    if name not in _circuit_breakers:
        _circuit_breakers[name] = CircuitBreaker(name, config or CircuitBreakerConfig())

    return _circuit_breakers[name]


def reset_all_circuit_breakers() -> None:
    """Reset all circuit breakers to closed state."""
    for breaker in _circuit_breakers.values():
        breaker.reset()
    logger.info("All circuit breakers reset")
