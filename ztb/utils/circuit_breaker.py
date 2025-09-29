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

    pass


class CircuitBreaker:
    """Circuit breaker implementation."""

    def __init__(self, name: str, config: CircuitBreakerConfig):
        """
        Initialize circuit breaker.

        Args:
            name: Name identifier for this breaker
            config: Circuit breaker configuration
        """
        self.name = name
        self.config = config
        self.state = CircuitState.CLOSED
        self.failure_count = 0
        self.success_count = 0
        self.last_failure_time: Optional[float] = None
        self._lock = asyncio.Lock()

    async def call(self, func: Callable, *args, **kwargs) -> Any:
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
            return False

    def _should_attempt_reset(self) -> bool:
        """Check if enough time has passed to attempt reset."""
        if self.last_failure_time is None:
            return True
        return time.time() - self.last_failure_time >= self.config.recovery_timeout

    async def _on_success(self):
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

    async def _on_failure(self):
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

    def get_state(self) -> CircuitState:
        """Get current circuit breaker state."""
        return self.state

    def reset(self):
        """Manually reset circuit breaker to closed state."""
        self.state = CircuitState.CLOSED
        self.failure_count = 0
        self.success_count = 0
        self.last_failure_time = None
        logger.info(f"Circuit breaker '{self.name}' manually reset")


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
        if config is None:
            raise ValueError(
                f"Circuit breaker '{name}' not found and no config provided"
            )
        _circuit_breakers[name] = CircuitBreaker(name, config)

    return _circuit_breakers[name]


def reset_all_circuit_breakers():
    """Reset all circuit breakers to closed state."""
    for breaker in _circuit_breakers.values():
        breaker.reset()
    logger.info("All circuit breakers reset")
