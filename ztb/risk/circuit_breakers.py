"""
Circuit breakers and kill switches for risk management.

This module provides emergency shutdown mechanisms and circuit breaker
patterns to protect against cascading failures and maintain system stability.
"""

import asyncio
import enum
import time
from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional

from ztb.utils.observability import get_logger

logger = get_logger(__name__)


class CircuitBreakerState(enum.Enum):
    """Circuit breaker states."""
    CLOSED = "closed"      # Normal operation
    OPEN = "open"         # Failing, requests blocked
    HALF_OPEN = "half_open"  # Testing if service recovered


@dataclass
class CircuitBreakerConfig:
    """Configuration for circuit breaker."""
    failure_threshold: int = 5  # Failures before opening
    recovery_timeout: float = 60.0  # Seconds to wait before trying again
    success_threshold: int = 3  # Successes needed to close circuit
    timeout: float = 10.0  # Request timeout in seconds


class CircuitBreaker:
    """Circuit breaker implementation."""

    def __init__(self, name: str, config: Optional[CircuitBreakerConfig] = None):
        """Initialize circuit breaker.

        Args:
            name: Name identifier for this circuit breaker
            config: Configuration settings
        """
        self.name = name
        self.config = config or CircuitBreakerConfig()
        self.state = CircuitBreakerState.CLOSED
        self.failure_count = 0
        self.success_count = 0
        self.last_failure_time = 0.0
        self._lock = asyncio.Lock()

    async def call(self, func: Callable, *args, **kwargs) -> Any:
        """Execute function through circuit breaker.

        Args:
            func: Function to execute
            *args: Positional arguments
            **kwargs: Keyword arguments

        Returns:
            Function result

        Raises:
            CircuitBreakerOpenError: If circuit is open
            Exception: Original function exception
        """
        if not await self._can_execute():
            raise CircuitBreakerOpenError(f"Circuit breaker '{self.name}' is open")

        try:
            # Execute with timeout
            result = await asyncio.wait_for(
                func(*args, **kwargs),
                timeout=self.config.timeout
            )
            await self._on_success()
            return result
        except Exception as e:
            await self._on_failure()
            raise e

    async def _can_execute(self) -> bool:
        """Check if request can be executed."""
        async with self._lock:
            if self.state == CircuitBreakerState.CLOSED:
                return True
            elif self.state == CircuitBreakerState.OPEN:
                if time.time() - self.last_failure_time >= self.config.recovery_timeout:
                    self.state = CircuitBreakerState.HALF_OPEN
                    self.success_count = 0
                    logger.info(f"Circuit breaker '{self.name}' entering half-open state")
                    return True
                return False
            else:  # HALF_OPEN
                return True

    async def _on_success(self) -> None:
        """Handle successful execution."""
        async with self._lock:
            if self.state == CircuitBreakerState.HALF_OPEN:
                self.success_count += 1
                if self.success_count >= self.config.success_threshold:
                    self.state = CircuitBreakerState.CLOSED
                    self.failure_count = 0
                    logger.info(f"Circuit breaker '{self.name}' closed - service recovered")
            else:
                # Reset failure count on success in closed state
                self.failure_count = 0

    async def _on_failure(self) -> None:
        """Handle failed execution."""
        async with self._lock:
            self.failure_count += 1
            self.last_failure_time = time.time()

            if self.state == CircuitBreakerState.HALF_OPEN:
                self.state = CircuitBreakerState.OPEN
                logger.warning(f"Circuit breaker '{self.name}' opened due to failure in half-open state")
            elif (self.state == CircuitBreakerState.CLOSED and
                  self.failure_count >= self.config.failure_threshold):
                self.state = CircuitBreakerState.OPEN
                logger.warning(f"Circuit breaker '{self.name}' opened after {self.failure_count} failures")

    def get_state(self) -> CircuitBreakerState:
        """Get current circuit breaker state."""
        return self.state

    def reset(self) -> None:
        """Reset circuit breaker to closed state."""
        self.state = CircuitBreakerState.CLOSED
        self.failure_count = 0
        self.success_count = 0
        self.last_failure_time = 0.0
        logger.info(f"Circuit breaker '{self.name}' manually reset")


class CircuitBreakerOpenError(Exception):
    """Exception raised when circuit breaker is open."""
    pass


class KillSwitch:
    """Global kill switch for emergency shutdown."""

    def __init__(self, name: str = "global"):
        """Initialize kill switch.

        Args:
            name: Name identifier for this kill switch
        """
        self.name = name
        self._killed = False
        self._reason = ""
        self._kill_time = 0.0
        self._lock = asyncio.Lock()
        self._callbacks: list[Callable[[str], None]] = []

    def kill(self, reason: str = "Emergency shutdown") -> None:
        """Activate kill switch.

        Args:
            reason: Reason for shutdown
        """
        self._killed = True
        self._reason = reason
        self._kill_time = time.time()

        logger.critical(f"Kill switch '{self.name}' activated: {reason}")

        # Notify callbacks
        for callback in self._callbacks:
            try:
                callback(reason)
            except Exception as e:
                logger.error(f"Kill switch callback failed: {e}")

    def is_killed(self) -> bool:
        """Check if kill switch is active."""
        return self._killed

    def get_reason(self) -> str:
        """Get shutdown reason."""
        return self._reason

    def get_kill_time(self) -> float:
        """Get timestamp when kill switch was activated."""
        return self._kill_time

    def reset(self) -> None:
        """Reset kill switch (use with caution)."""
        self._killed = False
        self._reason = ""
        self._kill_time = 0.0
        logger.warning(f"Kill switch '{self.name}' manually reset")

    def add_callback(self, callback: Callable[[str], None]) -> None:
        """Add callback to be called when kill switch is activated.

        Args:
            callback: Function to call with reason string
        """
        self._callbacks.append(callback)

    async def check_and_raise(self) -> None:
        """Check kill switch and raise exception if active."""
        if self._killed:
            raise KillSwitchActivatedError(f"Kill switch '{self.name}' is active: {self._reason}")


class KillSwitchActivatedError(Exception):
    """Exception raised when kill switch is active."""
    pass


class CircuitBreakerRegistry:
    """Registry for managing multiple circuit breakers."""

    def __init__(self):
        """Initialize registry."""
        self.breakers: Dict[str, CircuitBreaker] = {}
        self._lock = asyncio.Lock()

    def get_or_create(self, name: str, config: Optional[CircuitBreakerConfig] = None) -> CircuitBreaker:
        """Get existing or create new circuit breaker.

        Args:
            name: Circuit breaker name
            config: Configuration for new breaker

        Returns:
            Circuit breaker instance
        """
        if name not in self.breakers:
            self.breakers[name] = CircuitBreaker(name, config)
        return self.breakers[name]

    def get_all_states(self) -> Dict[str, CircuitBreakerState]:
        """Get states of all circuit breakers."""
        return {name: breaker.get_state() for name, breaker in self.breakers.items()}

    def reset_all(self) -> None:
        """Reset all circuit breakers."""
        for breaker in self.breakers.values():
            breaker.reset()


# Global instances
_global_kill_switch = KillSwitch("global")
_circuit_breaker_registry = CircuitBreakerRegistry()


def get_global_kill_switch() -> KillSwitch:
    """Get global kill switch instance."""
    return _global_kill_switch


def get_circuit_breaker_registry() -> CircuitBreakerRegistry:
    """Get global circuit breaker registry."""
    return _circuit_breaker_registry


def emergency_shutdown(reason: str = "Emergency shutdown initiated") -> None:
    """Activate global emergency shutdown.

    Args:
        reason: Reason for shutdown
    """
    _global_kill_switch.kill(reason)


async def check_kill_switch() -> None:
    """Check global kill switch and raise if active."""
    await _global_kill_switch.check_and_raise()


def create_api_circuit_breaker(endpoint: str) -> CircuitBreaker:
    """Create circuit breaker for API endpoint.

    Args:
        endpoint: API endpoint name

    Returns:
        Configured circuit breaker
    """
    config = CircuitBreakerConfig(
        failure_threshold=3,
        recovery_timeout=30.0,
        success_threshold=2,
        timeout=5.0
    )
    return _circuit_breaker_registry.get_or_create(f"api_{endpoint}", config)


def create_database_circuit_breaker(db_name: str) -> CircuitBreaker:
    """Create circuit breaker for database connection.

    Args:
        db_name: Database name

    Returns:
        Configured circuit breaker
    """
    config = CircuitBreakerConfig(
        failure_threshold=5,
        recovery_timeout=60.0,
        success_threshold=3,
        timeout=15.0
    )
    return _circuit_breaker_registry.get_or_create(f"db_{db_name}", config)