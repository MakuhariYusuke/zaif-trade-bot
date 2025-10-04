"""
Fault Injection Utilities for Testing Resilience

Provides controlled failure injection for testing system resilience
and recovery mechanisms in canary harness scenarios.
"""

import asyncio
import contextlib
import time
from dataclasses import dataclass
from typing import AsyncContextManager, Callable, Dict, Optional

from ztb.utils.logging_utils import get_logger

logger = get_logger(__name__)


@dataclass
class FaultInjectionConfig:
    """Configuration for fault injection."""

    name: str
    fault_type: str
    duration_s: float
    severity: float  # 0.0 to 1.0
    expected_action: str  # 'pause', 'resume', 'continue'


class FaultInjector:
    """Manages fault injection during test execution."""

    def __init__(self):
        """Initialize fault injector."""
        self.active_faults: Dict[str, FaultInjectionConfig] = {}
        self._fault_handlers: Dict[str, Callable] = {}

    def register_handler(self, fault_type: str, handler: Callable):
        """Register a handler for a specific fault type."""
        self._fault_handlers[fault_type] = handler
        logger.info(f"Registered fault handler for type: {fault_type}")

    async def inject_fault(self, config: FaultInjectionConfig) -> AsyncContextManager:
        """
        Inject a fault according to configuration.

        Returns a context manager that handles the fault lifecycle.
        """
        return FaultContext(self, config)

    def is_fault_active(self, fault_name: str) -> bool:
        """Check if a specific fault is currently active."""
        return fault_name in self.active_faults

    def get_active_faults(self) -> list[str]:
        """Get list of currently active fault names."""
        return list(self.active_faults.keys())


class FaultContext:
    """Context manager for fault injection."""

    def __init__(self, injector: FaultInjector, config: FaultInjectionConfig):
        """Initialize fault context."""
        self.injector = injector
        self.config = config
        self.correlation_id = f"fault_{config.name}_{int(time.time())}"

    async def __aenter__(self):
        """Enter fault context - activate the fault."""
        self.injector.active_faults[self.config.name] = self.config

        logger.warning(
            f"FAULT_INJECTION_START: {self.config.name} "
            f"(type={self.config.fault_type}, duration={self.config.duration_s}s, "
            f"severity={self.config.severity}) correlation_id={self.correlation_id}"
        )

        # Apply the fault
        handler = self.injector._fault_handlers.get(self.config.fault_type)
        if handler:
            await handler(self.config)
        else:
            logger.warning(
                f"No handler registered for fault type: {self.config.fault_type}"
            )

        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Exit fault context - deactivate the fault."""
        del self.injector.active_faults[self.config.name]

        logger.info(
            f"FAULT_INJECTION_END: {self.config.name} correlation_id={self.correlation_id}"
        )


# Global fault injector instance
_fault_injector: Optional[FaultInjector] = None


def get_fault_injector() -> FaultInjector:
    """Get global fault injector instance."""
    global _fault_injector
    if _fault_injector is None:
        _fault_injector = FaultInjector()
        _register_default_handlers(_fault_injector)
    return _fault_injector


def _register_default_handlers(injector: FaultInjector):
    """Register default fault handlers."""

    async def ws_disconnect_handler(config: FaultInjectionConfig):
        """Simulate WebSocket disconnect."""
        # This would be implemented by mocking the WS connection
        logger.info(f"Simulating WS disconnect for {config.duration_s}s")
        await asyncio.sleep(config.duration_s)

    async def network_delay_handler(config: FaultInjectionConfig):
        """Simulate network delay spike."""
        delay = config.severity * 5.0  # Up to 5s delay
        logger.info(f"Injecting network delay: {delay}s")
        await asyncio.sleep(delay)

    async def data_gap_handler(config: FaultInjectionConfig):
        """Simulate data gap."""
        logger.info(f"Simulating data gap for {config.duration_s}s")
        await asyncio.sleep(config.duration_s)

    async def duplicate_ticks_handler(config: FaultInjectionConfig):
        """Simulate duplicate market data ticks."""
        logger.info("Injecting duplicate ticks")
        # This would modify the data stream

    async def slow_disk_handler(config: FaultInjectionConfig):
        """Simulate slow disk I/O."""
        logger.info(f"Simulating slow disk for {config.duration_s}s")
        await asyncio.sleep(config.duration_s)

    async def cpu_pause_handler(config: FaultInjectionConfig):
        """Simulate CPU pause (sleep)."""
        pause_time = config.severity * 2.0  # Up to 2s pause
        logger.info(f"Injecting CPU pause: {pause_time}s")
        time.sleep(pause_time)  # Blocking sleep to simulate CPU pause

    async def corrupted_checkpoint_handler(config: FaultInjectionConfig):
        """Simulate corrupted checkpoint."""
        logger.info("Injecting corrupted checkpoint")
        # This would modify checkpoint data

    async def stream_throttle_handler(config: FaultInjectionConfig):
        """Simulate stream throttling."""
        logger.info(f"Simulating stream throttle for {config.duration_s}s")
        await asyncio.sleep(config.duration_s)

    # Register all handlers
    injector.register_handler("ws_disconnect", ws_disconnect_handler)
    injector.register_handler("network_delay", network_delay_handler)
    injector.register_handler("data_gap", data_gap_handler)
    injector.register_handler("duplicate_ticks", duplicate_ticks_handler)
    injector.register_handler("slow_disk", slow_disk_handler)
    injector.register_handler("cpu_pause", cpu_pause_handler)
    injector.register_handler("corrupted_checkpoint", corrupted_checkpoint_handler)
    injector.register_handler("stream_throttle", stream_throttle_handler)


@contextlib.asynccontextmanager
async def inject_fault(
    name: str,
    fault_type: str,
    duration_s: float = 5.0,
    severity: float = 0.5,
    expected_action: str = "continue",
):
    """
    Convenience context manager for fault injection.

    Args:
        name: Fault name
        fault_type: Type of fault to inject
        duration_s: Duration in seconds
        severity: Severity level (0.0-1.0)
        expected_action: Expected system action
    """
    config = FaultInjectionConfig(
        name=name,
        fault_type=fault_type,
        duration_s=duration_s,
        severity=severity,
        expected_action=expected_action,
    )

    injector = get_fault_injector()
    async with await injector.inject_fault(config) as ctx:
        yield ctx
