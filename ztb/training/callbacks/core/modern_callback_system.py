#!/usr/bin/env python3
"""
Modern Callback System for Training.

This module provides a comprehensive, event-driven callback system that supports:
- Event-driven architecture with multiple callback types
- Plugin-based callbacks with easy registration
- Callback chaining and prioritization
- Async callback support
- Comprehensive error handling and logging
- Metrics collection and reporting
- Configuration-driven callback setup
"""

import asyncio
import logging
import threading
import time
from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Protocol


class CallbackEvent(Enum):
    """Training callback events."""

    TRAINING_START = "training_start"
    TRAINING_END = "training_end"
    EPOCH_START = "epoch_start"
    EPOCH_END = "epoch_end"
    STEP_START = "step_start"
    STEP_END = "step_end"
    EVALUATION_START = "evaluation_start"
    EVALUATION_END = "evaluation_end"
    CHECKPOINT_SAVE = "checkpoint_save"
    METRICS_UPDATE = "metrics_update"
    ERROR_OCCURRED = "error_occurred"


class CallbackPriority(Enum):
    """Callback execution priority."""

    HIGHEST = 0
    HIGH = 1
    NORMAL = 2
    LOW = 3
    LOWEST = 4


@dataclass
class CallbackContext:
    """Context information passed to callbacks."""

    event: CallbackEvent = CallbackEvent.TRAINING_START
    step: int = 0
    epoch: int = 0
    total_steps: int = 0
    total_epochs: int = 0
    metrics: Dict[str, Any] = field(default_factory=dict)
    model_info: Dict[str, Any] = field(default_factory=dict)
    environment_info: Dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)
    custom_data: Dict[str, Any] = field(default_factory=dict)


class CallbackResult:
    """Result of callback execution."""

    def __init__(
        self,
        success: bool,
        data: Optional[Any] = None,
        error: Optional[Exception] = None,
    ):
        self.success = success
        self.data = data
        self.error = error
        self.execution_time = 0.0


class CallbackProtocol(Protocol):
    """Protocol for callback functions."""

    def __call__(self, context: CallbackContext) -> Optional[CallbackResult]:
        ...


class AsyncCallbackProtocol(Protocol):
    """Protocol for async callback functions."""

    async def __call__(self, context: CallbackContext) -> Optional[CallbackResult]:
        ...


@dataclass
class CallbackConfig:
    """Configuration for a callback."""

    name: str
    enabled: bool = True
    priority: CallbackPriority = CallbackPriority.NORMAL
    events: List[CallbackEvent] = field(default_factory=list)
    config: Dict[str, Any] = field(default_factory=dict)
    async_enabled: bool = False
    timeout: Optional[float] = None


class BaseCallback(ABC):
    """Base class for all callbacks."""

    def __init__(self, config: Optional[CallbackConfig] = None):
        self.config = config or CallbackConfig(name=self.__class__.__name__)
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        self._execution_count = 0
        self._total_execution_time = 0.0
        self._error_count = 0

    @property
    def name(self) -> str:
        """Get callback name."""
        return self.config.name

    @abstractmethod
    def on_training_start(self, context: CallbackContext) -> Optional[CallbackResult]:
        """Called when training starts."""
        pass

    @abstractmethod
    def on_training_end(self, context: CallbackContext) -> Optional[CallbackResult]:
        """Called when training ends."""
        pass

    def on_epoch_start(self, context: CallbackContext) -> Optional[CallbackResult]:
        """Called when an epoch starts."""
        return None

    def on_epoch_end(self, context: CallbackContext) -> Optional[CallbackResult]:
        """Called when an epoch ends."""
        return None

    def on_step_start(self, context: CallbackContext) -> Optional[CallbackResult]:
        """Called before each training step."""
        return None

    def on_step_end(self, context: CallbackContext) -> Optional[CallbackResult]:
        """Called after each training step."""
        return None

    def on_evaluation_start(self, context: CallbackContext) -> Optional[CallbackResult]:
        """Called when evaluation starts."""
        return None

    def on_evaluation_end(self, context: CallbackContext) -> Optional[CallbackResult]:
        """Called when evaluation ends."""
        return None

    def on_checkpoint_save(self, context: CallbackContext) -> Optional[CallbackResult]:
        """Called when a checkpoint is saved."""
        return None

    def on_metrics_update(self, context: CallbackContext) -> Optional[CallbackResult]:
        """Called when metrics are updated."""
        return None

    def on_error(self, context: CallbackContext) -> Optional[CallbackResult]:
        """Called when an error occurs."""
        return None

    def get_stats(self) -> Dict[str, Any]:
        """Get callback execution statistics."""
        return {
            "execution_count": self._execution_count,
            "total_execution_time": self._total_execution_time,
            "average_execution_time": self._total_execution_time
            / max(1, self._execution_count),
            "error_count": self._error_count,
            "error_rate": self._error_count / max(1, self._execution_count),
        }


class CallbackManager:
    """Manager for training callbacks with event-driven architecture."""

    def __init__(self, logger: Optional[logging.Logger] = None):
        self.logger = logger or logging.getLogger(__name__)
        self.callbacks: Dict[str, BaseCallback] = {}
        self.event_handlers: Dict[
            CallbackEvent, List[tuple[BaseCallback, CallbackPriority]]
        ] = {}
        self.async_executor = ThreadPoolExecutor(
            max_workers=4, thread_name_prefix="callback"
        )
        self._lock = threading.RLock()
        self._stats = {
            "total_callbacks": 0,
            "successful_callbacks": 0,
            "failed_callbacks": 0,
            "total_execution_time": 0.0,
        }

    def register_callback(self, callback: BaseCallback) -> None:
        """Register a callback."""
        with self._lock:
            if callback.name in self.callbacks:
                self.logger.warning(
                    f"Callback '{callback.name}' already registered, replacing"
                )

            self.callbacks[callback.name] = callback

            # Register event handlers based on callback configuration
            for event in callback.config.events:
                if event not in self.event_handlers:
                    self.event_handlers[event] = []
                # Ensure priority is a CallbackPriority enum
                priority = callback.config.priority
                if isinstance(priority, int):
                    priority = CallbackPriority(priority)
                self.event_handlers[event].append((callback, priority))
                # Sort by priority
                self.event_handlers[event].sort(key=lambda x: x[1].value)

            self.logger.info(f"Registered callback: {callback.name}")

    def unregister_callback(self, name: str) -> bool:
        """Unregister a callback."""
        with self._lock:
            if name not in self.callbacks:
                return False

            self.callbacks[name]

            # Remove from event handlers
            for event_handlers in self.event_handlers.values():
                event_handlers[:] = [
                    (cb, prio) for cb, prio in event_handlers if cb.name != name
                ]

            del self.callbacks[name]
            self.logger.info(f"Unregistered callback: {name}")
            return True

    def trigger_event(
        self, event: CallbackEvent, context: CallbackContext
    ) -> List[CallbackResult]:
        """Trigger a callback event."""
        if event not in self.event_handlers:
            return []

        handlers = self.event_handlers[event]
        if not handlers:
            return []

        results = []
        start_time = time.time()

        for callback, _ in handlers:
            if not callback.config.enabled:
                continue

            try:
                result = self._execute_callback(callback, event, context)
                if result:
                    results.append(result)
                    if result.success:
                        self._stats["successful_callbacks"] += 1
                    else:
                        self._stats["failed_callbacks"] += 1
                        self._stats["total_execution_time"] += result.execution_time

            except Exception as e:
                self.logger.error(
                    f"Error executing callback {callback.name} for event {event.value}: {e}"
                )
                error_result = CallbackResult(success=False, error=e)
                error_result.execution_time = time.time() - start_time
                results.append(error_result)
                self._stats["failed_callbacks"] += 1

        self._stats["total_callbacks"] += len(results)
        return results

    def get_statistics(self) -> Dict[str, Any]:
        """Get callback execution statistics."""
        return self._stats.copy()

    async def trigger_event_async(
        self, event: CallbackEvent, context: CallbackContext
    ) -> List[CallbackResult]:
        """Trigger a callback event asynchronously."""
        if event not in self.event_handlers:
            return []

        handlers = self.event_handlers[event]
        if not handlers:
            return []

        tasks = []
        for callback, _ in handlers:
            if not callback.config.enabled or not callback.config.async_enabled:
                continue

            task = asyncio.create_task(
                self._execute_callback_async(callback, event, context)
            )
            tasks.append(task)

        if not tasks:
            return []

        results = await asyncio.gather(*tasks, return_exceptions=True)
        return [r for r in results if isinstance(r, CallbackResult)]

    def _execute_callback(
        self, callback: BaseCallback, event: CallbackEvent, context: CallbackContext
    ) -> Optional[CallbackResult]:
        """Execute a callback for a specific event."""
        method_name = f"on_{event.value}"
        method = getattr(callback, method_name, None)

        if not method:
            return None

        start_time = time.time()
        try:
            result = method(context)
            execution_time = time.time() - start_time

            if result is None:
                result = CallbackResult(success=True)
            result.execution_time = execution_time

            callback._execution_count += 1
            callback._total_execution_time += execution_time

            return result

        except Exception as e:
            execution_time = time.time() - start_time
            callback._error_count += 1

            result = CallbackResult(success=False, error=e)
            result.execution_time = execution_time
            return result

    async def _execute_callback_async(
        self, callback: BaseCallback, event: CallbackEvent, context: CallbackContext
    ) -> CallbackResult:
        """Execute a callback asynchronously."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            self.async_executor, self._execute_callback, callback, event, context
        )

    def get_stats(self) -> Dict[str, Any]:
        """Get callback manager statistics."""
        return dict(self._stats)

    def list_callbacks(self) -> List[str]:
        """List registered callback names."""
        return list(self.callbacks.keys())

    def get_callback(self, name: str) -> Optional[BaseCallback]:
        """Get a callback by name."""
        return self.callbacks.get(name)

    def enable_callback(self, name: str) -> bool:
        """Enable a callback."""
        callback = self.callbacks.get(name)
        if callback:
            callback.config.enabled = True
            return True
        return False

    def disable_callback(self, name: str) -> bool:
        """Disable a callback."""
        callback = self.callbacks.get(name)
        if callback:
            callback.config.enabled = False
            return True
        return False

    def shutdown(self) -> None:
        """Shutdown the callback manager."""
        self.async_executor.shutdown(wait=True)
        self.logger.info("Callback manager shutdown complete")


# Convenience functions for creating common callbacks
def create_progress_callback(
    name: str = "progress", log_interval: int = 100
) -> BaseCallback:
    """Create a progress monitoring callback."""
    pass  # Implementation will be added


def create_checkpoint_callback(
    name: str = "checkpoint",
    save_interval: int = 1000,
    save_path: str = "./checkpoints",
) -> BaseCallback:
    """Create a checkpoint saving callback."""
    pass  # Implementation will be added


def create_metrics_callback(
    name: str = "metrics", metrics_interval: int = 50
) -> BaseCallback:
    """Create a metrics collection callback."""
    pass  # Implementation will be added


def create_logging_callback(
    name: str = "logging", log_level: str = "INFO"
) -> BaseCallback:
    """Create a logging callback."""
    pass  # Implementation will be added
