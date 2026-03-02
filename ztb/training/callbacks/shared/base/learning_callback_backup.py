#!/usr/bin/env python3
"""
Shared Base Classes for Learning Callbacks.

This module provides base classes and interfaces that are shared
across different learning types (reinforcement, supervised, etc.).
"""

from __future__ import annotations

import abc
import functools
import logging
import threading
import time
import traceback
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Callable, Optional

from ztb.training.callbacks.monitoring.metrics_collector import MetricsCollector
from ztb.training.callbacks.performance.memory_optimizer import LRUCache
from ztb.training.callbacks.shared.base.learning_callback import ErrorHandlingStrategy

class ErrorSeverity(Enum):
    """Error severity levels."""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

# Legacy ErrorContext/handler removed; use the canonical
# dataclass-based `ErrorContext` and `ErrorHandlingMixin` defined below.

    def _should_skip_execution(self) -> bool:
        """Determine if execution should be skipped due to excessive errors."""
        threshold = self._error_config.get("error_threshold_before_disable", 5)
        return self._consecutive_errors >= threshold

    def get_error_stats(self) -> dict[str, Any]:
        """Get error statistics."""
        with self._error_lock:
            return {
                "total_errors": len(self._error_history),
                "consecutive_errors": self._consecutive_errors,
                "error_counts_by_method": self._error_counts.copy(),
                "recent_errors": [
                    {
                        "method": err.method_name,
                        "error_type": type(err.error).__name__,
                        "severity": err.severity.value,
                        "timestamp": err.timestamp.isoformat(),
                        "retry_count": err.retry_count,
                    }
                    for err in self._error_history[-10:]  # Last 10 errors
                ],
                "error_config": self._error_config.copy(),
            }

    def reset_error_state(self) -> None:
        """Reset error state for recovery."""
        with self._error_lock:
            self._consecutive_errors = 0
            self.logger.info("Error state reset for recovery")

    def enable_error_recovery(self) -> None:
        """Enable error recovery mode."""
        with self._error_lock:
            self._is_recovering = True
            self.reset_error_state()
            self.logger.info("Error recovery mode enabled")

    def disable_error_recovery(self) -> None:
        """Disable error recovery mode."""
        with self._error_lock:
            self._is_recovering = False
            self.logger.info("Error recovery mode disabled")

@dataclass
class LearningContext:
    """
    Context information for learning callbacks.

    epoch: int = 0
    total_epochs: int = 0
    batch: int = 0
    step: int = 0
    # Performance metrics
    metrics: dict[str, Any] = field(default_factory=dict)

    # Custom data
    custom_data: dict[str, Any] = field(default_factory=dict)

class ErrorHandlingMixin:
    """
    Mixin class providing comprehensive error handling capabilities.

    Features:
    - Configurable error handling strategies
    - Automatic retry mechanisms
    - Enhanced logging with context
    - Graceful degradation
    - Error recovery patterns
    """
    """

    def safe_execute(self, func: Callable, *args, **kwargs) -> Any:
        """
        Execute a function with comprehensive error handling.

        Args:
            func: Function to execute
        """
        method_name = getattr(func, "__name__", str(func))
        callback_name = self.__class__.__name__

        try:
            # Check if we should skip due to too many errors
            if self._should_skip_execution():
                self.logger.warning(
                    f"Skipping {callback_name}.{method_name} due to excessive errors"
                )
                return None

            # Execute the function
            result = func(*args, **kwargs)

            # Reset consecutive error counter on success
            with self._error_lock:
                self._consecutive_errors = 0

            return result

        except Exception as e:
            return self._handle_error(e, method_name, callback_name, *args, **kwargs)

    def _handle_error(
        self, error: Exception, method_name: str, callback_name: str, *args, **kwargs
    ) -> Any:
        """Handle an error with appropriate strategy."""
        with self._error_lock:
            self._consecutive_errors += 1

            # Create error context
            error_context = ErrorContext(
                callback_name=callback_name,
                method_name=method_name,
                learning_context=getattr(self, "_last_context", LearningContext()),
                error=error,
                timestamp=datetime.now(),
                retry_count=0,
                max_retries=self._error_config["max_retries"],
            )

            # Determine error severity and strategy
            error_context.severity = self._assess_error_severity(error)
            error_context.strategy = self._determine_error_strategy(error_context)

            # Log the error with enhanced context
            self._log_error(error_context)

            # Record error in history
            self._error_history.append(error_context)
            self._error_counts[method_name] = self._error_counts.get(method_name, 0) + 1

            # Keep history bounded
            if len(self._error_history) > 100:
                self._error_history.pop(0)

            # Execute error handling strategy
            return self._execute_error_strategy(error_context, *args, **kwargs)

    def _assess_error_severity(self, error: Exception) -> ErrorSeverity:
        """Assess the severity of an error."""
        error_type = type(error).__name__

        # Critical errors that should abort training
        if error_type in ["MemoryError", "SystemExit", "KeyboardInterrupt"]:
            return ErrorSeverity.CRITICAL

        # High severity errors
        if error_type in ["ValueError", "TypeError", "AttributeError", "ImportError"]:
            return ErrorSeverity.HIGH

        # Check for error type specific strategy
        error_type = type(error_context.error).__name__
        error_strategy = strategy_config.get(error_type)
        if error_strategy:
            return error_strategy

        # Check severity-based strategy
        if error_context.severity == ErrorSeverity.CRITICAL:
            return ErrorHandlingStrategy.ABORT
        elif error_context.severity == ErrorSeverity.HIGH:
            # If too many consecutive errors, disable callback
            if (
                self._consecutive_errors
                >= self._error_config["error_threshold_before_disable"]
            if hasattr(self, "enabled"):
                self.enabled = False
            return None

        elif strategy == ErrorHandlingStrategy.ABORT:
            self.logger.critical(
                f"Aborting training due to critical error in {error_context.callback_name}.{error_context.method_name}"
            )
            raise error_context.error

        return None

    def _retry_operation(self, error_context: ErrorContext, *args, **kwargs) -> Any:
        """Retry an operation with exponential backoff and circuit breaker pattern."""
        max_retries = error_context.max_retries
        base_delay = self._error_config["retry_delay"]
        max_delay = self._error_config.get("max_retry_delay", 60.0)

        # Circuit breaker check
        if self._should_open_circuit(error_context):
            self.logger.warning(
                f"Circuit breaker open for {error_context.callback_name}.{error_context.method_name}"
            )
            return None

        for attempt in range(max_retries):
            try:
                error_context.retry_count = attempt + 1

                # Calculate delay with jitter
                delay = min(base_delay * (2**attempt), max_delay)
                if attempt > 0:
                    # Add jitter to prevent thundering herd
                    import random

                    jitter = random.uniform(0.1, 1.0)
                    delay *= jitter

                    self.logger.info(
                        f"Retrying {error_context.callback_name}.{error_context.method_name} "
                        f"(attempt {attempt + 1}/{max_retries}) after {delay:.2f}s delay"
                    )
                    time.sleep(delay)

                # Try to find and call the original method
                method = getattr(self, error_context.method_name, None)
                if method and callable(method):
                    # Store context for error handling
                    self._last_context = error_context.learning_context

                    result = method(*args, **kwargs)

                    # Success - reset circuit breaker
                    self._reset_circuit_breaker(error_context)
                    return result
                else:
                    raise AttributeError(
                        f"Method {error_context.method_name} not found"
                    )

            except Exception as retry_error:
                error_context.error = retry_error

                # Enhanced error logging for retries
                self._log_retry_error(error_context, attempt + 1, max_retries)

                # Check if this is a retryable error
                if not self._is_retryable_error(retry_error):
                    self.logger.warning(
                        f"Non-retryable error in {error_context.callback_name}.{error_context.method_name}: {retry_error}"
                    )
                    break

                # Update circuit breaker on failure
                self._record_circuit_failure(error_context)

                if attempt == max_retries - 1:
                    self.logger.error(
                        f"Failed to retry {error_context.callback_name}.{error_context.method_name} "
                        f"after {max_retries} attempts"
                    )
                    # Could fall back to degraded mode here
                    return self._fallback_operation(error_context, *args, **kwargs)

        return None

    def _is_retryable_error(self, error: Exception) -> bool:
        """Determine if an error is retryable."""
        non_retryable_errors = [
            "ValueError",
            "TypeError",
            "AttributeError",
            "ImportError",
            "SyntaxError",
            "IndentationError",
            "NameError",
        ]

        error_type = type(error).__name__

        # Memory errors are generally not retryable
        if error_type == "MemoryError":
            return False

        # Configuration errors are not retryable
        if "config" in str(error).lower() or "configuration" in str(error).lower():
            return False

        # System-level errors might be retryable
        if error_type in ["OSError", "IOError", "ConnectionError", "TimeoutError"]:
            return True

        # Default: retry if not in non-retryable list
        return error_type not in non_retryable_errors

    def _fallback_operation(self, error_context: ErrorContext, *args, **kwargs) -> Any:
        """Execute fallback operation when retries are exhausted."""
        self.logger.warning(
            f"Executing fallback for {error_context.callback_name}.{error_context.method_name}"
        )

        # Try to find a fallback method
        fallback_method_name = f"{error_context.method_name}_fallback"
        fallback_method = getattr(self, fallback_method_name, None)

        if fallback_method and callable(fallback_method):
            try:
                return fallback_method(*args, **kwargs)
            except Exception as fallback_error:
                self.logger.error(f"Fallback operation also failed: {fallback_error}")

        # Default fallback: return None and continue
        return None

    def _should_open_circuit(self, error_context: ErrorContext) -> bool:
        """Check if circuit breaker should be opened."""
        # Simple circuit breaker implementation
        recent_errors = [
            err
            for err in self._error_history[-10:]
            if err.method_name == error_context.method_name
            and (datetime.now() - err.timestamp).seconds < 300
        ]  # Last 5 minutes

                datetime.now() - error_context.timestamp
            ).total_seconds()
            error_msg += f" ({time_since_start:.2f}s elapsed)"

        self.logger.warning(error_msg)

    def _log_error(self, error_context: ErrorContext) -> None:
        """Log an error with enhanced context information and structured logging."""
        callback_name = error_context.callback_name
        method_name = error_context.method_name

        # Determine log level based on severity and error count
        log_level = {
            ErrorSeverity.LOW: logging.DEBUG,
            ErrorSeverity.MEDIUM: logging.WARNING,
            ErrorSeverity.HIGH: logging.ERROR,
            ErrorSeverity.CRITICAL: logging.CRITICAL,
        }.get(error_context.severity, logging.ERROR)

        # Build structured error message
        error_parts = [
            f"[{error_context.severity.value.upper()}]",
            f"{callback_name}.{method_name}",
            f"{type(error_context.error).__name__}: {error_context.error}",
        ]

        # Add context information
        context_parts = []
        ctx = error_context.learning_context
        if hasattr(ctx, "epoch") and ctx.epoch is not None:
            context_parts.append(f"epoch={ctx.epoch}")
        if hasattr(ctx, "batch") and ctx.batch is not None:
            context_parts.append(f"batch={ctx.batch}")
        if hasattr(ctx, "step") and ctx.step is not None:
            context_parts.append(f"step={ctx.step}")
        if hasattr(ctx, "learning_type") and ctx.learning_type:
            context_parts.append(f"type={ctx.learning_type}")
        if hasattr(ctx, "algorithm_name") and ctx.algorithm_name:
            context_parts.append(f"algo={ctx.algorithm_name}")

        if context_parts:
            error_parts.append(f"[Context: {', '.join(context_parts)}]")

        # Add retry information
        if error_context.retry_count > 0:
            error_parts.append(
                f"[Retry: {error_context.retry_count}/{error_context.max_retries}]"
            )

        # Log the main error message
        self.logger.log(log_level, error_msg)

        # Log additional structured information at debug level
        if self.logger.isEnabledFor(logging.DEBUG):
            debug_info = {
                "callback": callback_name,
                "method": method_name,
                "error_type": type(error_context.error).__name__,
                "error_message": str(error_context.error),
                "severity": error_context.severity.value,
                "timestamp": error_context.timestamp.isoformat(),
                "retry_count": error_context.retry_count,
                "max_retries": error_context.max_retries,
                "context": {
                    "step": getattr(ctx, "step", None),
                    "learning_type": getattr(ctx, "learning_type", None),
                    "algorithm_name": getattr(ctx, "algorithm_name", None),
                    "loss": getattr(ctx, "loss", None),
                },
            }

            self.logger.debug(f"Structured error info: {debug_info}")

            cleaned_stack = "\n".join(stack_lines[1:]).rstrip()

            self.logger.debug(
                f"Stack trace for {callback_name}.{method_name}:\n{cleaned_stack}"
            )

        # Log error pattern analysis if we have enough history
        if len(self._error_history) >= 3:
            self._log_error_pattern_analysis(error_context)

    def _log_error_pattern_analysis(self, error_context: ErrorContext) -> None:
        """Log analysis of error patterns for debugging."""
        recent_errors = self._error_history[-10:]

        # Analyze error frequency
        error_types = {}
        method_errors = {}

        for err in recent_errors:
        self.safe_execute(_cache_metrics_operation)

    def get_cached_metrics(self, key: str) -> dict[str, Any] | None:
        """Retrieve cached metrics with error handling."""

        def _get_metrics_operation():
            return self.metrics_cache.get(key)

        return self.safe_execute(_get_metrics_operation)

class MetricsCallback(LearningCallback):
    """
    Base class for callbacks that collect and report metrics.

    Provides integration with the metrics collection system.
    """

    def __init__(self, metrics_collector: MetricsCollector | None = None):
        super().__init__()
        self.metrics_collector = metrics_collector

    def record_metric(
        self,
        name: str,
        value: int | float,
        tags: dict[str, str] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Record a metric value."""
        if self.metrics_collector:
            self.metrics_collector.add_metric_value(name, value, tags, metadata)

    def get_metrics_summary(self) -> dict[str, Any]:
        """Get summary of collected metrics."""
        if self.metrics_collector:
            return self.metrics_collector.get_latest_metrics()
        return {}

class AdaptiveCallback(LearningCallback):
    """
    Base class for adaptive callbacks.

    Provides capabilities for callbacks that adapt their behavior
    based on training progress and performance.
    """

    def __init__(self, adaptation_frequency: int = 10):
        super().__init__()
        self.adaptation_frequency = adaptation_frequency
        self.adaptation_history: list[dict[str, Any]] = []

    def should_adapt(self, context: LearningContext) -> bool:
        """Determine if adaptation should occur."""
        return context.epoch % self.adaptation_frequency == 0

    def adapt(
        self, context: LearningContext, logs: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        """Perform adaptation and return adaptation details."""
        adaptation_info = {
            "epoch": context.epoch,
            "timestamp": datetime.now().isoformat(),
            "context": context.__dict__,
            "logs": logs or {},
        }

        self.adaptation_history.append(adaptation_info)

        # Keep history bounded

        self.callbacks.append(callback)
        # Sort by priority (higher priority first)
        self.callbacks.sort(key=lambda cb: cb.priority, reverse=True)
        self.logger.info(
            f"Added callback: {callback.__class__.__name__} (priority: {callback.priority})"
        )

    def remove_callback(self, callback_class: type) -> bool:
        """Remove a callback by class type."""
        for i, callback in enumerate(self.callbacks):
            if isinstance(callback, callback_class):
                removed = self.callbacks.pop(i)
                self.logger.info(f"Removed callback: {removed.__class__.__name__}")
                return True
        return False

    def get_callbacks(self) -> list[LearningCallback]:
        """Get all registered callbacks."""
        return self.callbacks.copy()

    def call_method(
        self,
        method_name: str,
        context: LearningContext,
        logs: dict[str, Any] | None = None,
        timeout: float | None = None,
    ) -> dict[str, Any]:
        """
        Call a specific method on all callbacks with enhanced error handling.

        Args:
            method_name: Name of the method to call
            context: Learning context
            logs: Optional logs dictionary
            timeout: Timeout for individual callback execution

        Returns:
            Dictionary with execution results and error information
        """
        if self._is_shutting_down:
            self.logger.warning(
                "Callback manager is shutting down, skipping method call"
            )
            return {"skipped": True, "reason": "shutting_down"}

        results = {
            "successful_callbacks": 0,
            "failed_callbacks": 0,
            "disabled_callbacks": 0,
            "timeout_callbacks": 0,
            "execution_times": {},
            "errors": [],
        }

        with self._execution_lock:
            for callback in self.callbacks[
                :
            ]:  # Copy list to avoid modification during iteration
                if self._is_shutting_down:
                    break

                callback_name = callback.__class__.__name__

                    results["successful_callbacks"] += 1
                    results["execution_times"][callback_name] = execution_time

                    # Reset error count on success
                    self._error_counts[callback_name] = 0

                except Exception as e:
                    results["failed_callbacks"] += 1
                    error_info = {
                        "callback": callback_name,
                        "method": method_name,
                        "error": str(e),
                        "error_type": type(e).__name__,
                        "timestamp": datetime.now().isoformat(),
                    }
                    results["errors"].append(error_info)

                    # Update error statistics
                    self.callback_stats["errors"] += 1
                    self._error_counts[callback_name] = (
                        self._error_counts.get(callback_name, 0) + 1
                    )

                    # Enhanced error logging
                    self._log_callback_error(callback, method_name, e, context)

                    # Auto-disable failing callbacks if enabled
                    if self._error_config.get("enable_auto_disable", False):

        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
            self.logger.warning(error_msg)

        # Log stack trace for debugging
        if self.logger.isEnabledFor(logging.DEBUG):
            stack_trace = "".join(
                traceback.format_exception(type(error), error, error.__traceback__)
            )
            self.logger.debug(
                f"Stack trace for {callback_name}.{method_name}:\n{stack_trace}"
            )

    def graceful_shutdown(self, timeout: float | None = None) -> dict[str, Any]:
        """
        Perform graceful shutdown of all callbacks.

        Args:
            timeout: Maximum time to wait for shutdown

        Returns:
            Shutdown results
        """
        if self._is_shutting_down:
            return {"already_shutting_down": True}

        timeout = timeout or self._error_config["shutdown_timeout"]
        self._is_shutting_down = True
        self._shutdown_event.set()

        self.logger.info(f"Initiating graceful shutdown with {timeout}s timeout")

        shutdown_results = {
            "successful_shutdowns": 0,
            "failed_shutdowns": 0,
            "timeout_shutdowns": 0,
            "shutdown_errors": [],
            "total_time": 0.0,
        }

        start_time = time.time()

        # Create shutdown context
        shutdown_context = LearningContext(
            epoch=-1,
            is_training=False,
            learning_type="shutdown",  # Special marker for shutdown
        )

        # Call on_training_end on all callbacks
        self.call_method(
            "on_training_end", shutdown_context, timeout=timeout / 2
        )

        # Additional cleanup for callbacks with cleanup methods
        for callback in self.callbacks:
                if hasattr(callback, "cleanup") and callable(callback.cleanup):
                    callback.cleanup()
                    shutdown_results["successful_shutdowns"] += 1
                else:
                    shutdown_results[
                        "successful_shutdowns"
                    ] += 1  # Consider no cleanup as success

            except Exception as e:
                shutdown_results["failed_shutdowns"] += 1
                error_info = {
                    "callback": callback.__class__.__name__,
                    "error": str(e),
                    "error_type": type(e).__name__,
                }
                shutdown_results["shutdown_errors"].append(error_info)
                self.logger.error(
                    f"Error during shutdown of {callback.__class__.__name__}: {e}"
                )

        shutdown_results["total_time"] = time.time() - start_time

        # Clear callback list
        self.callbacks.clear()
                {
                    "name": cb.__class__.__name__,
                    "enabled": getattr(cb, "enabled", True),
                    "priority": getattr(cb, "priority", 0),
                    "error_count": self._error_counts.get(cb.__class__.__name__, 0),
                }
                for cb in self.callbacks
            ],
            "error_config": self._error_config.copy(),
            "is_shutting_down": self._is_shutting_down,
        }

    def reset_error_counts(self) -> None:
        """Reset error counts for all callbacks."""
        self._error_counts.clear()
        self.callback_stats["errors"] = 0
        self.logger.info("Error counts reset")

    def enable_all_callbacks(self) -> None:
        """Re-enable all disabled callbacks."""
        enabled_count = 0
        for callback in self.callbacks:
            if not getattr(callback, "enabled", True):
                callback.enabled = True
                enabled_count += 1
def create_reinforcement_context(**kwargs) -> LearningContext:
    """Create a reinforcement learning context."""
    defaults = {"learning_type": "reinforcement", "algorithm_name": "unknown"}
    defaults.update(kwargs)
    return LearningContext(**defaults)

def create_supervised_context(**kwargs) -> LearningContext:
    """Create a supervised learning context."""
    defaults = {"learning_type": "supervised", "algorithm_name": "unknown"}
    defaults.update(kwargs)
    return LearningContext(**defaults)
    return LearningContext(**defaults)

def safe_callback_execution(error_config: dict[str, Any] | None = None):
    """
    Decorator for safe callback method execution with error handling.

    Args:
        error_config: Error handling configuration override

    Returns:
        Decorated function
            if hasattr(self, "safe_execute"):
                # Use the instance's error handling
                return self.safe_execute(func, *args, **kwargs)
            else:
                # Fallback to basic error handling
                try:
                    return func(self, *args, **kwargs)
                except Exception as e:
                    logger = logging.getLogger(self.__class__.__name__)
                    logger.error(
                        f"Error in {self.__class__.__name__}.{func.__name__}: {e}"
                    )
                    return None

        return wrapper

    return decorator
