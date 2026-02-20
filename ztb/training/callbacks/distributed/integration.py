#!/usr/bin/env python3
"""
Distributed Training Integration Module.

This module provides integration between the distributed training
system and the existing training callbacks and monitoring systems.
"""

from __future__ import annotations

import logging
import threading
import time
from datetime import datetime
from typing import Callable, Mapping, Optional

from ..monitoring.metrics_collector import MetricsCollector
from ..monitoring.real_time_monitor import RealTimeMonitor
from ..performance.memory_optimizer import MemoryMonitor
from .coordinator import (
    DistributedCallbackMixin,
    DistributedConfig,
    DistributedCoordinator,
    WorkerInfo,
)
from .threading_mixin import BackgroundThreadController
from .worker import TaskCallback, WorkerPool


def _as_object_map(payload: object) -> dict[str, object]:
    if isinstance(payload, dict):
        normalized: dict[str, object] = {}
        for key, value in payload.items():
            normalized[str(key)] = value
        return normalized
    return {}


def _safe_int(value: object, default: int = 0) -> int:
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, (int, float)):
        return int(value)
    try:
        return int(str(value))
    except (TypeError, ValueError):
        return default


class DistributedTrainingManager(BackgroundThreadController):
    """
    Manager for distributed training operations.

    Integrates distributed capabilities with existing training systems,
    providing seamless scaling and fault tolerance.
    """

    def __init__(self, config: Optional[DistributedConfig] = None):
        self.config = config or DistributedConfig()
        self.logger = logging.getLogger(__name__)

        # Core components
        self.coordinator = DistributedCoordinator(self.config)
        self.worker_pool = WorkerPool(self.config.num_workers, self.config)

        # Integration components
        self.memory_monitor = MemoryMonitor()
        self.real_time_monitor: Optional[RealTimeMonitor] = None
        self.metrics_collector: Optional[MetricsCollector] = None

        # State
        self.is_initialized = False
        self.training_active = False
        self.distributed_mode = self.config.enable_distributed
        self.training_config: dict[str, object] = {}

        # Synchronization
        self.sync_lock = threading.RLock()
        self._sync_thread: Optional[threading.Thread] = None

        # Callbacks
        self.training_callbacks: list[Callable[..., object]] = []

    def initialize(
        self,
        real_time_monitor: Optional[RealTimeMonitor] = None,
        metrics_collector: Optional[MetricsCollector] = None,
    ) -> bool:
        """Initialize the distributed training manager."""
        try:
            self.logger.info("Initializing distributed training manager")

            # Set up integrations
            self.real_time_monitor = real_time_monitor
            self.metrics_collector = metrics_collector

            if self.distributed_mode:
                # Start coordinator
                self.coordinator.start_coordination()

                # Start worker pool
                if not self.worker_pool.start_pool():
                    self.logger.error("Failed to start worker pool")
                    self.coordinator.stop_coordination()
                    return False

            # Start memory monitoring
            self.memory_monitor.start_monitoring()

            self.is_initialized = True
            self.logger.info("Distributed training manager initialized successfully")
            return True

        except Exception as e:
            self.logger.error(f"Failed to initialize distributed training manager: {e}")
            return False

    def shutdown(self) -> None:
        """Shutdown the distributed training manager."""
        self.logger.info("Shutting down distributed training manager")

        self.stop_distributed_training()
        self.training_active = False
        self._join_background_thread(attr_name="_sync_thread", timeout=5.0)

        # Stop worker pool
        self.worker_pool.stop_pool()

        # Stop coordinator
        self.coordinator.stop_coordination()

        # Stop memory monitoring
        self.memory_monitor.stop_monitoring()

        self.is_initialized = False

    def start_distributed_training(self, training_config: Mapping[str, object]) -> bool:
        """Start distributed training session."""
        if not self.is_initialized:
            self.logger.error("Distributed training manager not initialized")
            return False

        if self.training_active:
            self.logger.warning("Training already active")
            return False

        try:
            self.logger.info("Starting distributed training session")

            with self.sync_lock:
                self.training_active = True

                # Initialize training state
                self._initialize_training_state(training_config)

                # Register training callbacks
                self._register_training_callbacks()

                # Start monitoring
                if self.real_time_monitor:
                    self.real_time_monitor.start_monitoring()

                if self.metrics_collector:
                    self.metrics_collector.start_collection()

                self._start_sync_thread()

            self.logger.info("Distributed training session started")
            return True

        except Exception as e:
            self.logger.error(f"Failed to start distributed training: {e}")
            return False

    def stop_distributed_training(self) -> None:
        """Stop distributed training session."""
        if not self.training_active:
            return

        self.logger.info("Stopping distributed training session")

        with self.sync_lock:
            self.training_active = False
            self._join_background_thread(attr_name="_sync_thread", timeout=5.0)

            # Stop monitoring
            if self.real_time_monitor:
                self.real_time_monitor.stop_monitoring()

            if self.metrics_collector:
                self.metrics_collector.stop_collection()

            # Cleanup training state
            self._cleanup_training_state()

    def submit_training_task(
        self,
        task_type: str,
        task_data: Mapping[str, object],
        callback: Optional[TaskCallback] = None,
    ) -> Optional[str]:
        """Submit a training task to the distributed system."""
        if not self.training_active:
            return None

        task_payload = dict(task_data)
        task_payload["distributed"] = True
        task_payload["timestamp"] = datetime.now().isoformat()
        task_payload["memory_info"] = self.memory_monitor.get_memory_stats()

        return self.worker_pool.submit_task(task_type, task_payload, callback)

    def get_training_status(self) -> dict[str, object]:
        """Get comprehensive training status."""
        if not self.is_initialized:
            return {"status": "not_initialized"}

        with self.sync_lock:
            status: dict[str, object] = {
                "training_active": self.training_active,
                "distributed_mode": self.distributed_mode,
                "coordinator_status": self.coordinator.get_worker_stats()
                if self.distributed_mode
                else {},
                "worker_pool_status": self.worker_pool.get_pool_status(),
                "memory_status": self.memory_monitor.get_memory_stats(),
                "timestamp": datetime.now().isoformat(),
            }

            # Add monitoring data if available
            if self.real_time_monitor:
                status["real_time_metrics"] = self.real_time_monitor.get_current_metrics()

            if self.metrics_collector:
                status["collected_metrics"] = self.metrics_collector.get_latest_metrics()

            return status

    def register_training_callback(self, callback: Callable[..., object]) -> None:
        """Register a training callback."""
        self.training_callbacks.append(callback)

    def _initialize_training_state(self, config: Mapping[str, object]) -> None:
        """Initialize training state for distributed session."""
        self.training_config = dict(config)

        # Initialize worker states
        if self.distributed_mode:
            for worker_id in range(1, self.config.num_workers + 1):
                worker_info = WorkerInfo(
                    worker_id=worker_id,
                    host="localhost",
                    port=self.config.master_port + worker_id,
                )
                self.coordinator.register_worker(worker_info)

    def _cleanup_training_state(self) -> None:
        """Clean up training state."""
        # Clear worker registrations
        if self.distributed_mode:
            worker_ids = list(self.coordinator.workers.keys())
            for worker_id in worker_ids:
                self.coordinator.unregister_worker(worker_id)

        # Clear callbacks
        self.training_callbacks.clear()

    def _register_training_callbacks(self) -> None:
        """Register callbacks for training events."""
        # Register with real-time monitor if available
        if self.real_time_monitor:
            self.real_time_monitor.register_callback(
                "training_progress", self._handle_training_progress
            )

        # Register with metrics collector if available
        if self.metrics_collector:
            self.metrics_collector.register_callback(
                "epoch_complete", self._handle_epoch_complete
            )

    def _start_sync_thread(self) -> None:
        """Start periodic synchronization thread for active distributed sessions."""
        if not (self.training_active and self.distributed_mode):
            return

        self._start_background_thread(
            attr_name="_sync_thread",
            target=self._synchronization_loop,
            name="distributed-sync",
            daemon=True,
        )

    @staticmethod
    def _memory_pressure(memory_stats: Mapping[str, object]) -> float:
        pressure = memory_stats.get("memory_pressure")
        if isinstance(pressure, (int, float)):
            return max(0.0, float(pressure))

        current_mb = memory_stats.get("current_mb")
        threshold_mb = memory_stats.get("threshold_mb")
        if (
            isinstance(current_mb, (int, float))
            and isinstance(threshold_mb, (int, float))
            and threshold_mb > 0
        ):
            return max(0.0, float(current_mb) / float(threshold_mb))
        return 0.0

    def _handle_training_progress(self, metrics: Mapping[str, object]) -> None:
        """Handle training progress updates."""
        if not self.training_active:
            return

        # Distribute metrics to coordinator
        self.coordinator.aggregate_metrics({0: dict(metrics)})  # Worker 0 = master

        # Check for memory issues
        memory_stats = self.memory_monitor.get_memory_stats()
        if self._memory_pressure(memory_stats) > 0.8:  # 80% memory usage
            self.logger.warning("High memory usage detected, triggering cleanup")
            self.memory_monitor.force_cleanup()

    def _handle_epoch_complete(self, metrics: Mapping[str, object]) -> None:
        """Handle epoch completion."""
        if not self.training_active:
            return

        # Synchronize weights across workers
        if self.distributed_mode:
            logs = _as_object_map(metrics.get("model_weights"))
            sync_task: dict[str, object] = {
                "epoch": _safe_int(metrics.get("epoch"), default=0),
                "weights": logs,
                "metrics": dict(metrics),
            }
            self.submit_training_task("sync_weights", sync_task)

    def _synchronization_loop(self) -> None:
        """Main synchronization loop for distributed training."""
        while self.training_active and self.distributed_mode:
            try:
                # Synchronize metrics
                worker_metrics: dict[int, dict[str, object]] = {}
                for worker_id, worker in self.coordinator.workers.items():
                    if worker.metrics:
                        worker_metrics[worker_id] = dict(worker.metrics)

                if worker_metrics:
                    aggregated = self.coordinator.aggregate_metrics(worker_metrics)

                    # Update global metrics
                    if self.metrics_collector:
                        self.metrics_collector.update_metrics(
                            {
                                "distributed_aggregated": aggregated,
                                "sync_timestamp": datetime.now().isoformat(),
                            }
                        )

                # Memory synchronization
                memory_stats = self.memory_monitor.get_memory_stats()
                if self._memory_pressure(memory_stats) > 0.9:  # 90% memory usage
                    self.logger.warning(
                        "Critical memory usage, triggering emergency cleanup"
                    )
                    self.memory_monitor.force_cleanup()

                time.sleep(self.config.sync_interval)

            except Exception as e:
                self.logger.error(f"Error in synchronization loop: {e}")
                time.sleep(5.0)


class DistributedCallbackAdapter(DistributedCallbackMixin):
    """
    Adapter to add distributed capabilities to existing callbacks.

    This class wraps existing callbacks and adds distributed functionality
    without modifying the original callback code.
    """

    def __init__(
        self, base_callback: object, coordinator: Optional[DistributedCoordinator] = None
    ):
        super().__init__(coordinator)
        self.base_callback = base_callback
        self.logger = logging.getLogger(f"{self.__class__.__name__}")

        # Wrap callback methods to add distributed functionality
        self._wrap_callback_methods()

    def _wrap_callback_methods(self) -> None:
        """Wrap callback methods to add distributed functionality."""
        original_methods = [
            "on_training_start",
            "on_training_end",
            "on_epoch_start",
            "on_epoch_end",
            "on_batch_start",
            "on_batch_end",
        ]

        for method_name in original_methods:
            if hasattr(self.base_callback, method_name):
                original_method = getattr(self.base_callback, method_name)
                if callable(original_method):
                    wrapped_method = self._create_wrapped_method(
                        method_name, original_method
                    )
                    setattr(self, method_name, wrapped_method)
                    continue

            setattr(self, method_name, self._noop_callback)

    @staticmethod
    def _noop_callback(*_args: object, **_kwargs: object) -> None:
        return None

    def _create_wrapped_method(
        self, method_name: str, original_method: Callable[..., object]
    ) -> Callable[..., object]:
        """Create a wrapped method that adds distributed functionality."""

        def wrapped_method(*args: object, **kwargs: object) -> object:
            try:
                # Call original method
                result = original_method(*args, **kwargs)

                # Add distributed functionality
                if self.is_distributed:
                    self._handle_distributed_event(method_name, kwargs, result)

                return result

            except Exception as e:
                self.logger.error(f"Error in distributed callback {method_name}: {e}")
                if self.is_distributed:
                    self.report_error_to_coordinator(str(e))
                raise

        return wrapped_method

    def _handle_distributed_event(
        self, event_type: str, kwargs: Mapping[str, object], result: object
    ) -> None:
        """Handle distributed event processing."""
        _ = result

        # Send heartbeat
        self.heartbeat_to_coordinator()

        # Send relevant metrics
        metrics: dict[str, object] = {
            "event_type": event_type,
            "timestamp": datetime.now().isoformat(),
            "worker_id": self.worker_id,
        }

        # Add event-specific metrics
        if event_type == "on_epoch_end":
            metrics.update(
                {"epoch": _safe_int(kwargs.get("epoch"), 0), "logs": kwargs.get("logs", {})}
            )
        elif event_type == "on_batch_end":
            metrics.update(
                {"batch": _safe_int(kwargs.get("batch"), 0), "logs": kwargs.get("logs", {})}
            )

        self.send_metrics_to_coordinator(metrics)


# Global instance
_global_distributed_manager: Optional[DistributedTrainingManager] = None


def get_distributed_manager() -> DistributedTrainingManager:
    """Get the global distributed training manager instance."""
    global _global_distributed_manager
    if _global_distributed_manager is None:
        _global_distributed_manager = DistributedTrainingManager()
    return _global_distributed_manager
