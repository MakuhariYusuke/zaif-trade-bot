#!/usr/bin/env python3
"""
Distributed Training Support Module.

This module provides support for distributed training scenarios,
including multi-process communication, data synchronization, and
scalability optimizations.
"""

import logging
import multiprocessing as mp
import pickle
import queue
import threading
import time
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, Mapping, Optional

from .threading_mixin import BackgroundThreadController


@dataclass
class DistributedConfig:
    """Configuration for distributed training."""

    enable_distributed: bool = False
    master_host: str = "localhost"
    master_port: int = 12345
    num_workers: int = 4
    sync_interval: float = 10.0  # seconds
    heartbeat_interval: float = 30.0
    max_queue_size: int = 1000
    enable_process_pool: bool = True
    enable_thread_pool: bool = True
    compression_enabled: bool = True


@dataclass
class WorkerInfo:
    """Information about a worker process."""

    worker_id: int
    host: str
    port: int
    status: str = "idle"  # idle, busy, error
    last_heartbeat: datetime = field(default_factory=datetime.now)
    metrics: Dict[str, Any] = field(default_factory=dict)


class Message:
    """Message for inter-process communication."""

    def __init__(
        self,
        msg_type: str,
        sender_id: int,
        data: Any,
        timestamp: Optional[datetime] = None,
    ):
        self.msg_type = msg_type
        self.sender_id = sender_id
        self.data = data
        self.timestamp = timestamp or datetime.now()

    def to_bytes(self) -> bytes:
        """Serialize message to bytes."""
        msg_dict = {
            "type": self.msg_type,
            "sender_id": self.sender_id,
            "data": self.data,
            "timestamp": self.timestamp.isoformat(),
        }
        return pickle.dumps(msg_dict)

    @classmethod
    def from_bytes(cls, data: bytes) -> "Message":
        """Deserialize message from bytes."""
        msg_dict = pickle.loads(data)
        timestamp = datetime.fromisoformat(msg_dict["timestamp"])
        return cls(
            msg_type=msg_dict["type"],
            sender_id=msg_dict["sender_id"],
            data=msg_dict["data"],
            timestamp=timestamp,
        )


class DistributedCoordinator(BackgroundThreadController):
    """
    Coordinator for distributed training operations.

    Features:
    - Master-worker coordination
    - Load balancing
    - Fault tolerance
    - Metrics aggregation
    """

    def __init__(self, config: Optional[DistributedConfig] = None):
        self.config = config or DistributedConfig()
        self.logger = logging.getLogger(__name__)
        self.is_master = True
        self.master_host = self.config.master_host
        self.master_port = self.config.master_port

        # Worker management
        self.workers: Dict[int, WorkerInfo] = {}
        self.worker_lock = threading.RLock()

        # Communication
        self.message_queue: mp.Queue = mp.Queue(maxsize=self.config.max_queue_size)
        self.response_queues: Dict[int, mp.Queue] = {}

        # Threading
        self._running = False
        self._coordinator_thread: Optional[threading.Thread] = None
        self._heartbeat_thread: Optional[threading.Thread] = None

        # Pools
        self.process_pool: Optional[ProcessPoolExecutor] = None
        self.thread_pool: Optional[ThreadPoolExecutor] = None

        # Statistics
        self.stats = {
            "messages_sent": 0,
            "messages_received": 0,
            "tasks_distributed": 0,
            "tasks_completed": 0,
            "worker_failures": 0,
        }

    @staticmethod
    def _normalize_worker_info(worker_info: object, default_port: int) -> Optional[WorkerInfo]:
        """Normalize worker registration input into WorkerInfo."""
        if isinstance(worker_info, WorkerInfo):
            return worker_info

        if isinstance(worker_info, Mapping):
            worker_id = worker_info.get("worker_id")
            if not isinstance(worker_id, int):
                return None
            host_value = worker_info.get("host", "localhost")
            port_value = worker_info.get("port", default_port)
            status_value = worker_info.get("status", "idle")
            metrics_value = worker_info.get("metrics", {})
            return WorkerInfo(
                worker_id=worker_id,
                host=str(host_value),
                port=int(port_value) if isinstance(port_value, (int, float)) else default_port,
                status=str(status_value),
                metrics=metrics_value if isinstance(metrics_value, dict) else {},
            )

        if hasattr(worker_info, "worker_id"):
            worker_id = getattr(worker_info, "worker_id", None)
            if not isinstance(worker_id, int):
                return None
            host = str(getattr(worker_info, "host", "localhost"))
            raw_port = getattr(worker_info, "port", default_port)
            port = int(raw_port) if isinstance(raw_port, (int, float)) else default_port
            status = str(getattr(worker_info, "status", "idle"))
            metrics = getattr(worker_info, "metrics", {})
            if not isinstance(metrics, dict):
                metrics = {}
            last_heartbeat = getattr(worker_info, "last_heartbeat", datetime.now())
            if not isinstance(last_heartbeat, datetime):
                last_heartbeat = datetime.now()
            return WorkerInfo(
                worker_id=worker_id,
                host=host,
                port=port,
                status=status,
                last_heartbeat=last_heartbeat,
                metrics=metrics,
            )

        return None

    @staticmethod
    def _worker_status(worker: WorkerInfo | dict[str, object]) -> str:
        if isinstance(worker, dict):
            status = worker.get("status", "idle")
            return str(status)
        return worker.status

    @staticmethod
    def _set_worker_status(worker: WorkerInfo | dict[str, object], status: str) -> None:
        if isinstance(worker, dict):
            worker["status"] = status
        else:
            worker.status = status

    @staticmethod
    def _worker_metrics(worker: WorkerInfo | dict[str, object]) -> dict[str, object]:
        if isinstance(worker, dict):
            metrics = worker.get("metrics")
            if not isinstance(metrics, dict):
                metrics = {}
                worker["metrics"] = metrics
            return metrics
        return worker.metrics

    @staticmethod
    def _set_worker_last_heartbeat(
        worker: WorkerInfo | dict[str, object], timestamp: datetime
    ) -> None:
        if isinstance(worker, dict):
            worker["last_heartbeat"] = timestamp
        else:
            worker.last_heartbeat = timestamp

    @staticmethod
    def _worker_last_heartbeat(worker: WorkerInfo | dict[str, object]) -> datetime:
        if isinstance(worker, dict):
            value = worker.get("last_heartbeat")
            if isinstance(value, datetime):
                return value
            return datetime.now()
        return worker.last_heartbeat

    def start_coordination(self) -> None:
        """Start the distributed coordination."""
        if self._running:
            return

        self._running = True
        self.logger.info("Starting distributed coordination")

        # Start pools
        if self.config.enable_process_pool:
            self.process_pool = ProcessPoolExecutor(max_workers=self.config.num_workers)
        if self.config.enable_thread_pool:
            self.thread_pool = ThreadPoolExecutor(
                max_workers=self.config.num_workers * 2
            )

        self._start_background_thread(
            attr_name="_coordinator_thread",
            target=self._coordination_loop,
            name="distributed-coordinator",
            daemon=True,
        )
        self._start_background_thread(
            attr_name="_heartbeat_thread",
            target=self._heartbeat_loop,
            name="heartbeat-monitor",
            daemon=True,
        )

    def stop_coordination(self) -> None:
        """Stop the distributed coordination."""
        if not self._running:
            return

        self.logger.info("Stopping distributed coordination")
        self._running = False

        # Stop pools
        if self.process_pool:
            self.process_pool.shutdown(wait=True)
        if self.thread_pool:
            self.thread_pool.shutdown(wait=True)

        self._join_background_thread(attr_name="_coordinator_thread", timeout=5.0)
        self._join_background_thread(attr_name="_heartbeat_thread", timeout=5.0)

    def register_worker(self, worker_info: WorkerInfo | Mapping[str, object] | object) -> bool:
        """Register a worker with the coordinator."""
        with self.worker_lock:
            normalized = self._normalize_worker_info(worker_info, self.master_port)
            if normalized is None:
                self.logger.warning(f"Invalid worker registration payload: {worker_info!r}")
                return False

            if normalized.worker_id in self.workers:
                self.logger.warning(
                    f"Worker {normalized.worker_id} already registered"
                )
                return False

            self.workers[normalized.worker_id] = normalized
            self.response_queues[normalized.worker_id] = mp.Queue()
            self.logger.info(f"Registered worker {normalized.worker_id}")
            return True

    def unregister_worker(self, worker_id: int) -> bool:
        """Unregister a worker."""
        with self.worker_lock:
            if worker_id not in self.workers:
                return False

            del self.workers[worker_id]
            self.response_queues.pop(worker_id, None)
            self.logger.info(f"Unregistered worker {worker_id}")
            return True

    def distribute_task(
        self, task_data: Any, worker_id: Optional[int] = None
    ) -> Optional[int]:
        """Distribute a task to a worker."""
        with self.worker_lock:
            if not self.workers:
                self.logger.warning("No workers available")
                return None

            # Select worker
            if worker_id is None:
                # Simple load balancing - pick least busy worker
                available_workers = [
                    w for w in self.workers.values() if self._worker_status(w) == "idle"
                ]
                if not available_workers:
                    self.logger.warning("All workers are busy")
                    return None
                selected_worker = min(
                    available_workers,
                    key=lambda w: len(self._worker_metrics(w).get("active_tasks", [])),
                )
                worker_id = selected_worker.worker_id
            elif (
                worker_id not in self.workers
                or self._worker_status(self.workers[worker_id]) != "idle"
            ):
                self.logger.warning(f"Worker {worker_id} not available")
                return None

            # Send task
            try:
                message = Message("task", 0, task_data)  # sender_id 0 = master
                self.message_queue.put(message, timeout=1.0)
                self._set_worker_status(self.workers[worker_id], "busy")
                self.stats["tasks_distributed"] += 1
                self.logger.debug(f"Distributed task to worker {worker_id}")
                return worker_id
            except queue.Full:
                self.logger.error("Message queue full")
                return None

    def get_worker_stats(self) -> Dict[str, Any]:
        """Get statistics about all workers."""
        with self.worker_lock:
            worker_snapshots = {
                wid: {
                    "status": self._worker_status(w),
                    "last_heartbeat": self._worker_last_heartbeat(w).isoformat(),
                    "metrics": self._worker_metrics(w),
                }
                for wid, w in self.workers.items()
            }
            return {
                "total_workers": len(self.workers),
                "active_workers": len(
                    [w for w in self.workers.values() if self._worker_status(w) == "busy"]
                ),
                "idle_workers": len(
                    [w for w in self.workers.values() if self._worker_status(w) == "idle"]
                ),
                "failed_workers": len(
                    [w for w in self.workers.values() if self._worker_status(w) == "error"]
                ),
                "workers": worker_snapshots,
            }

    def aggregate_metrics(
        self, worker_metrics: Dict[int, Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Aggregate metrics from all workers."""
        if not worker_metrics:
            return {}

        # Simple aggregation - can be extended for more complex logic
        aggregated = {}
        metric_keys = set()

        # Collect all metric keys
        for metrics in worker_metrics.values():
            metric_keys.update(metrics.keys())

        # Aggregate each metric
        for key in metric_keys:
            values = [
                metrics.get(key)
                for metrics in worker_metrics.values()
                if metrics.get(key) is not None
            ]

            if not values:
                continue

            if isinstance(values[0], (int, float)):
                aggregated[key] = {
                    "mean": sum(values) / len(values),
                    "min": min(values),
                    "max": max(values),
                    "sum": sum(values),
                    "count": len(values),
                }
            else:
                # For non-numeric values, just collect them
                aggregated[key] = {"values": values, "count": len(values)}

        return aggregated

    def _coordination_loop(self) -> None:
        """Main coordination loop."""
        last_sync = time.time()

        while self._running:
            try:
                current_time = time.time()

                # Process incoming messages
                self._process_messages()

                # Periodic synchronization
                if current_time - last_sync >= self.config.sync_interval:
                    self._sync_workers()
                    last_sync = current_time

                time.sleep(0.1)  # Small sleep to prevent busy waiting

            except Exception as e:
                self.logger.error(f"Error in coordination loop: {e}")
                time.sleep(1.0)

    def _process_messages(self) -> None:
        """Process incoming messages from workers."""
        try:
            while True:
                message = self.message_queue.get_nowait()
                self.stats["messages_received"] += 1

                if message.msg_type == "heartbeat":
                    self._handle_heartbeat(message)
                elif message.msg_type == "task_result":
                    self._handle_task_result(message)
                elif message.msg_type == "metrics":
                    self._handle_metrics(message)
                elif message.msg_type == "error":
                    self._handle_error(message)

        except queue.Empty:
            pass  # No messages to process

    def _handle_heartbeat(self, message: Message) -> None:
        """Handle heartbeat message from worker."""
        worker_id = message.sender_id
        with self.worker_lock:
            if worker_id in self.workers:
                w = self.workers[worker_id]
                self._set_worker_last_heartbeat(w, message.timestamp)
                self._set_worker_status(w, "idle")  # Reset status on heartbeat

    def _handle_task_result(self, message: Message) -> None:
        """Handle task result from worker."""
        worker_id = message.sender_id
        with self.worker_lock:
            if worker_id in self.workers:
                w = self.workers[worker_id]
                self._set_worker_status(w, "idle")
                self.stats["tasks_completed"] += 1

    def _handle_metrics(self, message: Message) -> None:
        """Handle metrics update from worker."""
        worker_id = message.sender_id
        with self.worker_lock:
            if worker_id in self.workers:
                w = self.workers[worker_id]
                metrics = self._worker_metrics(w)
                if isinstance(message.data, dict):
                    metrics.update(message.data)

    def _handle_error(self, message: Message) -> None:
        """Handle error message from worker."""
        worker_id = message.sender_id
        with self.worker_lock:
            if worker_id in self.workers:
                self._set_worker_status(self.workers[worker_id], "error")
                self.stats["worker_failures"] += 1
                self.logger.error(f"Worker {worker_id} reported error: {message.data}")

    def _sync_workers(self) -> None:
        """Synchronize state with all workers."""
        # Send sync message to all workers
        sync_message = Message("sync", 0, {"timestamp": datetime.now().isoformat()})
        try:
            self.message_queue.put(sync_message, timeout=0.1)
            self.stats["messages_sent"] += 1
        except queue.Full:
            self.logger.warning("Sync message queue full")

    def _heartbeat_loop(self) -> None:
        """Monitor worker heartbeats and detect failures."""
        while self._running:
            try:
                current_time = datetime.now()

                with self.worker_lock:
                    for worker_id, worker in self.workers.items():
                        time_since_heartbeat = (
                            current_time - self._worker_last_heartbeat(worker)
                        ).total_seconds()
                        if time_since_heartbeat > self.config.heartbeat_interval * 3:
                            # Worker is considered failed
                            self._set_worker_status(worker, "error")
                            self.stats["worker_failures"] += 1
                            self.logger.error(
                                f"Worker {worker_id} failed (no heartbeat for {time_since_heartbeat:.1f}s)"
                            )

                time.sleep(self.config.heartbeat_interval)

            except Exception as e:
                self.logger.error(f"Error in heartbeat loop: {e}")
                time.sleep(10.0)


class DistributedCallbackMixin:
    """
    Mixin class to add distributed capabilities to callbacks.

    This mixin provides methods for distributed operation and
    communication with the coordinator.
    """

    def __init__(self, coordinator: Optional[DistributedCoordinator] = None):
        self.coordinator = coordinator
        self.worker_id = 0  # Default to master
        self.is_distributed = coordinator is not None

    def send_metrics_to_coordinator(self, metrics: Dict[str, Any]) -> None:
        """Send metrics to the distributed coordinator."""
        if self.is_distributed and self.coordinator:
            message = Message("metrics", self.worker_id, metrics)
            try:
                self.coordinator.message_queue.put(message, timeout=0.1)
            except queue.Full:
                logging.warning("Metrics queue full, dropping metrics")

    def report_error_to_coordinator(self, error: str) -> None:
        """Report an error to the distributed coordinator."""
        if self.is_distributed and self.coordinator:
            message = Message("error", self.worker_id, error)
            try:
                self.coordinator.message_queue.put(message, timeout=0.1)
            except queue.Full:
                logging.error("Error queue full")

    def heartbeat_to_coordinator(self) -> None:
        """Send heartbeat to the distributed coordinator."""
        if self.is_distributed and self.coordinator:
            message = Message("heartbeat", self.worker_id, {})
            try:
                self.coordinator.message_queue.put(message, timeout=0.1)
            except queue.Full:
                logging.warning("Heartbeat queue full")


# Global coordinator instance
_global_coordinator = None


def get_global_coordinator() -> DistributedCoordinator:
    """Get the global distributed coordinator instance."""
    global _global_coordinator
    if _global_coordinator is None:
        _global_coordinator = DistributedCoordinator()
    return _global_coordinator
