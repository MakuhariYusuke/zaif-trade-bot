#!/usr/bin/env python3
"""
Distributed Worker Implementation.

This module provides worker processes for distributed training,
handling task execution, communication with coordinator, and
fault tolerance.
"""

from __future__ import annotations

import logging
import multiprocessing as mp
import os
import queue
import signal
import threading
import time
import traceback
from concurrent.futures import Future, ThreadPoolExecutor
from datetime import datetime
from types import FrameType
from typing import Callable, Mapping, Optional, TypedDict

from .coordinator import DistributedConfig, Message
from .threading_mixin import BackgroundThreadController


class WorkerStats(TypedDict):
    """Runtime statistics for a distributed worker."""

    tasks_executed: int
    tasks_failed: int
    uptime: float
    cpu_usage: float
    memory_usage: float


class WorkerStatusSnapshot(TypedDict):
    """Public worker status payload."""

    worker_id: int
    is_running: bool
    status: str
    last_heartbeat: str
    stats: WorkerStats
    process_alive: bool


TaskHandler = Callable[[Mapping[str, object]], object]
TaskCallback = Callable[[object | None], None]


class TaskInfo(TypedDict):
    """Task envelope passed through worker pool queue."""

    task_id: str
    type: str
    data: object
    callback: Optional[TaskCallback]
    submitted_at: datetime


class TaskResultInfo(TypedDict):
    """Task result stored in worker pool result queue."""

    task_id: str
    result: object | None
    completed_at: datetime
    worker_id: int


def _as_payload_map(payload: object) -> dict[str, object]:
    """Normalize dynamic payloads into a string-keyed object mapping."""
    if isinstance(payload, dict):
        normalized: dict[str, object] = {}
        for key, value in payload.items():
            if isinstance(key, str):
                normalized[key] = value
            else:
                normalized[str(key)] = value
        return normalized
    return {}


class DistributedWorker:
    """
    Worker process for distributed training operations.

    Features:
    - Task execution in separate process
    - Heartbeat monitoring
    - Error handling and recovery
    - Resource management
    """

    def __init__(self, worker_id: int, config: Optional[DistributedConfig] = None):
        self.worker_id = worker_id
        self.config = config or DistributedConfig()
        self.logger = logging.getLogger(f"Worker-{worker_id}")

        # Process management
        self.process: Optional[mp.Process] = None
        self.input_queue: mp.Queue = mp.Queue(maxsize=self.config.max_queue_size)
        self.output_queue: mp.Queue = mp.Queue(maxsize=self.config.max_queue_size)

        # Status
        self.is_running = False
        self.last_heartbeat = datetime.now()
        self.status = "idle"

        # Statistics
        self.stats: WorkerStats = {
            "tasks_executed": 0,
            "tasks_failed": 0,
            "uptime": 0.0,
            "cpu_usage": 0.0,
            "memory_usage": 0.0,
        }

        # Synchronization
        self._task_lock = threading.Lock()
        self._state_lock = threading.RLock()

        # Task execution
        self.task_handlers: dict[str, TaskHandler] = {}
        self._register_default_handlers()

    def __getstate__(self) -> dict[str, object]:
        """Make instance pickle-safe for spawn-based multiprocessing."""
        state = dict(self.__dict__)
        state["_task_lock"] = None
        state["_state_lock"] = None
        return state

    def __setstate__(self, state: dict[str, object]) -> None:
        """Restore process-local locks after unpickling."""
        self.__dict__.update(state)
        self._task_lock = threading.Lock()
        self._state_lock = threading.RLock()

    def _register_default_handlers(self) -> None:
        """Register default task handlers."""
        self.task_handlers.update(
            {
                "train_epoch": self._handle_train_epoch,
                "evaluate_model": self._handle_evaluate_model,
                "sync_weights": self._handle_sync_weights,
                "shutdown": self._handle_shutdown,
            }
        )

    def start(self) -> bool:
        """Start the worker process."""
        if self.is_running:
            return False

        try:
            self.process = mp.Process(
                target=self._worker_loop,
                args=(self.worker_id, self.input_queue, self.output_queue, self.config),
                name=f"Worker-{self.worker_id}",
                daemon=True,
            )
            self.process.start()
            with self._state_lock:
                self.is_running = True
                self.status = "idle"
                self.last_heartbeat = datetime.now()
            self.logger.info(f"Started worker {self.worker_id}")
            return True
        except Exception as e:
            self.logger.error(f"Failed to start worker {self.worker_id}: {e}")
            return False

    def stop(self, timeout: float = 5.0) -> bool:
        """Stop the worker process."""
        if not self.is_running:
            return False

        try:
            # Send shutdown message
            shutdown_msg = Message("shutdown", 0, {})
            self.input_queue.put(shutdown_msg, timeout=1.0)

            # Wait for process to finish
            if self.process:
                self.process.join(timeout=timeout)
                if self.process.is_alive():
                    self.logger.warning(
                        f"Worker {self.worker_id} did not shutdown gracefully, terminating"
                    )
                    self.process.terminate()
                    self.process.join(timeout=2.0)

            with self._state_lock:
                self.is_running = False
                self.status = "stopped"
            self.logger.info(f"Stopped worker {self.worker_id}")
            return True
        except Exception as e:
            self.logger.error(f"Error stopping worker {self.worker_id}: {e}")
            return False

    def send_task(
        self, task_type: str, task_data: object, timeout: float = 10.0
    ) -> object | None:
        """Send a task to the worker and wait for result."""
        if not self.is_running:
            return None

        with self._task_lock:
            deadline = time.monotonic() + timeout
            with self._state_lock:
                self.status = "busy"

            try:
                message = Message("task", 0, {"type": task_type, "data": task_data})
                self.input_queue.put(message, timeout=1.0)

                while True:
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        raise queue.Empty

                    result_msg = self.output_queue.get(timeout=min(0.5, remaining))
                    with self._state_lock:
                        self.last_heartbeat = result_msg.timestamp

                    # Heartbeat and sync ack are out-of-band; wait for task result.
                    if result_msg.msg_type in {"heartbeat", "sync_ack"}:
                        continue

                    if result_msg.msg_type == "task_result":
                        with self._state_lock:
                            self.stats["tasks_executed"] += 1
                        return result_msg.data

                    if result_msg.msg_type == "error":
                        with self._state_lock:
                            self.stats["tasks_failed"] += 1
                        self.logger.error(
                            f"Task failed on worker {self.worker_id}: {result_msg.data}"
                        )
                        return None

                    self.logger.warning(
                        f"Unexpected message type from worker {self.worker_id}: {result_msg.msg_type}"
                    )

            except queue.Full:
                self.logger.error("Input queue full")
                return None
            except queue.Empty:
                self.logger.error("Timeout waiting for task result")
                return None
            except Exception as e:
                self.logger.error(f"Error sending task to worker {self.worker_id}: {e}")
                return None
            finally:
                if self.is_running:
                    with self._state_lock:
                        self.status = "idle"

    def get_status(self) -> WorkerStatusSnapshot:
        """Get the current status of the worker."""
        with self._state_lock:
            process_alive = self.process.is_alive() if self.process else False
            return {
                "worker_id": self.worker_id,
                "is_running": self.is_running,
                "status": self.status,
                "last_heartbeat": self.last_heartbeat.isoformat(),
                "stats": self.stats.copy(),
                "process_alive": process_alive,
            }

    def _worker_loop(
        self,
        worker_id: int,
        input_queue: mp.Queue,
        output_queue: mp.Queue,
        config: DistributedConfig,
    ) -> None:
        """Main worker process loop."""
        try:
            # Set up signal handlers
            signal.signal(signal.SIGTERM, self._signal_handler)
            signal.signal(signal.SIGINT, self._signal_handler)

            self.logger.info(f"Worker {worker_id} process started")

            # Initialize worker state
            start_time = time.time()
            last_heartbeat = time.time()

            while True:
                try:
                    # Check heartbeat interval
                    current_time = time.time()
                    if current_time - last_heartbeat > config.heartbeat_interval:
                        heartbeat_msg = Message("heartbeat", worker_id, {})
                        output_queue.put(heartbeat_msg, timeout=0.1)
                        last_heartbeat = current_time

                    # Try to get a task
                    try:
                        message = input_queue.get(timeout=0.1)
                    except queue.Empty:
                        continue

                    if message.msg_type == "shutdown":
                        self.logger.info(f"Worker {worker_id} received shutdown signal")
                        break
                    if message.msg_type == "task":
                        self._execute_task(message, output_queue)
                    elif message.msg_type == "sync":
                        sync_response = Message("sync_ack", worker_id, {})
                        output_queue.put(sync_response, timeout=0.1)

                    with self._state_lock:
                        self.stats["uptime"] = current_time - start_time

                except Exception as e:
                    error_msg = Message("error", worker_id, str(e))
                    try:
                        output_queue.put(error_msg, timeout=0.1)
                    except queue.Full:
                        pass  # Queue full, can't report error
                    self.logger.error(f"Error in worker {worker_id} loop: {e}")
                    time.sleep(1.0)  # Brief pause before continuing

        except Exception as e:
            self.logger.error(f"Fatal error in worker {worker_id}: {e}")
        finally:
            self.logger.info(f"Worker {worker_id} process exiting")

    def _execute_task(self, message: Message, output_queue: mp.Queue) -> None:
        """Execute a task and send the result."""
        try:
            task_data = _as_payload_map(message.data)
            task_type_value = task_data.get("type")
            task_type = task_type_value if isinstance(task_type_value, str) else ""
            task_payload = _as_payload_map(task_data.get("data"))

            handler = self.task_handlers.get(task_type)
            if handler is None:
                raise ValueError(f"Unknown task type: {task_type}")

            # Execute task
            with self._state_lock:
                self.status = "busy"
            result = handler(task_payload)

            # Send result
            result_msg = Message("task_result", self.worker_id, result)
            output_queue.put(result_msg, timeout=1.0)

            with self._state_lock:
                self.stats["tasks_executed"] += 1
                self.status = "idle"

        except Exception as e:
            error_msg = Message(
                "error",
                self.worker_id,
                {
                    "error": str(e),
                    "traceback": traceback.format_exc(),
                    "task_data": message.data,
                },
            )
            try:
                output_queue.put(error_msg, timeout=1.0)
            except queue.Full:
                pass  # Can't report error if queue is full

            with self._state_lock:
                self.stats["tasks_failed"] += 1
                self.status = "idle"
            self.logger.error(f"Task execution failed: {e}")

    def _handle_train_epoch(self, data: Mapping[str, object]) -> dict[str, object]:
        """Handle training epoch task."""
        self.logger.info(f"Executing training epoch with data keys: {list(data.keys())}")

        # Placeholder implementation - would integrate with actual training logic
        time.sleep(0.1)  # Simulate computation time

        epoch = data.get("epoch")
        epoch_value = int(epoch) if isinstance(epoch, (int, float)) else 0
        return {
            "epoch": epoch_value,
            "loss": 0.5,  # Mock loss value
            "accuracy": 0.85,  # Mock accuracy
            "completed": True,
        }

    def _handle_evaluate_model(self, data: Mapping[str, object]) -> dict[str, object]:
        """Handle model evaluation task."""
        self.logger.info(f"Evaluating model with data keys: {list(data.keys())}")

        # Simulate evaluation work
        time.sleep(0.05)

        return {
            "validation_loss": 0.3,
            "validation_accuracy": 0.90,
            "evaluation_time": 0.05,
            "completed": True,
        }

    def _handle_sync_weights(self, data: Mapping[str, object]) -> dict[str, object]:
        """Handle weight synchronization task."""
        _ = data  # payload reserved for future extension
        self.logger.info("Synchronizing model weights")

        # Simulate weight sync
        time.sleep(0.01)

        return {"weights_synced": True, "sync_time": 0.01, "completed": True}

    def _handle_shutdown(self, data: Mapping[str, object]) -> dict[str, object]:
        """Handle shutdown task."""
        _ = data
        return {"shutdown": True}

    def _signal_handler(self, signum: int, frame: Optional[FrameType]) -> None:
        """Handle termination signals."""
        _ = frame
        self.logger.info(f"Worker {self.worker_id} received signal {signum}")
        # The main loop will handle cleanup


class WorkerPool(BackgroundThreadController):
    """Pool of distributed workers for load balancing and fault tolerance."""

    def __init__(self, *args, config: Optional[DistributedConfig] = None):
        """Initialize WorkerPool with flexible signature.

        Supports both:
            WorkerPool(num_workers, config)
            WorkerPool(config=config)
        """
        # Backwards-compatible parsing of positional args
        num_workers = 4
        if len(args) == 1 and isinstance(args[0], int):
            num_workers = args[0]
        elif len(args) >= 2 and isinstance(args[0], int):
            num_workers = args[0]
            config = args[1]

        self.config = config or DistributedConfig()
        self.config.num_workers = num_workers
        self.logger = logging.getLogger(__name__)

        self.workers: dict[int, DistributedWorker] = {}
        self.worker_lock = threading.RLock()

        # Load balancing
        self.task_queue: queue.Queue[TaskInfo] = queue.Queue(
            maxsize=self.config.max_queue_size
        )
        self.result_queue: queue.Queue[TaskResultInfo] = queue.Queue(
            maxsize=self.config.max_queue_size
        )
        self._round_robin_index = 0

        # Threading
        self._running = False
        self._dispatcher_thread: Optional[threading.Thread] = None
        self._task_executor: Optional[ThreadPoolExecutor] = None

    def start_pool(self) -> bool:
        """Start all workers in the pool."""
        if self._running:
            return False

        self.logger.info(f"Starting worker pool with {self.config.num_workers} workers")

        try:
            for i in range(self.config.num_workers):
                worker = DistributedWorker(i + 1, self.config)
                if worker.start():
                    self.workers[i + 1] = worker
                else:
                    self.logger.error(f"Failed to start worker {i + 1}")

            if not self.workers:
                self.logger.error("No workers started successfully")
                return False

            self._running = True
            self._task_executor = ThreadPoolExecutor(
                max_workers=max(1, len(self.workers)),
                thread_name_prefix="distributed-worker",
            )

            self._start_background_thread(
                attr_name="_dispatcher_thread",
                target=self._dispatch_loop,
                name="task-dispatcher",
                daemon=True,
            )

            self.logger.info(f"Worker pool started with {len(self.workers)} workers")
            return True

        except Exception as e:
            self.logger.error(f"Failed to start worker pool: {e}")
            self.stop_pool()
            return False

    def stop_pool(self) -> None:
        """Stop all workers in the pool."""
        if not self._running and not self.workers:
            return

        self.logger.info("Stopping worker pool")
        self._running = False

        self._join_background_thread(attr_name="_dispatcher_thread", timeout=5.0)

        # Stop all workers
        for worker in self.workers.values():
            worker.stop()
        self.workers.clear()

        if self._task_executor:
            self._task_executor.shutdown(wait=True)
            self._task_executor = None

    def submit_task(
        self, task_type: str, task_data: object, callback: Optional[TaskCallback] = None
    ) -> Optional[str]:
        """Submit a task to the worker pool."""
        if not self._running:
            return None

        task_id = f"{task_type}_{int(time.time() * 1000)}_{os.getpid()}"
        task_info: TaskInfo = {
            "task_id": task_id,
            "type": task_type,
            "data": task_data,
            "callback": callback,
            "submitted_at": datetime.now(),
        }

        try:
            self.task_queue.put(task_info, timeout=1.0)
            return task_id
        except queue.Full:
            self.logger.error("Task queue full")
            return None

    def get_pool_status(self) -> dict[str, object]:
        """Get status of the worker pool."""
        with self.worker_lock:
            worker_statuses: dict[int, WorkerStatusSnapshot] = {
                worker_id: worker.get_status()
                for worker_id, worker in self.workers.items()
            }

            return {
                "pool_running": self._running,
                "num_workers": len(self.workers),
                "active_workers": self._count_workers_with_status(
                    worker_statuses, "busy"
                ),
                "idle_workers": self._count_workers_with_status(worker_statuses, "idle"),
                "failed_workers": self._count_workers_with_status(
                    worker_statuses, "error"
                ),
                "queued_tasks": self.task_queue.qsize(),
                "pending_results": self.result_queue.qsize(),
                "worker_details": worker_statuses,
            }

    @staticmethod
    def _count_workers_with_status(
        worker_statuses: Mapping[int, WorkerStatusSnapshot], status: str
    ) -> int:
        return sum(
            1 for worker in worker_statuses.values() if worker.get("status") == status
        )

    def _dispatch_loop(self) -> None:
        """Main task dispatching loop."""
        while self._running:
            try:
                # Get next task
                try:
                    task_info = self.task_queue.get(timeout=0.1)
                except queue.Empty:
                    continue

                # Find available worker
                available_worker = self._get_available_worker()
                if not available_worker:
                    # No worker available, put task back
                    try:
                        self.task_queue.put(task_info, timeout=0.1)
                    except queue.Full:
                        self.logger.error("Task queue full when re-queuing")
                    time.sleep(0.1)
                    continue

                executor = self._task_executor
                if executor is None:
                    self.logger.error("Task executor is not initialized")
                    time.sleep(0.1)
                    continue

                task_id = task_info["task_id"]
                user_callback = task_info["callback"]
                worker_id = available_worker.worker_id

                future: Future[object | None] = executor.submit(
                    self._execute_task_async, available_worker, task_info
                )
                future.add_done_callback(
                    lambda done, tid=task_id, wid=worker_id, cb=user_callback: self._on_task_done(
                        done, tid, wid, cb
                    )
                )

            except Exception as e:
                self.logger.error(f"Error in dispatch loop: {e}")
                time.sleep(1.0)

    def _get_available_worker(self) -> Optional[DistributedWorker]:
        """Get an available worker using round-robin load balancing."""
        with self.worker_lock:
            if not self.workers:
                return None

            worker_ids = sorted(self.workers.keys())
            total_workers = len(worker_ids)

            for offset in range(total_workers):
                index = (self._round_robin_index + offset) % total_workers
                worker = self.workers[worker_ids[index]]
                if worker.is_running and worker.status == "idle":
                    self._round_robin_index = (index + 1) % total_workers
                    return worker

            return None

    def _execute_task_async(
        self, worker: DistributedWorker, task_info: TaskInfo
    ) -> object | None:
        """Execute a task asynchronously."""
        return worker.send_task(task_info["type"], task_info["data"])

    def _on_task_done(
        self,
        future: Future[object | None],
        task_id: str,
        worker_id: int,
        callback: Optional[TaskCallback],
    ) -> None:
        """Handle finished task execution."""
        try:
            result = future.result()
        except Exception as e:
            self.logger.error(f"Async task execution failed: {e}")
            result = None

        result_info: TaskResultInfo = {
            "task_id": task_id,
            "result": result,
            "completed_at": datetime.now(),
            "worker_id": worker_id,
        }
        self._enqueue_result(result_info)

        if callback is not None:
            try:
                callback(result)
            except Exception as e:
                self.logger.error(f"Error in task callback: {e}")

    def _enqueue_result(self, result_info: TaskResultInfo) -> None:
        """Keep result queue bounded to avoid unbounded memory growth."""
        try:
            self.result_queue.put_nowait(result_info)
            return
        except queue.Full:
            pass

        try:
            self.result_queue.get_nowait()
        except queue.Empty:
            pass

        try:
            self.result_queue.put_nowait(result_info)
        except queue.Full:
            self.logger.warning("Result queue full, dropping latest result")
