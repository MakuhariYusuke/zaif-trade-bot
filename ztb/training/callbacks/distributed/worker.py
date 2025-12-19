#!/usr/bin/env python3
"""
Distributed Worker Implementation.

This module provides worker processes for distributed training,
handling task execution, communication with coordinator, and
fault tolerance.
"""

import logging
import multiprocessing as mp
import os
import queue
import signal
import threading
import time
import traceback
from datetime import datetime
from typing import Any, Callable, Dict, Optional

from .coordinator import DistributedConfig, Message


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
        self.stats = {
            "tasks_executed": 0,
            "tasks_failed": 0,
            "uptime": 0,
            "cpu_usage": 0.0,
            "memory_usage": 0.0,
        }

        # Task execution
        self.task_handlers: Dict[str, Callable] = {}
        self._register_default_handlers()

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
            self.is_running = True
            self.status = "idle"
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

            self.is_running = False
            self.status = "stopped"
            self.logger.info(f"Stopped worker {self.worker_id}")
            return True
        except Exception as e:
            self.logger.error(f"Error stopping worker {self.worker_id}: {e}")
            return False

    def send_task(
        self, task_type: str, task_data: Any, timeout: float = 10.0
    ) -> Optional[Any]:
        """Send a task to the worker and wait for result."""
        if not self.is_running:
            return None

        try:
            message = Message("task", 0, {"type": task_type, "data": task_data})
            self.input_queue.put(message, timeout=1.0)

            # Wait for response
            result_msg = self.output_queue.get(timeout=timeout)
            if result_msg.msg_type == "task_result":
                return result_msg.data
            elif result_msg.msg_type == "error":
                self.logger.error(
                    f"Task failed on worker {self.worker_id}: {result_msg.data}"
                )
                return None
            else:
                self.logger.warning(f"Unexpected message type: {result_msg.msg_type}")
                return None

        except queue.Full:
            self.logger.error("Input queue full")
            return None
        except queue.Empty:
            self.logger.error("Timeout waiting for task result")
            return None
        except Exception as e:
            self.logger.error(f"Error sending task to worker {self.worker_id}: {e}")
            return None

    def get_status(self) -> Dict[str, Any]:
        """Get the current status of the worker."""
        return {
            "worker_id": self.worker_id,
            "is_running": self.is_running,
            "status": self.status,
            "last_heartbeat": self.last_heartbeat.isoformat(),
            "stats": self.stats.copy(),
            "process_alive": self.process.is_alive() if self.process else False,
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
                    # Check for heartbeat timeout
                    current_time = time.time()
                    if current_time - last_heartbeat > config.heartbeat_interval:
                        # Send heartbeat
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
                    elif message.msg_type == "task":
                        self._execute_task(message, output_queue)
                    elif message.msg_type == "sync":
                        # Respond to sync request
                        sync_response = Message("sync_ack", worker_id, {})
                        output_queue.put(sync_response, timeout=0.1)

                    # Update stats
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
            task_data = message.data
            task_type = task_data.get("type")
            task_payload = task_data.get("data")

            if task_type not in self.task_handlers:
                raise ValueError(f"Unknown task type: {task_type}")

            # Execute task
            self.status = "busy"
            result = self.task_handlers[task_type](task_payload)

            # Send result
            result_msg = Message("task_result", self.worker_id, result)
            output_queue.put(result_msg, timeout=1.0)

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

            self.stats["tasks_failed"] += 1
            self.status = "idle"
            self.logger.error(f"Task execution failed: {e}")

    def _handle_train_epoch(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Handle training epoch task."""
        # Placeholder implementation - would integrate with actual training logic
        self.logger.info(f"Executing training epoch with data: {data.keys()}")

        # Simulate training work
        time.sleep(0.1)  # Simulate computation time

        return {
            "epoch": data.get("epoch", 0),
            "loss": 0.5,  # Mock loss value
            "accuracy": 0.85,  # Mock accuracy
            "completed": True,
        }

    def _handle_evaluate_model(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Handle model evaluation task."""
        self.logger.info(f"Evaluating model with data: {data.keys()}")

        # Simulate evaluation work
        time.sleep(0.05)

        return {
            "validation_loss": 0.3,
            "validation_accuracy": 0.90,
            "evaluation_time": 0.05,
            "completed": True,
        }

    def _handle_sync_weights(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Handle weight synchronization task."""
        self.logger.info("Synchronizing model weights")

        # Simulate weight sync
        time.sleep(0.01)

        return {"weights_synced": True, "sync_time": 0.01, "completed": True}

    def _handle_shutdown(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Handle shutdown task."""
        return {"shutdown": True}

    def _signal_handler(self, signum, frame) -> None:
        """Handle termination signals."""
        self.logger.info(f"Worker {self.worker_id} received signal {signum}")
        # The main loop will handle cleanup


class WorkerPool:
    """
    Pool of distributed workers for load balancing and fault tolerance.
    """

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

        self.workers: Dict[int, DistributedWorker] = {}
        self.worker_lock = threading.RLock()

        # Load balancing
        self.task_queue: queue.Queue = queue.Queue()
        self.result_queue: queue.Queue = queue.Queue()

        # Threading
        self._running = False
        self._dispatcher_thread: Optional[threading.Thread] = None

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

            # Start task dispatcher
            self._dispatcher_thread = threading.Thread(
                target=self._dispatch_loop, name="task-dispatcher", daemon=True
            )
            self._dispatcher_thread.start()

            self.logger.info(f"Worker pool started with {len(self.workers)} workers")
            return True

        except Exception as e:
            self.logger.error(f"Failed to start worker pool: {e}")
            self.stop_pool()
            return False

    def stop_pool(self) -> None:
        """Stop all workers in the pool."""
        if not self._running:
            return

        self.logger.info("Stopping worker pool")

        self._running = False

        # Stop all workers
        for worker in self.workers.values():
            worker.stop()

        self.workers.clear()

        # Wait for dispatcher
        if self._dispatcher_thread and self._dispatcher_thread.is_alive():
            self._dispatcher_thread.join(timeout=5.0)

    def submit_task(
        self, task_type: str, task_data: Any, callback: Optional[Callable] = None
    ) -> Optional[str]:
        """Submit a task to the worker pool."""
        if not self._running:
            return None

        task_id = f"{task_type}_{int(time.time() * 1000)}_{os.getpid()}"

        task_info = {
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

    def get_pool_status(self) -> Dict[str, Any]:
        """Get status of the worker pool."""
        with self.worker_lock:
            worker_statuses = {}
            for worker_id, worker in self.workers.items():
                worker_statuses[worker_id] = worker.get_status()

            return {
                "pool_running": self._running,
                "num_workers": len(self.workers),
                "active_workers": len(
                    [w for w in worker_statuses.values() if w["status"] == "busy"]
                ),
                "idle_workers": len(
                    [w for w in worker_statuses.values() if w["status"] == "idle"]
                ),
                "failed_workers": len(
                    [w for w in worker_statuses.values() if w["status"] == "error"]
                ),
                "queued_tasks": self.task_queue.qsize(),
                "pending_results": self.result_queue.qsize(),
                "worker_details": worker_statuses,
            }

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

                # Submit task to worker
                def result_callback(result):
                    """Handle task result."""
                    result_info = {
                        "task_id": task_info["task_id"],
                        "result": result,
                        "completed_at": datetime.now(),
                        "worker_id": available_worker.worker_id,
                    }
                    self.result_queue.put(result_info)

                    # Call user callback if provided
                    if task_info["callback"]:
                        try:
                            task_info["callback"](result)
                        except Exception as e:
                            self.logger.error(f"Error in task callback: {e}")

                # Submit task asynchronously
                threading.Thread(
                    target=self._execute_task_async,
                    args=(available_worker, task_info, result_callback),
                    daemon=True,
                ).start()

            except Exception as e:
                self.logger.error(f"Error in dispatch loop: {e}")
                time.sleep(1.0)

    def _get_available_worker(self) -> Optional[DistributedWorker]:
        """Get an available worker using round-robin load balancing."""
        with self.worker_lock:
            available_workers = [
                w for w in self.workers.values() if w.is_running and w.status == "idle"
            ]

            if not available_workers:
                return None

            # Simple round-robin - could be enhanced with more sophisticated balancing
            return available_workers[0]

    def _execute_task_async(
        self, worker: DistributedWorker, task_info: Dict[str, Any], callback: Callable
    ) -> None:
        """Execute a task asynchronously."""
        try:
            result = worker.send_task(task_info["type"], task_info["data"])
            callback(result)
        except Exception as e:
            self.logger.error(f"Async task execution failed: {e}")
            callback(None)
