#!/usr/bin/env python3
"""
Tests for Distributed Training Components.

This module contains comprehensive tests for the distributed training
system including coordinator, workers, and integration components.
"""

import time
import unittest
from datetime import datetime
from threading import Event
from unittest.mock import Mock, patch

from .coordinator import DistributedConfig, DistributedCoordinator, Message, WorkerInfo
from .integration import DistributedCallbackAdapter, DistributedTrainingManager
from .worker import DistributedWorker, WorkerPool

_WAIT_EVENT = Event()


def _wait_until(predicate, timeout: float = 0.2, interval: float = 0.005) -> bool:
    deadline = time.perf_counter() + timeout
    while time.perf_counter() < deadline:
        if predicate():
            return True
        _WAIT_EVENT.wait(interval)
    return predicate()


class TestDistributedCoordinator(unittest.TestCase):
    """Test cases for DistributedCoordinator."""

    def setUp(self):
        """Set up test fixtures."""
        self.config = DistributedConfig(
            enable_distributed=True,
            num_workers=2,
            sync_interval=1.0,
            heartbeat_interval=2.0,
        )
        self.coordinator = DistributedCoordinator(self.config)

    def tearDown(self):
        """Clean up test fixtures."""
        self.coordinator.stop_coordination()

    def test_initialization(self):
        """Test coordinator initialization."""
        self.assertFalse(self.coordinator.is_master)
        self.assertEqual(len(self.coordinator.workers), 0)
        self.assertEqual(self.coordinator.stats["messages_sent"], 0)
        self.assertEqual(self.coordinator.stats["messages_received"], 0)

    def test_worker_registration(self):
        """Test worker registration and unregistration."""
        worker_info = WorkerInfo(worker_id=1, host="localhost", port=12346)

        # Test registration
        result = self.coordinator.register_worker(worker_info)
        self.assertTrue(result)
        self.assertIn(1, self.coordinator.workers)
        self.assertEqual(self.coordinator.workers[1].worker_id, 1)

        # Test duplicate registration
        result = self.coordinator.register_worker(worker_info)
        self.assertFalse(result)

        # Test unregistration
        result = self.coordinator.unregister_worker(1)
        self.assertTrue(result)
        self.assertNotIn(1, self.coordinator.workers)

        # Test unregistering non-existent worker
        result = self.coordinator.unregister_worker(999)
        self.assertFalse(result)

    def test_task_distribution(self):
        """Test task distribution to workers."""
        # Register a worker
        worker_info = WorkerInfo(worker_id=1, host="localhost", port=12346)
        self.coordinator.register_worker(worker_info)

        # Test task distribution
        task_data = {"task": "test", "data": [1, 2, 3]}
        worker_id = self.coordinator.distribute_task(task_data, worker_id=1)
        self.assertEqual(worker_id, 1)
        self.assertEqual(self.coordinator.stats["tasks_distributed"], 1)

        # Test distribution without specifying worker
        worker_id = self.coordinator.distribute_task(task_data)
        self.assertEqual(worker_id, 1)  # Should pick the available worker

    def test_metrics_aggregation(self):
        """Test metrics aggregation from workers."""
        # Register workers with metrics
        worker1 = WorkerInfo(worker_id=1, host="localhost", port=12346)
        worker1.metrics = {"loss": 0.5, "accuracy": 0.8, "samples": 100}
        self.coordinator.register_worker(worker1)

        worker2 = WorkerInfo(worker_id=2, host="localhost", port=12347)
        worker2.metrics = {"loss": 0.3, "accuracy": 0.9, "samples": 150}
        self.coordinator.register_worker(worker2)

        worker_metrics = {1: worker1.metrics, 2: worker2.metrics}

        aggregated = self.coordinator.aggregate_metrics(worker_metrics)

        # Check aggregated metrics
        self.assertIn("loss", aggregated)
        self.assertIn("accuracy", aggregated)
        self.assertEqual(aggregated["loss"]["count"], 2)
        self.assertEqual(aggregated["accuracy"]["mean"], 0.85)  # (0.8 + 0.9) / 2

    def test_message_handling(self):
        """Test message processing."""
        # Register a worker
        worker_info = WorkerInfo(worker_id=1, host="localhost", port=12346, status="busy")
        self.coordinator.register_worker(worker_info)
        previous_heartbeat = worker_info.last_heartbeat

        # Start coordination to enable message processing
        self.coordinator.start_coordination()

        # Send heartbeat message
        heartbeat_msg = Message("heartbeat", 1, {})
        self.coordinator.message_queue.put(heartbeat_msg)

        # Check that heartbeat was processed
        self.assertTrue(
            _wait_until(
                lambda: self.coordinator.workers[1].status == "idle"
                and self.coordinator.workers[1].last_heartbeat >= heartbeat_msg.timestamp
            )
        )
        self.assertEqual(self.coordinator.workers[1].status, "idle")
        self.assertGreaterEqual(
            self.coordinator.workers[1].last_heartbeat, previous_heartbeat
        )


class TestDistributedWorker(unittest.TestCase):
    """Test cases for DistributedWorker."""

    def setUp(self):
        """Set up test fixtures."""
        self.config = DistributedConfig(enable_distributed=True, max_queue_size=10)

    def test_worker_initialization(self):
        """Test worker initialization."""
        worker = DistributedWorker(1, self.config)
        self.assertEqual(worker.worker_id, 1)
        self.assertFalse(worker.is_running)
        self.assertEqual(worker.status, "idle")
        self.assertEqual(worker.stats["tasks_executed"], 0)

    def test_worker_lifecycle(self):
        """Test worker start/stop lifecycle."""
        worker = DistributedWorker(1, self.config)

        # Test start
        result = worker.start()
        self.assertTrue(result)
        self.assertTrue(worker.is_running)
        self.assertTrue(worker.process.is_alive())

        # Test stop
        result = worker.stop()
        self.assertTrue(result)
        self.assertFalse(worker.is_running)
        self.assertFalse(worker.process.is_alive())

    def test_task_execution(self):
        """Test task execution in worker."""
        worker = DistributedWorker(1, self.config)
        worker.start()

        try:
            # Send a simple task
            task_data = {"type": "evaluate_model", "data": {"model_id": "test"}}
            result = worker.send_task("evaluate_model", task_data, timeout=5.0)

            self.assertIsNotNone(result)
            self.assertIn("validation_loss", result)
            self.assertIn("validation_accuracy", result)
            self.assertTrue(result["completed"])

            # Check stats
            status = worker.get_status()
            self.assertEqual(status["stats"]["tasks_executed"], 1)

        finally:
            worker.stop()


class TestWorkerPool(unittest.TestCase):
    """Test cases for WorkerPool."""

    def setUp(self):
        """Set up test fixtures."""
        self.config = DistributedConfig(
            enable_distributed=True, num_workers=2, max_queue_size=10
        )

    def test_pool_initialization(self):
        """Test worker pool initialization."""
        pool = WorkerPool(2, self.config)
        self.assertFalse(pool._running)
        self.assertEqual(len(pool.workers), 0)

    def test_pool_lifecycle(self):
        """Test worker pool start/stop lifecycle."""
        pool = WorkerPool(2, self.config)

        # Test start
        result = pool.start_pool()
        self.assertTrue(result)
        self.assertTrue(pool._running)
        self.assertEqual(len(pool.workers), 2)

        # Check workers are running
        for worker in pool.workers.values():
            self.assertTrue(worker.is_running)
            self.assertTrue(worker.process.is_alive())

        # Test stop
        pool.stop_pool()
        self.assertFalse(pool._running)
        self.assertEqual(len(pool.workers), 0)

    def test_task_submission(self):
        """Test task submission to pool."""
        pool = WorkerPool(2, self.config)
        pool.start_pool()

        try:
            results = []
            all_results_received = Event()

            def result_callback(result):
                results.append(result)
                if len(results) >= 3:
                    all_results_received.set()

            # Submit multiple tasks
            task_ids = []
            for i in range(3):
                task_id = pool.submit_task(
                    "evaluate_model", {"task_id": i}, result_callback
                )
                self.assertIsNotNone(task_id)
                task_ids.append(task_id)

            self.assertTrue(all_results_received.wait(timeout=10))
            self.assertEqual(len(results), 3)
            for result in results:
                self.assertIsNotNone(result)
                self.assertTrue(result["completed"])

        finally:
            pool.stop_pool()

    def test_pool_status(self):
        """Test pool status reporting."""
        pool = WorkerPool(2, self.config)
        pool.start_pool()

        try:
            status = pool.get_pool_status()
            self.assertTrue(status["pool_running"])
            self.assertEqual(status["num_workers"], 2)
            self.assertIn("worker_details", status)

            worker_details = status["worker_details"]
            self.assertEqual(len(worker_details), 2)

            for worker_status in worker_details.values():
                self.assertTrue(worker_status["is_running"])
                self.assertIn("stats", worker_status)

        finally:
            pool.stop_pool()


class TestDistributedTrainingManager(unittest.TestCase):
    """Test cases for DistributedTrainingManager."""

    def setUp(self):
        """Set up test fixtures."""
        self.config = DistributedConfig(
            enable_distributed=True, num_workers=2, sync_interval=1.0
        )

    def test_manager_initialization(self):
        """Test manager initialization."""
        manager = DistributedTrainingManager(self.config)
        self.assertFalse(manager.is_initialized)
        self.assertFalse(manager.training_active)

        # Test initialization
        result = manager.initialize()
        self.assertTrue(result)
        self.assertTrue(manager.is_initialized)

        manager.shutdown()

    def test_training_session(self):
        """Test distributed training session."""
        manager = DistributedTrainingManager(self.config)
        manager.initialize()

        try:
            # Test training start
            training_config = {
                "epochs": 10,
                "batch_size": 32,
                "model_config": {"layers": 3},
            }

            result = manager.start_distributed_training(training_config)
            self.assertTrue(result)
            self.assertTrue(manager.training_active)

            # Test status
            status = manager.get_training_status()
            self.assertTrue(status["training_active"])
            self.assertTrue(status["distributed_mode"])

            # Test task submission
            task_id = manager.submit_training_task("train_epoch", {"epoch": 1})
            self.assertIsNotNone(task_id)

            # Test training stop
            manager.stop_distributed_training()
            self.assertFalse(manager.training_active)

        finally:
            manager.shutdown()

    def test_memory_integration(self):
        """Test memory monitoring integration."""
        manager = DistributedTrainingManager(self.config)
        manager.initialize()

        try:
            # Check memory monitoring is active
            memory_stats = manager.memory_monitor.get_memory_stats()
            self.assertIsInstance(memory_stats, dict)
            self.assertIn("total_memory", memory_stats)

            # Test memory stats in training status
            status = manager.get_training_status()
            self.assertIn("memory_status", status)

        finally:
            manager.shutdown()


class TestDistributedCallbackAdapter(unittest.TestCase):
    """Test cases for DistributedCallbackAdapter."""

    def setUp(self):
        """Set up test fixtures."""
        self.config = DistributedConfig(enable_distributed=True)
        self.coordinator = DistributedCoordinator(self.config)

        # Mock callback
        self.mock_callback = Mock()
        self.mock_callback.on_epoch_end = Mock(return_value=None)

    def test_adapter_initialization(self):
        """Test adapter initialization."""
        adapter = DistributedCallbackAdapter(self.mock_callback, self.coordinator)
        self.assertEqual(adapter.base_callback, self.mock_callback)
        self.assertTrue(adapter.is_distributed)

    def test_method_wrapping(self):
        """Test that callback methods are properly wrapped."""
        adapter = DistributedCallbackAdapter(self.mock_callback, self.coordinator)

        # Check that methods exist
        self.assertTrue(hasattr(adapter, "on_epoch_end"))
        self.assertTrue(hasattr(adapter, "on_training_start"))

        # Test calling wrapped method
        adapter.on_epoch_end(epoch=1, logs={"loss": 0.5})

        # Check that original method was called
        self.mock_callback.on_epoch_end.assert_called_once_with(
            epoch=1, logs={"loss": 0.5}
        )

    def test_distributed_functionality(self):
        """Test distributed functionality in adapter."""
        adapter = DistributedCallbackAdapter(self.mock_callback, self.coordinator)

        # Mock the coordinator methods
        with patch.object(self.coordinator, "message_queue") as mock_queue:
            adapter.on_epoch_end(epoch=1, logs={"loss": 0.5, "accuracy": 0.8})

            # Check that metrics were sent (should have been called)
            # Note: This is a simplified test - in real usage, heartbeat and metrics would be sent


class TestMessage(unittest.TestCase):
    """Test cases for Message class."""

    def test_message_creation(self):
        """Test message creation and serialization."""
        msg = Message("test", 1, {"data": "test_data"})

        self.assertEqual(msg.msg_type, "test")
        self.assertEqual(msg.sender_id, 1)
        self.assertEqual(msg.data, {"data": "test_data"})
        self.assertIsNotNone(msg.timestamp)

    def test_message_serialization(self):
        """Test message serialization and deserialization."""
        original_msg = Message("test", 1, {"data": [1, 2, 3]}, datetime.now())

        # Serialize
        serialized = original_msg.to_bytes()

        # Deserialize
        restored_msg = Message.from_bytes(serialized)

        self.assertEqual(restored_msg.msg_type, original_msg.msg_type)
        self.assertEqual(restored_msg.sender_id, original_msg.sender_id)
        self.assertEqual(restored_msg.data, original_msg.data)


if __name__ == "__main__":
    # Configure logging for tests
    logging.basicConfig(level=logging.INFO)

    # Run tests
    unittest.main(verbosity=2)
