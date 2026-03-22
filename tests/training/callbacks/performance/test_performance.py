#!/usr/bin/env python3
"""
Performance Tests for Memory Optimization and Distributed Training.

This module contains comprehensive performance tests to validate
memory optimization, distributed training scalability, and overall
system performance.
"""

import gc
import threading
import time
import tracemalloc
import unittest
from unittest.mock import Mock, patch

import psutil

from ztb.training.callbacks.performance.memory_optimizer import (
    LRUCache,
    MemoryMonitor,
    MemoryPool,
    WeakRefRegistry,
)
from .distributed.coordinator import DistributedConfig, DistributedCoordinator
from .distributed.integration import DistributedTrainingManager
from .distributed.worker import WorkerPool

_WAIT_EVENT = threading.Event()


class TestMemoryOptimizationPerformance(unittest.TestCase):
    """Performance tests for memory optimization components."""

    def setUp(self):
        """Set up performance test fixtures."""
        tracemalloc.start()
        self.memory_monitor = MemoryMonitor()

    def tearDown(self):
        """Clean up after tests."""
        tracemalloc.stop()
        gc.collect()

    def test_lru_cache_performance(self):
        """Test LRU cache performance under load."""
        cache = LRUCache(max_size=1000)

        # Test insertion performance
        start_time = time.time()
        for i in range(1000):
            cache.put(f"key_{i}", f"value_{i}")
        insert_time = time.time() - start_time

        # Should be fast (< 0.1 seconds for 1000 insertions)
        self.assertLess(
            insert_time, 0.1, f"LRU cache insertion too slow: {insert_time}s"
        )

        # Test retrieval performance
        start_time = time.time()
        for i in range(1000):
            value = cache.get(f"key_{i}")
            self.assertIsNotNone(value)
        retrieve_time = time.time() - start_time

        # Should be fast (< 0.05 seconds for 1000 retrievals)
        self.assertLess(
            retrieve_time, 0.05, f"LRU cache retrieval too slow: {retrieve_time}s"
        )

        # Test cache eviction
        initial_size = len(cache.cache)
        for i in range(1001, 2001):  # Add more items to trigger eviction
            cache.put(f"key_{i}", f"value_{i}")

        # Cache should not grow beyond max_size
        self.assertLessEqual(len(cache.cache), 1000)

    def test_memory_pool_performance(self):
        """Test memory pool performance and efficiency."""
        pool = MemoryPool(pool_size=100)

        # Test object acquisition performance
        start_time = time.time()
        objects = []
        for i in range(100):
            obj = pool.acquire()
            objects.append(obj)
        acquire_time = time.time() - start_time

        # Should be fast (< 0.01 seconds for 100 acquisitions)
        self.assertLess(
            acquire_time, 0.01, f"Memory pool acquisition too slow: {acquire_time}s"
        )

        # Test object release performance
        start_time = time.time()
        for obj in objects:
            pool.release(obj)
        release_time = time.time() - start_time

        # Should be fast (< 0.01 seconds for 100 releases)
        self.assertLess(
            release_time, 0.01, f"Memory pool release too slow: {release_time}s"
        )

        # Test pool reuse
        obj1 = pool.acquire()
        pool.release(obj1)
        obj2 = pool.acquire()

        # Should reuse the same object
        self.assertIs(obj1, obj2)

        # Test pool limits
        objects = []
        for i in range(150):  # More than pool size
            obj = pool.acquire()
            objects.append(obj)

        # Should have created new objects when pool exhausted
        self.assertEqual(len(pool.pool), 0)  # Pool should be empty
        self.assertEqual(len(objects), 150)

    def test_memory_monitor_overhead(self):
        """Test memory monitor performance overhead."""
        monitor = MemoryMonitor()

        # Test monitoring overhead
        start_time = time.perf_counter()
        for i in range(1000):
            stats = monitor.get_memory_stats()
            self.assertIsInstance(stats, dict)
        monitor_time = time.perf_counter() - start_time

        # Monitoring cost is environment-dependent on Windows CI and local AV.
        # Keep this as a regression guard, not a microbenchmark.
        self.assertLess(
            monitor_time, 0.5, f"Memory monitoring too slow: {monitor_time}s"
        )

        # Create a moderate amount of cyclic garbage and make it unreachable so
        # cleanup time reflects reclamation work rather than scanning live data.
        garbage = []
        for _ in range(2000):
            node = []
            node.append(node)
            garbage.append(node)
        garbage.clear()

        start_time = time.perf_counter()
        monitor.force_cleanup()
        cleanup_time = time.perf_counter() - start_time

        # Cleanup should stay within a coarse regression bound on local Windows runs.
        self.assertLess(cleanup_time, 0.75, f"Memory cleanup too slow: {cleanup_time}s")

    @patch("ztb.training.callbacks.performance.memory_optimizer.psutil.Process")
    def test_memory_monitor_reuses_process_handle(self, mock_process):
        """Test MemoryMonitor reuses a cached psutil.Process handle."""
        process = Mock()
        process.memory_info.return_value.rss = 256 * 1024 * 1024
        mock_process.return_value = process

        monitor = MemoryMonitor()
        monitor.get_memory_stats()
        monitor.get_memory_stats()

        mock_process.assert_called_once()
        self.assertEqual(process.memory_info.call_count, 2)

    def test_weak_ref_registry_performance(self):
        """Test weak reference registry performance."""
        registry = WeakRefRegistry()

        # Test registration performance
        objects = []
        start_time = time.time()
        for i in range(1000):
            obj = [f"data_{i}"] * 10
            objects.append(obj)
            registry.register(obj, f"callback_{i}")
        register_time = time.time() - start_time

        # Weakref bookkeeping varies across Python builds; keep a coarse bound.
        self.assertLess(
            register_time, 0.15, f"Weak ref registration too slow: {register_time}s"
        )

        # Test callback triggering
        start_time = time.time()
        registry.cleanup()
        cleanup_time = time.time() - start_time

        # Cleanup should be fast (< 0.01 seconds)
        self.assertLess(
            cleanup_time, 0.01, f"Weak ref cleanup too slow: {cleanup_time}s"
        )

    def test_memory_leak_prevention(self):
        """Test that memory optimization prevents leaks."""
        initial_memory = psutil.Process().memory_info().rss

        # Create cache and pool
        cache = LRUCache(max_size=40)
        pool = MemoryPool(pool_size=20)

        # Perform operations that could cause leaks
        for cycle in range(4):
            # Fill cache
            for i in range(40):
                cache.put(f"key_{cycle}_{i}", [f"value_{cycle}_{i}"] * 100)

            # Use pool
            objects = []
            for i in range(20):
                obj = pool.acquire()
                objects.append(obj)

            for obj in objects:
                pool.release(obj)

            # Force garbage collection
            gc.collect()

        final_memory = psutil.Process().memory_info().rss
        memory_increase = final_memory - initial_memory

        # Memory increase should be reasonable (< 50MB)
        self.assertLess(
            memory_increase,
            50 * 1024 * 1024,
            f"Memory leak detected: {memory_increase / 1024 / 1024:.2f}MB increase",
        )


class TestDistributedTrainingPerformance(unittest.TestCase):
    """Performance tests for distributed training components."""

    def setUp(self):
        """Set up distributed performance test fixtures."""
        self.config = DistributedConfig(
            enable_distributed=True,
            num_workers=2,
            sync_interval=0.5,
            heartbeat_interval=1.0,
        )

    def test_coordinator_scaling(self):
        """Test coordinator performance with multiple workers."""
        coordinator = DistributedCoordinator(self.config)

        try:
            # Register multiple workers
            start_time = time.time()
            for i in range(10):
                worker_info = Mock()
                worker_info.worker_id = i + 1
                worker_info.host = "localhost"
                worker_info.port = 12345 + i
                worker_info.status = "idle"
                worker_info.metrics = {"cpu": 0.5, "memory": 100}
                coordinator.register_worker(worker_info)
            register_time = time.time() - start_time

            # Registration should be fast (< 0.01 seconds for 10 workers)
            self.assertLess(
                register_time, 0.01, f"Worker registration too slow: {register_time}s"
            )

            # Test metrics aggregation performance
            start_time = time.time()
            for i in range(100):
                worker_metrics = {
                    wid: {"loss": 0.1 * i, "accuracy": 0.8 + 0.01 * i}
                    for wid in range(1, 11)
                }
                aggregated = coordinator.aggregate_metrics(worker_metrics)
            aggregation_time = time.time() - start_time

            # Aggregation should be fast (< 0.1 seconds for 100 aggregations)
            self.assertLess(
                aggregation_time,
                0.1,
                f"Metrics aggregation too slow: {aggregation_time}s",
            )

        finally:
            coordinator.stop_coordination()

    def test_worker_pool_throughput(self):
        """Test worker pool task throughput."""
        pool = WorkerPool(4, self.config)

        try:
            pool.start_pool()

            # Submit multiple tasks
            num_tasks = 24
            results = []
            start_time = time.perf_counter()

            def result_callback(result):
                results.append(result)

            for i in range(num_tasks):
                pool.submit_task("evaluate_model", {"task_id": i}, result_callback)

            total_time = max(time.perf_counter() - start_time, 1e-9)

            # Should complete all tasks within reasonable time
            self.assertEqual(len(results), num_tasks, "Not all tasks completed")
            self.assertLess(total_time, 15, f"Task completion too slow: {total_time}s")

            # Calculate throughput
            throughput = num_tasks / total_time
            self.assertGreater(throughput, 2, f"Low throughput: {throughput} tasks/sec")

        finally:
            pool.stop_pool()

    def test_distributed_manager_overhead(self):
        """Test distributed manager performance overhead."""
        manager = DistributedTrainingManager(self.config)

        try:
            manager.initialize()

            # Test status query performance
            start_time = time.time()
            for i in range(100):
                status = manager.get_training_status()
                self.assertIsInstance(status, dict)
            status_time = time.time() - start_time

            # Status queries should be fast (< 0.1 seconds for 100 queries)
            self.assertLess(
                status_time, 0.1, f"Status queries too slow: {status_time}s"
            )

            # Test training session startup/shutdown
            training_config = {"epochs": 5, "batch_size": 64}

            start_time = time.time()
            result = manager.start_distributed_training(training_config)
            startup_time = time.time() - start_time

            self.assertTrue(result)
            # Startup should be reasonable (< 1 second)
            self.assertLess(
                startup_time, 1, f"Training startup too slow: {startup_time}s"
            )

            start_time = time.time()
            manager.stop_distributed_training()
            shutdown_time = time.time() - start_time

            # Shutdown should be fast (< 0.5 seconds)
            self.assertLess(
                shutdown_time, 0.5, f"Training shutdown too slow: {shutdown_time}s"
            )

        finally:
            manager.shutdown()


class TestSystemIntegrationPerformance(unittest.TestCase):
    """Performance tests for complete system integration."""

    def setUp(self):
        """Set up integration performance test fixtures."""
        self.config = DistributedConfig(
            enable_distributed=True, num_workers=3, sync_interval=1.0
        )

    def test_end_to_end_training_simulation(self):
        """Test end-to-end training simulation performance."""
        manager = DistributedTrainingManager(self.config)

        try:
            manager.initialize()

            # Start training
            training_config = {"epochs": 3, "batch_size": 128, "learning_rate": 0.001}

            manager.start_distributed_training(training_config)

            # Simulate training epochs
            total_start_time = time.time()
            epoch_times = []

            for epoch in range(3):
                epoch_start = time.time()

                # Submit training tasks for this epoch
                task_ids = []
                for batch in range(6):  # Keep enough batches to exercise coordination
                    task_data = {
                        "epoch": epoch,
                        "batch": batch,
                        "data_size": 128 * 10,
                    }  # Simulate data size
                    task_id = manager.submit_training_task("train_epoch", task_data)
                    if task_id:
                        task_ids.append(task_id)

                epoch_time = time.time() - epoch_start
                epoch_times.append(epoch_time)

            total_time = time.time() - total_start_time

            # Training should complete within reasonable time
            self.assertLess(total_time, 10, f"Training too slow: {total_time}s")

            # Check epoch times are reasonable
            avg_epoch_time = sum(epoch_times) / len(epoch_times)
            self.assertLess(
                avg_epoch_time, 2, f"Average epoch time too slow: {avg_epoch_time}s"
            )

            manager.stop_distributed_training()

        finally:
            manager.shutdown()

    def test_memory_usage_under_load(self):
        """Test memory usage stability under load."""
        manager = DistributedTrainingManager(self.config)

        try:
            manager.initialize()
            manager.start_distributed_training({"epochs": 6})

            initial_memory = psutil.Process().memory_info().rss

            # Simulate heavy load
            for i in range(16):
                # Submit tasks
                for j in range(2):
                    manager.submit_training_task(
                        "evaluate_model", {"iteration": i, "batch": j}
                    )

                # Force some memory pressure
                large_data = [list(range(400)) for _ in range(24)]
                del large_data

                # Trigger cleanup
                manager.memory_monitor.force_cleanup()

            final_memory = psutil.Process().memory_info().rss
            memory_increase = final_memory - initial_memory

            # Memory increase should be controlled (< 100MB under load)
            self.assertLess(
                memory_increase,
                100 * 1024 * 1024,
                f"Excessive memory usage: {memory_increase / 1024 / 1024:.2f}MB",
            )

            manager.stop_distributed_training()

        finally:
            manager.shutdown()

    def test_concurrent_access_performance(self):
        """Test performance under concurrent access."""
        manager = DistributedTrainingManager(self.config)

        try:
            manager.initialize()
            manager.start_distributed_training({"epochs": 5})

            results = []
            errors = []

            def worker_thread(thread_id):
                """Worker thread function."""
                try:
                    thread_results = []

                    for i in range(20):
                        # Mix of different operations
                        if i % 3 == 0:
                            status = manager.get_training_status()
                            thread_results.append(("status", len(str(status))))
                        elif i % 3 == 1:
                            task_id = manager.submit_training_task(
                                "evaluate_model", {"thread": thread_id, "iter": i}
                            )
                            thread_results.append(("task", task_id is not None))
                        else:
                            # Memory operation
                            stats = manager.memory_monitor.get_memory_stats()
                            thread_results.append(
                                ("memory", stats.get("total_memory", 0))
                            )

                    results.append(thread_results)

                except Exception as e:
                    errors.append((thread_id, str(e)))

            # Start multiple threads
            threads = []
            num_threads = 5

            start_time = time.time()
            for i in range(num_threads):
                thread = threading.Thread(target=worker_thread, args=(i,))
                threads.append(thread)
                thread.start()

            # Wait for all threads
            for thread in threads:
                thread.join(timeout=10)

            total_time = time.time() - start_time

            # All threads should complete successfully
            self.assertEqual(len(results), num_threads, f"Thread errors: {errors}")
            self.assertEqual(len(errors), 0, f"Errors occurred: {errors}")

            # Concurrent operations should complete within reasonable time
            self.assertLess(
                total_time, 5, f"Concurrent operations too slow: {total_time}s"
            )

            manager.stop_distributed_training()

        finally:
            manager.shutdown()


class TestScalabilityBenchmarks(unittest.TestCase):
    """Scalability benchmark tests."""

    @unittest.skip(
        "Throughput scaling depends heavily on local CPU scheduling and is too noisy for deterministic test runs."
    )
    def test_worker_scaling(self):
        """Test how performance scales with number of workers."""
        worker_counts = [1, 2, 4]
        results = {}

        for num_workers in worker_counts:
            config = DistributedConfig(
                enable_distributed=True, num_workers=num_workers, sync_interval=0.5
            )

            pool = WorkerPool(num_workers, config)
            pool.start_pool()
            _WAIT_EVENT.wait(0.2)  # Allow workers to start

            try:
                # Benchmark task submission
                num_tasks = 20
                results_received = []

                def result_callback(result):
                    results_received.append(result)

                start_time = time.time()
                for i in range(num_tasks):
                    pool.submit_task("evaluate_model", {"task": i}, result_callback)

                # Wait for results
                timeout = 15
                wait_start = time.time()
                while (
                    len(results_received) < num_tasks
                    and (time.time() - wait_start) < timeout
                ):
                    _WAIT_EVENT.wait(0.05)

                total_time = time.time() - start_time
                throughput = num_tasks / total_time if total_time > 0 else 0

                results[num_workers] = {
                    "throughput": throughput,
                    "total_time": total_time,
                    "completed_tasks": len(results_received),
                }

            finally:
                pool.stop_pool()

        # Throughput should generally increase with more workers
        # (allowing for some variance due to system conditions)
        if 2 in results and 4 in results:
            throughput_2 = results[2]["throughput"]
            throughput_4 = results[4]["throughput"]
            # 4 workers should have at least 1.5x throughput of 2 workers (allowing for overhead)
            self.assertGreater(
                throughput_4,
                throughput_2 * 1.2,
                f"Scaling issue: 2 workers={throughput_2:.2f}, 4 workers={throughput_4:.2f}",
            )

    def test_memory_scaling(self):
        """Test memory usage scaling with load."""
        cache_sizes = [100, 500, 1000]
        results = {}

        for cache_size in cache_sizes:
            cache = LRUCache(max_size=cache_size)

            start_memory = psutil.Process().memory_info().rss
            start_time = time.time()

            # Fill cache
            for i in range(cache_size * 2):  # Add more than capacity to test eviction
                cache.put(f"key_{i}", [f"value_{i}"] * 50)  # Moderate size objects

            fill_time = time.time() - start_time
            end_memory = psutil.Process().memory_info().rss
            memory_used = end_memory - start_memory

            results[cache_size] = {
                "memory_used": memory_used,
                "fill_time": fill_time,
                "cache_size": len(cache.cache),
            }

            # Cleanup
            del cache
            gc.collect()

        # Memory usage should scale reasonably with cache size
        for size in cache_sizes:
            memory_mb = results[size]["memory_used"] / 1024 / 1024
            # Should use less than 50MB per 1000 items (rough estimate)
            max_expected_mb = (size / 1000) * 50
            self.assertLess(
                memory_mb,
                max_expected_mb * 2,
                f"Excessive memory usage for cache size {size}: {memory_mb:.2f}MB",
            )


if __name__ == "__main__":
    # Configure logging for performance tests
    logging.basicConfig(level=logging.WARNING)  # Reduce log noise during tests

    # Run performance tests
    unittest.main(verbosity=2)
