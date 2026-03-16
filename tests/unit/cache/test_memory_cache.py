#!/usr/bin/env python3
"""
Unit tests for memory_cache.py - TTLCache memory management system.

Tests MemoryManager, DynamicBufferManager, and cache functionality.
"""

import time
import unittest
from unittest.mock import Mock, patch
import numpy as np
import pandas as pd

from ztb.cache.memory_cache import (
    MemoryManager,
    DynamicBufferManager,
    get_memory_stats,
    optimize_memory,
)
from ztb.trading.environment.constants import BYTES_PER_MB


class TestMemoryManager(unittest.TestCase):
    """Test cases for MemoryManager class."""

    def setUp(self):
        """Set up test fixtures."""
        self.manager = MemoryManager(max_memory_mb=100.0, enable_monitoring=False)

    def tearDown(self):
        """Clean up after tests."""
        self.manager.stop_memory_monitoring()

    def test_memory_manager_initialization(self):
        """Test MemoryManager initialization."""
        self.assertIsInstance(self.manager, MemoryManager)
        self.assertEqual(self.manager.max_memory_mb, 100.0)
        self.assertFalse(self.manager.enable_monitoring)

    @patch("psutil.Process")
    def test_get_memory_usage_with_psutil(self, mock_process):
        """Test memory usage retrieval with psutil available."""
        mock_memory_info = Mock()
        mock_memory_info.rss = 64 * BYTES_PER_MB  # 64MB
        mock_memory_info.vms = 128 * BYTES_PER_MB  # 128MB
        mock_process.return_value.memory_info.return_value = mock_memory_info
        mock_process.return_value.cpu_percent.return_value = 15.5

        usage = self.manager.get_memory_usage()

        self.assertAlmostEqual(usage['rss_mb'], 64.0, places=1)
        self.assertAlmostEqual(usage['vms_mb'], 128.0, places=1)
        self.assertEqual(usage['cpu_percent'], 15.5)
        self.assertEqual(usage['cache_size'], 0)  # No cache entries yet

    def test_get_memory_usage_without_psutil(self):
        """Test memory usage retrieval when psutil is unavailable."""
        with patch.dict('sys.modules', {'psutil': None}):
            usage = self.manager.get_memory_usage()

        self.assertEqual(usage['rss_mb'], 0.0)
        self.assertEqual(usage['vms_mb'], 0.0)
        self.assertEqual(usage['cpu_percent'], 0.0)
        self.assertEqual(usage['cache_size'], 0)

    def test_cache_feature_data(self):
        """Test feature data caching."""
        test_data = {"feature1": [1, 2, 3], "feature2": [4, 5, 6]}

        self.manager.cache_feature_data("test_key", test_data)

        # Verify data was cached
        cached = self.manager.get_cached_feature_data("test_key")
        self.assertEqual(cached, test_data)

    def test_cache_training_data(self):
        """Test training data caching."""
        test_df = pd.DataFrame({"col1": [1, 2, 3], "col2": [4, 5, 6]})

        self.manager.cache_training_data("train_key", test_df)

        # Verify data was cached
        cached = self.manager.get_cached_training_data("train_key")
        pd.testing.assert_frame_equal(cached, test_df)

    def test_cache_model_state(self):
        """Test model state caching."""
        test_state = {"layer1": np.random.rand(10, 5), "layer2": np.random.rand(5, 1)}

        self.manager.cache_model_state("model_key", test_state)

        # Verify state was cached
        cached = self.manager.get_cached_model_state("model_key")
        self.assertEqual(cached, test_state)

    def test_cache_expiration(self):
        """Test cache expiration functionality."""
        test_data = {"data": "value"}
        clock = {"now": 1000.0}

        class FakeTTLCache(dict):
            def __init__(self, maxsize, ttl):
                super().__init__()
                self.maxsize = maxsize
                self.ttl = ttl
                self._timestamps = {}

            def __setitem__(self, key, value):
                super().__setitem__(key, value)
                self._timestamps[key] = clock["now"]

            def __getitem__(self, key):
                if clock["now"] - self._timestamps[key] > self.ttl:
                    super().__delitem__(key)
                    del self._timestamps[key]
                    raise KeyError(key)
                return super().__getitem__(key)

            def get(self, key, default=None):
                try:
                    return self[key]
                except KeyError:
                    return default

        with patch("ztb.cache.memory_cache.TTLCache", FakeTTLCache):
            manager = MemoryManager(max_memory_mb=100.0, enable_monitoring=False)

            manager.cache_feature_data("expire_key", test_data, ttl=1)

            # Should be available immediately
            cached = manager.get_cached_feature_data("expire_key")
            self.assertEqual(cached, test_data)

            # Advance the fake clock beyond TTL
            clock["now"] = 1001.1

            # Should be expired
            cached = manager.get_cached_feature_data("expire_key")
            self.assertIsNone(cached)

    def test_clear_expired_cache(self):
        """Test manual cache expiration clearing."""
        # Add some expired entries (simulate by directly manipulating timestamps)
        self.manager.feature_cache._timestamps["expired_key"] = time.time() - 10
        self.manager.feature_cache["expired_key"] = "expired_value"

        self.manager.clear_expired_cache()

        # Expired entry should be removed
        self.assertNotIn("expired_key", self.manager.feature_cache)

    def test_custom_ttl_cache_variants_are_bounded(self):
        """Distinct TTL buckets should not grow without bound."""
        self.manager.max_custom_ttl_caches = 2

        self.manager.cache_feature_data("ttl_10", {"v": 10}, ttl=10)
        self.manager.cache_feature_data("ttl_20", {"v": 20}, ttl=20)
        self.manager.cache_feature_data("ttl_30", {"v": 30}, ttl=30)

        self.assertLessEqual(len(self.manager._custom_ttl_caches), 2)
        self.assertNotIn(10.0, self.manager._custom_ttl_caches)

    def test_clear_expired_cache_prunes_empty_custom_ttl_buckets(self):
        """Expired custom TTL entries should drop empty bucket caches too."""
        clock = {"now": 1000.0}

        class FakeTTLCache(dict):
            def __init__(self, maxsize, ttl):
                super().__init__()
                self.maxsize = maxsize
                self.ttl = ttl
                self._timestamps = {}

            def __setitem__(self, key, value):
                super().__setitem__(key, value)
                self._timestamps[key] = clock["now"]

            def __getitem__(self, key):
                if clock["now"] - self._timestamps[key] > self.ttl:
                    super().__delitem__(key)
                    del self._timestamps[key]
                    raise KeyError(key)
                return super().__getitem__(key)

        with patch("ztb.cache.memory_cache.TTLCache", FakeTTLCache):
            manager = MemoryManager(max_memory_mb=100.0, enable_monitoring=False)
            manager.cache_feature_data("expire_key", {"v": 1}, ttl=1)
            self.assertEqual(len(manager._custom_ttl_caches), 1)

            clock["now"] = 1002.0
            manager.clear_expired_cache()
            self.assertEqual(len(manager._custom_ttl_caches), 0)

    def test_get_cache_stats(self):
        """Test cache statistics retrieval."""
        # Add some cache entries
        self.manager.cache_feature_data("feat1", [1, 2, 3])
        self.manager.cache_training_data("data1", pd.DataFrame())
        self.manager.cache_model_state("model1", {})

        stats = self.manager.get_cache_stats()

        self.assertEqual(stats['feature_cache_size'], 1)
        self.assertEqual(stats['data_cache_size'], 1)
        self.assertEqual(stats['model_cache_size'], 1)
        self.assertEqual(stats['total_cache_entries'], 3)

    @patch("gc.collect")
    def test_optimize_memory_usage(self, mock_gc):
        """Test memory optimization."""
        mock_gc.return_value = 25

        # Add some cache entries
        self.manager.cache_feature_data("test", [1, 2, 3])

        stats = self.manager.optimize_memory_usage()

        # Verify GC was called
        mock_gc.assert_called_once()

        # Verify stats were returned
        self.assertIn('rss_mb', stats)
        self.assertIn('cache_size', stats)


class TestDynamicBufferManager(unittest.TestCase):
    """Test cases for DynamicBufferManager class."""

    def setUp(self):
        """Set up test fixtures."""
        self.buffer_manager = DynamicBufferManager(
            initial_buffer_size=1000,
            max_buffer_size=10000
        )

    def test_buffer_manager_initialization(self):
        """Test DynamicBufferManager initialization."""
        self.assertEqual(self.buffer_manager.buffer_size, 1000)
        self.assertEqual(self.buffer_manager.max_buffer_size, 10000)
        self.assertEqual(self.buffer_manager.min_buffer_size, 100)

    def test_adjust_buffer_size_improvement(self):
        """Test buffer size adjustment on performance improvement."""
        # Add some baseline performance data first
        self.buffer_manager.performance_history = [8.0, 9.0, 10.0, 11.0, 12.0]

        # Simulate performance improvement (current > recent average)
        self.buffer_manager.adjust_buffer_size(15.0, 12.0)  # Current > Target = improvement

        # Buffer size should increase
        self.assertGreater(self.buffer_manager.buffer_size, 1000)

    def test_adjust_buffer_size_decline(self):
        """Test buffer size adjustment on performance decline."""
        # Add some baseline performance data first
        self.buffer_manager.performance_history = [12.0, 11.0, 10.0, 9.0, 8.0]

        # Simulate performance decline (current < recent average)
        self.buffer_manager.adjust_buffer_size(6.0, 10.0)  # Current < Target = decline

        # Buffer size should decrease
        self.assertLess(self.buffer_manager.buffer_size, 1000)

    def test_adjust_buffer_size_no_change(self):
        """Test buffer size adjustment with minimal performance change."""
        initial_size = self.buffer_manager.buffer_size

        # Minimal performance change (below threshold)
        self.buffer_manager.adjust_buffer_size(9.9, 10.0)

        # Buffer size should remain the same
        self.assertEqual(self.buffer_manager.buffer_size, initial_size)

    def test_get_optimal_buffer_size(self):
        """Test optimal buffer size calculation."""
        data_size = 5000

        optimal = self.buffer_manager.get_optimal_buffer_size(data_size)

        # Should be around 15% of data size
        expected = min(max(int(data_size * 0.15), 100), 10000)
        self.assertEqual(optimal, expected)

    def test_get_optimal_buffer_size_high_performance(self):
        """Test optimal buffer size with high performance history."""
        # Build performance history with good performance
        self.buffer_manager.performance_history = [15.0, 16.0, 17.0, 18.0, 19.0]

        data_size = 5000
        optimal = self.buffer_manager.get_optimal_buffer_size(data_size)

        # Should be around 15% of data size (750), within bounds
        self.assertGreaterEqual(optimal, self.buffer_manager.min_buffer_size)
        self.assertLessEqual(optimal, self.buffer_manager.max_buffer_size)

    def test_buffer_size_bounds(self):
        """Test buffer size stays within bounds."""
        # Try to exceed max
        self.buffer_manager.buffer_size = 15000
        optimal = self.buffer_manager.get_optimal_buffer_size(100000)
        self.assertLessEqual(optimal, self.buffer_manager.max_buffer_size)

        # Try to go below min - should be clamped to min_buffer_size
        self.buffer_manager.buffer_size = 10
        optimal = self.buffer_manager.get_optimal_buffer_size(10)
        # get_optimal_buffer_size should return at least min_buffer_size
        self.assertGreaterEqual(optimal, self.buffer_manager.min_buffer_size)


class TestGlobalFunctions(unittest.TestCase):
    """Test cases for global utility functions."""

    @patch("ztb.cache.memory_cache.default_memory_manager")
    def test_get_memory_stats(self, mock_memory_manager):
        """Test global memory stats function."""
        mock_memory_manager.get_memory_usage.return_value = {
            'rss_mb': 150.0, 'vms_mb': 300.0, 'cpu_percent': 20.0, 'cache_size': 5
        }
        mock_memory_manager.get_cache_stats.return_value = {
            'feature_cache_size': 3, 'data_cache_size': 1,
            'model_cache_size': 1, 'total_cache_entries': 5
        }
        mock_memory_manager.buffer_size = 2000

        stats = get_memory_stats()

        self.assertIn('memory', stats)
        self.assertIn('cache', stats)
        self.assertEqual(stats['buffer_size'], 2000)

    @patch("ztb.cache.memory_cache.default_memory_manager")
    def test_optimize_memory(self, mock_memory_manager):
        """Test global optimize_memory function."""
        mock_stats = {'rss_mb': 100.0, 'cache_size': 3}
        mock_memory_manager.optimize_memory_usage.return_value = mock_stats

        result = optimize_memory()

        self.assertEqual(result, mock_stats)
        mock_memory_manager.optimize_memory_usage.assert_called_once()


class TestMemoryCacheIntegration(unittest.TestCase):
    """Integration tests for memory cache system."""

    def test_memory_manager_workflow(self):
        """Test complete MemoryManager workflow."""
        manager = MemoryManager(enable_monitoring=False)

        # Cache different types of data
        feature_data = {"features": np.random.rand(100, 10)}
        training_data = pd.DataFrame(np.random.rand(1000, 5))
        model_state = {"weights": np.random.rand(50, 20)}

        manager.cache_feature_data("features", feature_data)
        manager.cache_training_data("training", training_data)
        manager.cache_model_state("model", model_state)

        # Verify all data is cached
        self.assertEqual(manager.get_cached_feature_data("features"), feature_data)
        pd.testing.assert_frame_equal(
            manager.get_cached_training_data("training"),
            training_data
        )
        self.assertEqual(manager.get_cached_model_state("model"), model_state)

        # Check cache stats
        stats = manager.get_cache_stats()
        self.assertEqual(stats['total_cache_entries'], 3)

        # Test memory optimization
        opt_stats = manager.optimize_memory_usage()
        self.assertIsInstance(opt_stats, dict)

    def test_buffer_manager_adaptation(self):
        """Test DynamicBufferManager adaptation over time."""
        buffer_manager = DynamicBufferManager()

        # Simulate varying performance with enough baseline data
        baseline_performances = [10.0, 10.5, 11.0, 11.5, 12.0]  # Build baseline
        for perf in baseline_performances:
            buffer_manager.adjust_buffer_size(perf, 10.0)

        # Now test with improving trend
        improving_performances = [13.0, 14.0, 15.0]  # Improving trend
        for perf in improving_performances:
            buffer_manager.adjust_buffer_size(perf, 10.0)

        # Buffer should have increased due to performance improvement
        self.assertGreaterEqual(buffer_manager.buffer_size, 1000)

        # Test optimal size calculation
        optimal = buffer_manager.get_optimal_buffer_size(10000)
        self.assertGreaterEqual(optimal, buffer_manager.min_buffer_size)
        self.assertLessEqual(optimal, buffer_manager.max_buffer_size)


if __name__ == "__main__":
    unittest.main()
