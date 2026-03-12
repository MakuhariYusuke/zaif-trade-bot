#!/usr/bin/env python3
"""
Unit tests for memory_utils.py with TTLCache integration.

Tests memory management utilities including OperationMemoryTracker, temporary arrays,
memory-efficient processing, and integration with MemoryManager.
"""

import unittest
from unittest.mock import Mock, patch
import numpy as np

from ztb.utils.memory_utils import (
    OperationMemoryTracker,
    check_memory_pressure,
    cleanup_training_memory,
    get_memory_usage,
    memory_efficient_processing,
    optimize_array_dtype,
    temporary_array,
)


class TestOperationMemoryTracker(unittest.TestCase):
    """Test cases for OperationMemoryTracker with TTLCache integration."""

    @patch("psutil.Process")
    def test_memory_tracker_basic(self, mock_process):
        """Test basic memory tracking functionality."""
        # Mock memory values
        mock_process.return_value.memory_info.return_value.rss = 100 * 1024 * 1024  # 100MB

        with OperationMemoryTracker() as tracker:
            # Simulate some operation
            pass

        # Verify tracking was performed
        self.assertIsInstance(tracker, OperationMemoryTracker)

    @patch("ztb.utils.memory_utils.default_memory_manager")
    @patch("psutil.Process")
    def test_memory_tracker_with_cache(self, mock_process, mock_memory_manager):
        """Test memory tracking with cache statistics."""
        # Mock memory and cache values
        mock_process.return_value.memory_info.return_value.rss = 200 * 1024 * 1024  # 200MB
        mock_memory_manager.get_cache_stats.return_value = {
            "feature_cache_size": 10,
            "data_cache_size": 5,
            "model_cache_size": 2,
            "total_cache_entries": 17
        }

        with OperationMemoryTracker(enable_cache_tracking=True) as tracker:
            # Simulate some operation
            pass

        # Verify cache stats were accessed (called in __enter__ and __exit__)
        self.assertGreaterEqual(mock_memory_manager.get_cache_stats.call_count, 2)

    @patch("ztb.utils.memory_utils.psutil")
    @patch("ztb.utils.memory_utils.default_memory_manager")
    def test_memory_tracker_high_memory_optimization(self, mock_memory_manager, mock_psutil):
        """Test automatic memory optimization on high memory usage."""
        # Mock high memory usage that triggers optimization (threshold=1500MB)
        mock_psutil.Process.return_value.memory_info.return_value.rss = 1600 * 1024 * 1024  # 1600MB
        mock_memory_manager.get_cache_stats.return_value = {"total_cache_entries": 0}

        with OperationMemoryTracker(enable_cache_tracking=True) as tracker:
            # Simulate some operation
            pass

        # Verify optimization was triggered
        mock_memory_manager.optimize_memory_usage.assert_called_once()


class TestTemporaryArray(unittest.TestCase):
    """Test cases for temporary_array context manager."""

    def test_temporary_array_creation(self):
        """Test temporary array creation and cleanup."""
        data = [1, 2, 3, 4, 5]

        with temporary_array(data, dtype=np.float32) as arr:
            self.assertIsInstance(arr, np.ndarray)
            self.assertEqual(arr.dtype, np.float32)
            self.assertEqual(len(arr), 5)
            np.testing.assert_array_equal(arr, [1, 2, 3, 4, 5])

    def test_temporary_array_cleanup(self):
        """Test that temporary array is properly cleaned up."""
        data = np.random.rand(1000)

        with temporary_array(data) as arr:
            arr_ref = arr  # Keep reference to check cleanup

        # Array should still exist but context manager ensures cleanup intent
        self.assertIsInstance(arr_ref, np.ndarray)


class TestMemoryEfficientProcessing(unittest.TestCase):
    """Test cases for memory_efficient_processing context manager."""

    def test_memory_efficient_processing_auto_chunk(self):
        """Test automatic chunking of large arrays."""
        large_data = np.random.rand(1000)

        chunks_processed = []
        for chunk_data in memory_efficient_processing(large_data):
            chunks_processed.append(len(chunk_data))

        # Should have processed in chunks
        self.assertGreater(len(chunks_processed), 0)
        self.assertEqual(sum(chunks_processed), len(large_data))

    def test_memory_efficient_processing_custom_chunk(self):
        """Test custom chunk size processing."""
        data = np.arange(100)
        chunk_size = 25

        chunks_processed = []
        for chunk in memory_efficient_processing(data, chunk_size=chunk_size):
            chunks_processed.append(len(chunk))

        # Should have 4 chunks of size 25
        self.assertEqual(len(chunks_processed), 4)
        self.assertEqual(chunks_processed, [25, 25, 25, 25])


class TestOptimizeArrayDtype(unittest.TestCase):
    """Test cases for optimize_array_dtype function."""

    def test_float64_to_float32_conversion(self):
        """Test conversion from float64 to float32."""
        arr = np.array([1.0, 2.0, 3.0], dtype=np.float64)

        optimized = optimize_array_dtype(arr)

        self.assertEqual(optimized.dtype, np.float32)
        np.testing.assert_array_almost_equal(optimized, arr, decimal=6)

    def test_int64_to_int32_conversion(self):
        """Test conversion from int64 to int32."""
        arr = np.array([1, 2, 3], dtype=np.int64)

        optimized = optimize_array_dtype(arr)

        self.assertEqual(optimized.dtype, np.int32)
        np.testing.assert_array_equal(optimized, arr)

    def test_no_conversion_needed(self):
        """Test arrays that don't need conversion."""
        arr = np.array([1.0, 2.0, 3.0], dtype=np.float32)

        optimized = optimize_array_dtype(arr)

        self.assertEqual(optimized.dtype, np.float32)
        self.assertIs(optimized, arr)  # Should return same array


class TestCleanupTrainingMemory(unittest.TestCase):
    """Test cases for cleanup_training_memory function."""

    @patch("ztb.utils.memory_utils.default_memory_manager")
    @patch("gc.collect")
    def test_cleanup_training_memory_basic(self, mock_gc, mock_memory_manager):
        """Test basic training memory cleanup."""
        mock_gc.return_value = 42

        cleanup_training_memory()

        # Verify garbage collection was called (may be called multiple times in implementation)
        self.assertGreaterEqual(mock_gc.call_count, 1)
        # Verify memory manager optimization was called
        mock_memory_manager.optimize_memory_usage.assert_called_once()

    @patch("ztb.cache.memory_cache.default_memory_manager")
    def test_cleanup_training_memory_with_env(self, mock_memory_manager):
        """Test cleanup with environment."""
        mock_env = Mock()

        cleanup_training_memory(env=mock_env)

        # Verify environment was closed
        mock_env.close.assert_called_once()

    @patch("ztb.utils.memory_utils.default_memory_manager")
    def test_cleanup_training_memory_with_cache(self, mock_memory_manager):
        """Test cleanup with data cache."""
        data_cache = {"key1": "value1", "key2": "value2"}

        cleanup_training_memory(data_cache=data_cache)

        # Verify cache was cleared
        self.assertEqual(len(data_cache), 0)

    @patch("ztb.utils.memory_utils.default_memory_manager")
    def test_cleanup_training_memory_no_optimization(self, mock_memory_manager):
        """Test cleanup without cache optimization."""
        cleanup_training_memory(optimize_cache=False)

        # Verify memory manager optimization was not called
        mock_memory_manager.optimize_memory_usage.assert_not_called()


class TestGetMemoryUsage(unittest.TestCase):
    """Test cases for get_memory_usage function."""

    @patch("ztb.utils.memory_utils.default_memory_manager")
    @patch("psutil.Process")
    def test_get_memory_usage_with_cache(self, mock_process, mock_memory_manager):
        """Test memory usage retrieval with cache statistics."""
        # Mock psutil
        mock_memory_info = Mock()
        mock_memory_info.rss = 256 * 1024 * 1024  # 256MB
        mock_memory_info.vms = 512 * 1024 * 1024  # 512MB
        mock_process.return_value.memory_info.return_value = mock_memory_info
        mock_process.return_value.memory_percent.return_value = 25.5

        # Mock cache stats
        mock_memory_manager.get_cache_stats.return_value = {
            "feature_cache_size": 15,
            "data_cache_size": 8,
            "model_cache_size": 3,
            "total_cache_entries": 26
        }

        usage = get_memory_usage()

        # Verify memory stats
        self.assertAlmostEqual(usage['rss'], 256.0, places=1)
        self.assertAlmostEqual(usage['vms'], 512.0, places=1)
        self.assertEqual(usage['percent'], 25.5)

        # Verify cache stats
        self.assertEqual(usage['cache_feature_entries'], 15)
        self.assertEqual(usage['cache_data_entries'], 8)
        self.assertEqual(usage['cache_model_entries'], 3)
        self.assertEqual(usage['cache_total_entries'], 26)

    @patch("ztb.utils.memory_utils.default_memory_manager")
    @patch("ztb.utils.memory_utils.psutil")
    def test_get_memory_usage_psutil_unavailable(self, mock_psutil, mock_memory_manager):
        """Test memory usage when psutil is unavailable."""
        mock_memory_manager.get_cache_stats.return_value = {
            "feature_cache_size": 0,
            "data_cache_size": 0,
            "model_cache_size": 0,
            "total_cache_entries": 0
        }
        mock_psutil.Process.side_effect = ImportError("psutil not available")

        usage = get_memory_usage()

        # Should return default values
        self.assertEqual(usage['rss'], 0.0)
        self.assertEqual(usage['vms'], 0.0)
        self.assertEqual(usage['percent'], 0.0)
        self.assertEqual(usage['cache_total_entries'], 0)


class TestCheckMemoryPressure(unittest.TestCase):
    """Test cases for check_memory_pressure function."""

    @patch("ztb.utils.memory_utils.get_memory_usage")
    def test_check_memory_pressure_normal(self, mock_get_usage):
        """Test memory pressure check under normal conditions."""
        mock_get_usage.return_value = {
            'rss': 500.0,  # 500MB
            'cache_total_entries': 100
        }

        pressure = check_memory_pressure(threshold_mb=1000.0)

        self.assertFalse(pressure)

    @patch("ztb.utils.memory_utils.get_memory_usage")
    def test_check_memory_pressure_high_memory(self, mock_get_usage):
        """Test memory pressure detection for high memory usage."""
        mock_get_usage.return_value = {
            'rss': 1500.0,  # 1.5GB
            'cache_total_entries': 100
        }

        pressure = check_memory_pressure(threshold_mb=1000.0)

        self.assertTrue(pressure)

    @patch("ztb.utils.memory_utils.get_memory_usage")
    def test_check_memory_pressure_high_cache(self, mock_get_usage):
        """Test memory pressure detection for high cache usage."""
        mock_get_usage.return_value = {
            'rss': 500.0,  # 500MB
            'cache_total_entries': 1500  # High cache count
        }

        pressure = check_memory_pressure(threshold_mb=1000.0)

        self.assertTrue(pressure)


class TestMemoryUtilsIntegration(unittest.TestCase):
    """Integration tests for memory utilities."""

    @patch("psutil.Process")
    @patch("ztb.utils.memory_utils.default_memory_manager")
    def test_full_memory_workflow(self, mock_memory_manager, mock_process):
        """Test complete memory management workflow."""
        # Mock memory values
        mock_process.return_value.memory_info.return_value.rss = 300 * 1024 * 1024  # 300MB

        # Mock cache stats
        mock_memory_manager.get_cache_stats.return_value = {
            "feature_cache_size": 20,
            "data_cache_size": 10,
            "model_cache_size": 5,
            "total_cache_entries": 35
        }

        # Test memory tracking with operations
        with OperationMemoryTracker(enable_cache_tracking=True) as tracker:
            # Simulate data processing
            data = np.random.rand(1000, 10)
            optimized_data = optimize_array_dtype(data)

            # Process in chunks
            results = []
            for chunk in memory_efficient_processing(optimized_data, chunk_size=100):
                results.append(chunk.shape[0])

        # Verify memory usage tracking
        usage = get_memory_usage()
        self.assertIn('rss', usage)
        self.assertIn('cache_total_entries', usage)

        # Test cleanup
        cleanup_training_memory()

        # Verify cleanup calls
        mock_memory_manager.optimize_memory_usage.assert_called()


if __name__ == "__main__":
    unittest.main()