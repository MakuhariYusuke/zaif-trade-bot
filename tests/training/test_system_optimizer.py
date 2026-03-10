"""
Tests for system-level optimizations.

This module tests the SystemOptimizer and related components for:
- Memory management and leak prevention
- CPU performance optimization
- I/O caching improvements
- Integration with UnifiedTrainer
"""

import time
import unittest
from unittest.mock import MagicMock, patch

import psutil
import torch
import torch.nn as nn

from ztb.optimization.system_optimizer import (
    MemoryOptimizer,
    PerformanceOptimizer,
    SystemOptimizer,
)


class SimpleTestModel(nn.Module):
    """Simple model for testing."""

    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(10, 1)

    def forward(self, x):
        return self.linear(x)


class TestSystemOptimizer(unittest.TestCase):
    """Test SystemOptimizer functionality."""

    def setUp(self):
        """Set up test fixtures."""
        self.config = {
            "enable_memory_tracking": True,
            "enable_performance_profiling": True,
            "enable_io_caching": True,
            "memory_threshold_mb": 50.0,
            "cache_ttl_seconds": 60,
            "gc_interval_steps": 10,
        }
        self.optimizer = SystemOptimizer(**self.config)

    def tearDown(self):
        """Clean up after tests."""
        if hasattr(self, "optimizer"):
            self.optimizer.reset_stats()

    def test_initialization(self):
        """Test SystemOptimizer initialization."""
        assert self.optimizer.enable_memory_tracking is True
        assert self.optimizer.enable_performance_profiling is True
        assert self.optimizer.enable_io_caching is True
        assert self.optimizer.memory_threshold_mb == 50.0
        assert self.optimizer.cache_ttl_seconds == 60
        assert self.optimizer.gc_interval_steps == 10

    def test_get_system_stats(self):
        """Test getting system statistics."""
        stats = self.optimizer.get_system_stats()

        expected_keys = [
            "step_counter",
            "memory_tracking_enabled",
            "performance_profiling_enabled",
            "io_caching_enabled",
            "cache_size",
            "cache_hits",
            "cache_misses",
            "cache_hit_rate",
        ]

        for key in expected_keys:
            assert key in stats

        # Memory stats are only present if memory_history has data
        if self.optimizer.memory_history:
            memory_keys = ["current_memory_mb", "peak_memory_mb", "avg_memory_mb"]
            for key in memory_keys:
                assert key in stats

        # Performance stats are only present if performance_history has data
        if self.optimizer.performance_history:
            perf_keys = ["avg_step_time", "avg_cpu_percent"]
            for key in perf_keys:
                assert key in stats

    def test_optimize_model_memory(self):
        """Test model memory optimization."""
        model = SimpleTestModel()

        # Test optimization
        optimized_model = self.optimizer.optimize_model_memory(model)

        # Model should be returned (may be modified in place)
        assert optimized_model is not None
        assert isinstance(optimized_model, nn.Module)

    def test_optimize_dataloader(self):
        """Test dataloader optimization."""
        # Create a simple dataset and dataloader
        dataset = torch.utils.data.TensorDataset(
            torch.randn(100, 10), torch.randn(100, 1)
        )
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=10)

        # Test optimization
        optimized_dataloader = self.optimizer.optimize_dataloader(dataloader)

        # Dataloader should be returned (may be modified in place)
        assert optimized_dataloader is not None

    def test_cache_io_operation(self):
        """Test I/O operation caching."""
        call_count = 0

        def test_operation(x):
            nonlocal call_count
            call_count += 1
            return x * 2

        # First call should execute operation
        result1 = self.optimizer.cache_io_operation("test_key", test_operation, 5)
        assert result1 == 10
        assert call_count == 1

        # Second call should use cache
        result2 = self.optimizer.cache_io_operation("test_key", test_operation, 5)
        assert result2 == 10
        assert call_count == 1  # Should not have increased

        # Different key should execute again
        result3 = self.optimizer.cache_io_operation("different_key", test_operation, 3)
        assert result3 == 6
        assert call_count == 2

    def test_optimize_training_step_context_manager(self):
        """Test training step optimization context manager."""
        initial_counter = self.optimizer.step_counter

        with self.optimizer.optimize_training_step("test_step"):
            # Simulate some work
            time.sleep(0.01)

        # Counter should be incremented
        assert self.optimizer.step_counter == initial_counter + 1

        # Performance history should have an entry
        assert len(self.optimizer.performance_history) > 0

    def test_reset_stats(self):
        """Test statistics reset."""
        # Add some fake data
        self.optimizer.step_counter = 5
        self.optimizer.memory_history = [100.0, 110.0]
        self.optimizer.performance_history = [{"step_time": 1.0, "cpu_percent": 50.0}]
        self.optimizer.cache_hits = 3
        self.optimizer.cache_misses = 2

        # Reset stats
        self.optimizer.reset_stats()

        # All stats should be reset
        assert self.optimizer.step_counter == 0
        assert len(self.optimizer.memory_history) == 0
        assert len(self.optimizer.performance_history) == 0
        assert self.optimizer.cache_hits == 0
        assert self.optimizer.cache_misses == 0


class TestMemoryOptimizer(unittest.TestCase):
    """Test MemoryOptimizer functionality."""

    def test_optimize_tensor_memory(self):
        """Test tensor memory optimization."""
        # Create float64 tensor
        tensor_f64 = torch.randn(100, 100, dtype=torch.float64)

        # Optimize
        optimized = MemoryOptimizer.optimize_tensor_memory(tensor_f64)

        # Should be contiguous and potentially lower precision
        assert optimized.is_contiguous()
        # Note: dtype conversion depends on value range

    def test_clear_gpu_cache(self):
        """Test GPU cache clearing."""
        # This should not raise an exception even without GPU
        MemoryOptimizer.clear_gpu_cache()

    def test_get_memory_usage(self):
        """Test memory usage retrieval."""
        stats = MemoryOptimizer.get_memory_usage()

        expected_keys = ["rss_mb", "vms_mb"]
        for key in expected_keys:
            assert key in stats
            assert isinstance(stats[key], float)
            assert stats[key] > 0

        # GPU stats may or may not be present
        if torch.cuda.is_available():
            assert "gpu_allocated_mb" in stats
            assert "gpu_reserved_mb" in stats


class TestPerformanceOptimizer(unittest.TestCase):
    """Test PerformanceOptimizer functionality."""

    def test_optimize_numpy_operations(self):
        """Test NumPy optimization."""
        # Should not raise exceptions
        PerformanceOptimizer.optimize_numpy_operations()

    def test_enable_torch_optimizations(self):
        """Test PyTorch optimization enabling."""
        # Should not raise exceptions
        PerformanceOptimizer.enable_torch_optimizations()

    def test_get_cpu_stats(self):
        """Test CPU statistics retrieval."""
        stats = PerformanceOptimizer.get_cpu_stats()

        expected_keys = ["cpu_percent", "cpu_count", "cpu_freq_current"]
        for key in expected_keys:
            assert key in stats
            assert isinstance(stats[key], (int, float))


class TestSystemOptimizerIntegration(unittest.TestCase):
    """Test SystemOptimizer integration with training components."""

    def setUp(self):
        """Set up integration test fixtures."""
        self.optimizer = SystemOptimizer(
            enable_memory_tracking=True,
            enable_performance_profiling=True,
            enable_io_caching=True,
        )

    def tearDown(self):
        """Clean up after tests."""
        self.optimizer.reset_stats()

    @patch("psutil.cpu_percent")
    def test_performance_tracking_during_training(self, mock_cpu_percent):
        """Test performance tracking during simulated training."""
        mock_cpu_percent.return_value = 75.0

        # Simulate multiple training steps
        for i in range(5):
            with self.optimizer.optimize_training_step(f"step_{i}"):
                time.sleep(0.01)  # Simulate work

        # Check that performance data was collected (may be more than expected due to implementation details)
        assert len(self.optimizer.performance_history) >= 5

        # Check performance data structure
        for entry in self.optimizer.performance_history:
            assert "step_time" in entry
            assert "cpu_percent" in entry
            assert "step_name" in entry
            assert "timestamp" in entry

    def test_memory_threshold_warnings(self):
        """Test memory threshold warning system."""
        # Mock high memory usage
        with patch.object(
            psutil.Process, "memory_info", return_value=MagicMock(rss=200 * 1024 * 1024)
        ):  # 200MB
            with self.optimizer.optimize_training_step("high_memory_step"):
                pass

        # Should have recorded high memory usage
        assert len(self.optimizer.memory_history) > 0
        assert self.optimizer.memory_history[-1] > self.optimizer.memory_threshold_mb

    def test_cache_performance_tracking(self):
        """Test cache performance tracking."""

        def dummy_operation(x):
            return x**2

        # Perform multiple cache operations
        for i in range(10):
            key = f"key_{i % 3}"  # Reuse some keys
            self.optimizer.cache_io_operation(key, dummy_operation, i)

        stats = self.optimizer.get_system_stats()

        # Should have cache statistics
        assert "cache_hits" in stats
        assert "cache_misses" in stats
        assert "cache_hit_rate" in stats
        assert stats["cache_hits"] + stats["cache_misses"] > 0


if __name__ == "__main__":
    unittest.main()
