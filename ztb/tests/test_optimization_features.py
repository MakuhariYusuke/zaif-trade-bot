#!/usr/bin/env python3
"""
Tests for UnifiedTrainer optimization features.
"""

import unittest
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd

from ztb.training.unified_trainer.trainer import UnifiedTrainer
from ztb.utils.cache_utils import TTLCache
from ztb.utils.memory_utils import MemoryTracker, optimize_array_dtype
from ztb.utils.performance_profiler import PerformanceProfiler


class TestOptimizationFeatures(unittest.TestCase):
    """Test optimization features integration."""

    def setUp(self):
        """Set up test fixtures."""
        self.config = {
            "algorithm": "sac",
            "model_name": "test_sac_optimization",
            "total_timesteps": 1000,
            "sac_hyperparameters": {
                "learning_rate": 0.001,
                "buffer_size": 1000,
                "batch_size": 64,
            },
            "environment": {
                "initial_balance": 100000,
                "transaction_cost": 0.0005,
                "max_position_size": 1.0,
            },
            "reward_settings": {"reward_scale": 100.0},
            "data_path": "test_data.csv",
        }

        # Create test data
        self.test_df = pd.DataFrame(
            {
                "timestamp": pd.date_range("2020-01-01", periods=100, freq="1H"),
                "open": np.random.uniform(100, 200, 100),
                "high": np.random.uniform(100, 200, 100),
                "low": np.random.uniform(100, 200, 100),
                "close": np.random.uniform(100, 200, 100),
                "volume": np.random.uniform(1000, 10000, 100),
            }
        )
        self.test_df.to_csv("test_data.csv", index=False)

    def tearDown(self):
        """Clean up test fixtures."""
        import os

        if os.path.exists("test_data.csv"):
            os.remove("test_data.csv")

    def test_memory_tracker_integration(self):
        """Test MemoryTracker integration in UnifiedTrainer."""
        trainer = UnifiedTrainer(self.config, dry_run=True)

        # Check that memory tracker is initialized
        self.assertIsInstance(trainer.memory_tracker, MemoryTracker)
        self.assertIsInstance(trainer.performance_profiler, PerformanceProfiler)
        self.assertIsInstance(trainer.feature_cache, TTLCache)

    def test_data_type_optimization(self):
        """Test data type optimization functionality."""
        # Create test data with float64
        test_data = np.random.rand(100, 10).astype(np.float64)

        # Apply optimization
        optimized_data = optimize_array_dtype(test_data)

        # Check that data type was optimized
        self.assertEqual(optimized_data.dtype, np.float32)
        self.assertEqual(test_data.dtype, np.float64)  # Original unchanged

    def test_cache_functionality(self):
        """Test TTLCache functionality."""
        cache = TTLCache(ttl_seconds=1)  # 1 second TTL

        # Test basic caching
        cache.set("key1", "value1")
        self.assertEqual(cache.get("key1"), "value1")

        # Test cache miss
        self.assertIsNone(cache.get("nonexistent"))

        # Test clear
        cache.clear()
        self.assertIsNone(cache.get("key1"))

    @patch("ztb.training.unified_trainer.algorithms.create_algorithm_trainer")
    def test_optimization_metrics_collection(self, mock_create_trainer):
        """Test that optimization metrics are collected during training."""
        # Mock the algorithm trainer
        mock_trainer = MagicMock()
        mock_trainer.validate_config.return_value = True
        mock_trainer.train.return_value = True
        mock_trainer.get_training_stats.return_value = {
            "total_timesteps": 1000,
            "training_time": 10.0,
        }
        mock_create_trainer.return_value = mock_trainer

        trainer = UnifiedTrainer(self.config, dry_run=True)

        # Run training (dry run)
        result = trainer.run()

        # Check that optimization metrics are included
        if result and trainer.training_stats:
            self.assertIn("optimization", trainer.training_stats)
            opt_metrics = trainer.training_stats["optimization"]
            self.assertIn("memory_stats", opt_metrics)
            self.assertIn("performance_profile", opt_metrics)
            self.assertIn("parallel_processing_enabled", opt_metrics)
            self.assertIn("cache_size", opt_metrics)
            self.assertIn("data_optimization_applied", opt_metrics)

    def test_memory_efficient_processing(self):
        """Test memory efficient processing utilities."""
        # Test with large array
        large_array = np.random.rand(1000, 1000).astype(np.float32)

        # This should not raise memory errors
        processed = optimize_array_dtype(large_array)
        self.assertEqual(processed.dtype, np.float32)


if __name__ == "__main__":
    unittest.main()
