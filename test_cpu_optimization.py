#!/usr/bin/env python3
"""
Test script for SAC v446 CPU optimization foundation.

This script validates the implementation of:
1. Parallel processing utilities
2. Memory management with TTLCache
3. Model compression with pruning
4. Dynamic buffer size adjustment
"""

import logging
import time
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_memory_management():
    """Test memory management functionality."""
    logger.info("Testing memory management...")

    from ztb.cache.memory_cache import MemoryManager, DynamicBufferManager, get_memory_stats

    # Test MemoryManager
    manager = MemoryManager(max_memory_mb=100.0)

    # Test caching
    test_data = {"feature1": [1, 2, 3, 4, 5], "feature2": [6, 7, 8, 9, 10]}
    manager.cache_feature_data("test_features", test_data)

    cached_data = manager.get_cached_feature_data("test_features")
    assert cached_data == test_data, "Feature data caching failed"

    # Test memory stats
    stats = manager.get_memory_usage()
    logger.info(f"Memory stats: {stats}")

    # Test global functions
    global_stats = get_memory_stats()
    logger.info(f"Global memory stats: {global_stats}")

    # Test DynamicBufferManager
    buffer_manager = DynamicBufferManager()
    optimal_size = buffer_manager.get_optimal_buffer_size(10000)
    logger.info(f"Optimal buffer size: {optimal_size}")

    logger.info("Memory management tests passed!")

def test_parallel_processing():
    """Test parallel processing utilities."""
    logger.info("Testing parallel processing...")

    from ztb.training.utils.parallel_utils import CPUParallelProcessor, DataLoaderParallelizer

    # Test CPUParallelProcessor
    processor = CPUParallelProcessor()

    def test_func(x):
        return x * 2

    test_data = list(range(10))
    results = processor.parallel_map(test_func, test_data)

    expected = [x * 2 for x in test_data]
    assert results == expected, f"Parallel map failed: {results} != {expected}"

    # Test batch processing
    def batch_func(batch):
        return [x * 2 for x in batch]

    batch_results = processor.parallel_batch_process(batch_func, test_data, batch_size=3)
    expected_batch = [[x * 2 for x in test_data[i:i+3]] for i in range(0, len(test_data), 3)]
    assert batch_results == expected_batch, f"Batch processing failed: {batch_results} != {expected_batch}"

    logger.info("Parallel processing tests passed!")

def test_cache_integration():
    """Test cache system integration."""
    logger.info("Testing cache integration...")

    from ztb.cache.sqlite_cache import SQLiteCache
    from ztb.cache.data_loader import DataLoader

    # Test SQLiteCache with memory integration
    cache = SQLiteCache(enable_memory_cache=True)

    # Test set/get
    test_value = {"test": "data", "numbers": [1, 2, 3]}
    cache.set("integration_test", test_value)

    retrieved = cache.get("integration_test")
    assert retrieved == test_value, "Cache integration failed"

    cache.close()

    # Test DataLoader with memory cache
    import tempfile
    import pandas as pd

    with tempfile.TemporaryDirectory() as temp_dir:
        loader = DataLoader(cache_dir=temp_dir, enable_memory_cache=True)

        # Create test DataFrame
        test_df = pd.DataFrame({"col1": [1, 2, 3], "col2": [4, 5, 6]})

        def load_test_data():
            return test_df

        # Test loading with cache
        loaded_df = loader.load_with_cache("test_df", load_test_data)
        assert loaded_df.equals(test_df), "DataLoader cache integration failed"

    logger.info("Cache integration tests passed!")

def test_model_compression():
    """Test model compression with memory caching."""
    logger.info("Testing model compression...")

    import torch
    import torch.nn as nn
    from ztb.training.model_compression import PruningCompressor

    # Create simple test model
    model = nn.Sequential(
        nn.Linear(10, 5),
        nn.ReLU(),
        nn.Linear(5, 1)
    )

    # Test pruning compressor
    compressor = PruningCompressor(pruning_type="l1_unstructured", amount=0.1)

    # Apply compression
    compressed_model = compressor.compress(model)

    # Verify model is still functional
    test_input = torch.randn(1, 10)
    output = compressed_model(test_input)
    assert output.shape == (1, 1), "Model compression broke model functionality"

    logger.info("Model compression tests passed!")

def main():
    """Run all CPU optimization tests."""
    logger.info("Starting SAC v446 CPU optimization foundation tests...")

    try:
        test_memory_management()
        test_parallel_processing()
        test_cache_integration()
        test_model_compression()

        logger.info("All CPU optimization tests passed! ✅")
        logger.info("Week 7-8 CPU optimization foundation is complete.")

    except Exception as e:
        logger.error(f"Test failed: {e}")
        raise

if __name__ == "__main__":
    main()