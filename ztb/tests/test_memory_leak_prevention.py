#!/usr/bin/env python3
"""Test script for memory leak prevention in advanced features."""

import gc

# Add project root to path
import sys
from pathlib import Path

import psutil

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd

from ztb.features.attention_trainer import AttentionTrainer
from ztb.features.causal_inference import CausalFeatureSelector
from ztb.trading.environment.components.memory_manager import MemoryManager


def get_memory_usage():
    """Get current memory usage in MB."""
    process = psutil.Process()
    return process.memory_info().rss / 1024 / 1024


def test_attention_trainer_memory_management():
    """Test memory management in attention trainer."""
    print("Testing Attention Trainer Memory Management...")

    # Create memory manager
    memory_manager = MemoryManager(memory_logging_enabled=True)

    # Create trainer
    trainer = AttentionTrainer(
        n_features=20, batch_size=16, memory_manager=memory_manager
    )

    initial_memory = get_memory_usage()
    print(f"Initial memory: {initial_memory:.1f} MB")

    # Add many training samples to test buffer management
    import numpy as np

    np.random.seed(42)

    for i in range(1200):  # More than buffer limit (10000/4 = 2500, but we use smaller)
        features = np.random.randn(20)
        reward = np.random.randn()
        regime = np.random.choice(
            ["trending", "ranging", "high_volatility", "low_volatility"]
        )
        trainer.add_training_sample(features, reward, regime)

        if i % 200 == 0:
            current_memory = get_memory_usage()
            print(
                f"After {i} samples: {current_memory:.1f} MB (diff: {current_memory - initial_memory:.1f} MB)"
            )

    final_memory = get_memory_usage()
    memory_increase = final_memory - initial_memory
    print(f"Final memory: {final_memory:.1f} MB (increase: {memory_increase:.1f} MB)")

    # Check that memory increase is reasonable (< 50MB for this test)
    assert memory_increase < 50, f"Memory increase too large: {memory_increase:.1f} MB"
    print("✓ Attention Trainer memory management test passed!")
    return True


def test_causal_selector_memory_management():
    """Test memory management in causal feature selector."""
    print("Testing Causal Selector Memory Management...")

    # Create memory manager
    memory_manager = MemoryManager(memory_logging_enabled=True)

    # Create selector
    selector = CausalFeatureSelector(
        treatment_threshold=0.05, min_samples=50, memory_manager=memory_manager
    )

    initial_memory = get_memory_usage()
    print(f"Initial memory: {initial_memory:.1f} MB")

    # Create test data
    np.random.seed(42)
    n_samples = 1000

    data = {
        "feature_a": np.random.randn(n_samples),
        "feature_b": np.random.randn(n_samples),
        "feature_c": np.random.randn(n_samples),
        "feature_d": np.random.randn(n_samples),
        "feature_e": np.random.randn(n_samples),
        "price": np.random.randn(n_samples),
        "volume": np.random.randn(n_samples),
        "reward": np.random.randn(n_samples),
    }

    df = pd.DataFrame(data)
    features = ["feature_a", "feature_b", "feature_c", "feature_d", "feature_e"]

    # Test causal feature selection
    selected, results = selector.select_features_causal(df, features, "reward")
    print(f"Selected {len(selected)} features")

    mid_memory = get_memory_usage()
    memory_increase_mid = mid_memory - initial_memory
    print(
        f"After selection: {mid_memory:.1f} MB (increase: {memory_increase_mid:.1f} MB)"
    )

    # Test model update
    new_data = df.iloc[:500]  # DataFrameとして取得
    selector.update_causal_model(new_data, "reward")

    final_memory = get_memory_usage()
    memory_increase_final = final_memory - initial_memory
    print(
        f"After update: {final_memory:.1f} MB (total increase: {memory_increase_final:.1f} MB)"
    )

    # Check memory increase is reasonable
    assert (
        memory_increase_final < 100
    ), f"Memory increase too large: {memory_increase_final:.1f} MB"
    print("✓ Causal Selector memory management test passed!")
    return True


def test_memory_cleanup():
    """Test that objects are properly cleaned up."""
    print("Testing Memory Cleanup...")

    initial_memory = get_memory_usage()
    print(f"Initial memory: {initial_memory:.1f} MB")

    # Create and use objects
    memory_manager = MemoryManager()
    trainer = AttentionTrainer(n_features=10, memory_manager=memory_manager)

    # Add some data
    for i in range(100):
        trainer.add_training_sample(np.random.randn(10), np.random.randn(), "trending")

    created_memory = get_memory_usage()
    print(
        f"After creating objects: {created_memory:.1f} MB (increase: {created_memory - initial_memory:.1f} MB)"
    )

    # Delete objects
    del trainer
    del memory_manager
    gc.collect()

    cleaned_memory = get_memory_usage()
    cleanup_improvement = created_memory - cleaned_memory
    print(
        f"After cleanup: {cleaned_memory:.1f} MB (cleanup: {cleanup_improvement:.1f} MB)"
    )

    # Memory should decrease after cleanup
    assert (
        cleanup_improvement > 0
    ), f"No memory cleanup detected: {cleanup_improvement:.1f} MB"
    print("✓ Memory cleanup test passed!")
    return True


if __name__ == "__main__":
    try:
        test_attention_trainer_memory_management()
        print()
        test_causal_selector_memory_management()
        print()
        test_memory_cleanup()
        print("\n✓ All memory leak prevention tests passed!")
    except Exception as e:
        print(f"✗ Test failed: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)
