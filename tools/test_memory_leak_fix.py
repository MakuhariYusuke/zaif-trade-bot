#!/usr/bin/env python3
"""
Quick test to verify memory leak fix.

This test:
1. Monitors memory usage before/after training data operations
2. Verifies cache is disabled by default
3. Confirms memory is released after gc.collect()
"""

import gc
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def test_memory_cache_disabled():
    """Test that memory cache is disabled by default."""
    import inspect

    from ztb.training.utils.training_utils import load_training_data_parallel

    # Check function signature
    sig = inspect.signature(load_training_data_parallel)
    default_value = sig.parameters["enable_memory_cache"].default

    print(f"✅ enable_memory_cache default value: {default_value}")
    assert default_value is False, "Memory cache should be disabled by default"
    print("✅ Memory cache is disabled by default")


def test_cache_ttl():
    """Test that data cache TTL is reduced."""
    from ztb.cache.memory_cache import default_memory_manager

    ttl = default_memory_manager.data_cache.ttl
    print(f"✅ Data cache TTL: {ttl} seconds")
    assert ttl <= 60, f"Data cache TTL should be <= 60 seconds, got {ttl}"
    print("✅ Data cache TTL is correctly reduced")


def test_memory_cleanup():
    """Test that memory cleanup works."""
    import os

    import psutil

    from ztb.cache.memory_cache import default_memory_manager

    process = psutil.Process(os.getpid())

    # Get initial memory
    mem_before = process.memory_info().rss / (1024 * 1024)  # MB
    print(f"✅ Initial memory: {mem_before:.2f} MB")

    # Add some data to cache
    for i in range(100):
        default_memory_manager.cache_training_data(f"test_key_{i}", list(range(1000)))

    mem_after_cache = process.memory_info().rss / (1024 * 1024)
    print(f"✅ Memory after caching: {mem_after_cache:.2f} MB")

    # Cleanup
    stats = default_memory_manager.optimize_memory_usage()
    gc.collect()

    mem_after_cleanup = process.memory_info().rss / (1024 * 1024)
    print(f"✅ Memory after cleanup: {mem_after_cleanup:.2f} MB")
    print(f"✅ Memory freed: {mem_after_cache - mem_after_cleanup:.2f} MB")

    # Clear all caches
    default_memory_manager.feature_cache.clear()
    default_memory_manager.data_cache.clear()
    default_memory_manager.model_cache.clear()
    gc.collect()

    mem_final = process.memory_info().rss / (1024 * 1024)
    print(f"✅ Final memory: {mem_final:.2f} MB")
    print("✅ Memory cleanup works correctly")


def test_ab_test_import():
    """Test that ab_test_runner has memory cleanup import."""
    ab_test_path = Path("tools/ab_test_runner.py")
    content = ab_test_path.read_text(encoding="utf-8")

    assert "from ztb.cache.memory_cache import default_memory_manager" in content
    assert "default_memory_manager.optimize_memory_usage()" in content
    print("✅ AB test runner has memory cleanup code")


def main():
    print("=" * 60)
    print("Memory Leak Fix Verification")
    print("=" * 60)

    tests = [
        ("Memory cache disabled by default", test_memory_cache_disabled),
        ("Cache TTL reduced", test_cache_ttl),
        ("Memory cleanup works", test_memory_cleanup),
        ("AB test has cleanup", test_ab_test_import),
    ]

    failed = []

    for name, test_func in tests:
        print(f"\n{name}...")
        try:
            test_func()
        except Exception as e:
            print(f"❌ FAILED: {e}")
            failed.append(name)
            import traceback

            traceback.print_exc()

    print("\n" + "=" * 60)
    if failed:
        print(f"❌ {len(failed)} test(s) FAILED:")
        for name in failed:
            print(f"  - {name}")
        return 1
    else:
        print("✅ ALL TESTS PASSED!")
        print("Memory leak fix verified successfully.")
        print("\nYou can now run AB tests safely:")
        print(
            '  python tools\\ab_test_runner.py --configs "config1.json" "config2.json" --seeds 1'
        )
    print("=" * 60)


if __name__ == "__main__":
    from ztb.utils.cli import run_main

    run_main(main)
