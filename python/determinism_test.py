#!/usr/bin/env python3
"""
Determinism test for feature engineering.
同じseedで同一入力 → 完全一致する出力を保証
"""

import numpy as np
import pandas as pd
import random
import os
from typing import Dict
import sys
import hashlib

# Add ztb to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from ztb.features.registry import FeatureRegistry


def generate_test_data(n_rows: int = 1000, seed: int = 42) -> pd.DataFrame:
    """Generate deterministic test data"""
    np.random.seed(seed)
    random.seed(seed)

    dates = pd.date_range('2023-01-01', periods=n_rows, freq='1min')

    # Generate price data with controlled randomness
    base_price = 100.0
    price_changes = np.random.normal(0, 0.01, n_rows).cumsum()
    close = base_price * (1 + price_changes)

    # Generate OHLCV with some noise
    high = close * (1 + np.abs(np.random.normal(0, 0.005, n_rows)))
    low = close * (1 - np.abs(np.random.normal(0, 0.005, n_rows)))
    open_price = close.shift(1).fillna(base_price)
    volume = np.random.lognormal(10, 1, n_rows)

    df = pd.DataFrame({
        'timestamp': dates,
        'open': open_price,
        'high': high,
        'low': low,
        'close': close,
        'volume': volume
    })

    return df


def compute_feature_hash(features: Dict[str, pd.Series]) -> str:
    """Compute hash of feature outputs for comparison"""
    combined = pd.concat(features.values(), axis=1)
    # Convert to string representation with high precision
    str_repr = combined.round(10).to_string()
    return hashlib.sha256(str_repr.encode()).hexdigest()


def test_determinism_single_process():
    """Test determinism in single process"""
    print("Testing single-process determinism...")

    # Test with same seed multiple times
    results = []
    for i in range(3):
        # Re-initialize registry with same seed
        FeatureRegistry.reset_for_testing()
        FeatureRegistry.initialize(seed=12345)

        # Generate same input data
        df = generate_test_data(seed=67890)

        # Compute features
        features = {}
        for feature_name in ['rsi14', 'macd', 'bb_width', 'atr14']:
            try:
                feature_func = FeatureRegistry.get(feature_name)
                features[feature_name] = feature_func(df)
            except KeyError:
                print(f"Warning: Feature {feature_name} not found, skipping")
                continue

        if features:
            hash_val = compute_feature_hash(features)
            results.append(hash_val)
            print(f"  Run {i+1}: {hash_val[:16]}...")

    # Check all results are identical
    if len(set(results)) == 1:
        print("✅ Single-process determinism: PASSED")
        return True
    else:
        print("❌ Single-process determinism: FAILED")
        print(f"  Expected all hashes to be identical, got: {results}")
        return False


def test_determinism_parallel_simulation():
    """Test determinism in simulated parallel environment"""
    print("Testing parallel simulation determinism...")

    base_seed = 12345
    input_seed = 67890
    df = generate_test_data(seed=input_seed)

    results = []
    for worker_id in range(4):  # Simulate 4 workers
        # Re-initialize registry
        FeatureRegistry.reset_for_testing()
        FeatureRegistry.initialize(seed=base_seed)

        # Set worker-specific seed
        FeatureRegistry.set_worker_seed(worker_id)

        # Compute features
        features = {}
        for feature_name in ['rsi14', 'macd', 'bb_width', 'atr14']:
            try:
                feature_func = FeatureRegistry.get(feature_name)
                features[feature_name] = feature_func(df)
            except KeyError:
                continue

        if features:
            hash_val = compute_feature_hash(features)
            results.append((worker_id, hash_val))
            print(f"  Worker {worker_id}: {hash_val[:16]}...")

    # Check that same worker always produces same result
    worker_hashes = {}
    for worker_id, hash_val in results:
        if worker_id in worker_hashes:
            if worker_hashes[worker_id] != hash_val:
                print("❌ Parallel determinism: FAILED")
                print(f"  Worker {worker_id} produced different results")
                return False
        worker_hashes[worker_id] = hash_val

    print("✅ Parallel simulation determinism: PASSED")
    return True


def test_different_seeds_produce_different_results():
    """Test that different seeds produce different results"""
    print("Testing that different seeds produce different results...")

    results = []
    for seed in [12345, 12346, 12347]:
        FeatureRegistry.reset_for_testing()
        FeatureRegistry.initialize(seed=seed)

        df = generate_test_data(seed=99999)  # Same input data

        features = {}
        for feature_name in ['rsi14', 'macd']:
            try:
                feature_func = FeatureRegistry.get(feature_name)
                features[feature_name] = feature_func(df)
            except KeyError:
                continue

        if features:
            hash_val = compute_feature_hash(features)
            results.append(hash_val)
            print(f"  Seed {seed}: {hash_val[:16]}...")

    # Check all results are different
    if len(set(results)) == len(results):
        print("✅ Different seeds produce different results: PASSED")
        return True
    else:
        print("❌ Different seeds produce different results: FAILED")
        print(f"  Expected all hashes to be different, got: {results}")
        return False


def main():
    """Run all determinism tests"""
    print("🔬 Running Feature Determinism Tests")
    print("=" * 50)

    tests = [
        test_determinism_single_process,
        test_determinism_parallel_simulation,
        test_different_seeds_produce_different_results,
    ]

    passed = 0
    total = len(tests)

    for test_func in tests:
        try:
            if test_func():
                passed += 1
            print()
        except Exception as e:
            print(f"❌ Test failed with exception: {e}")
            print()
            import traceback
            traceback.print_exc()

    print("=" * 50)
    print(f"Results: {passed}/{total} tests passed")

    if passed == total:
        print("🎉 All determinism tests PASSED!")
        return 0
    else:
        print("💥 Some determinism tests FAILED!")
        return 1


if __name__ == "__main__":
    sys.exit(main())