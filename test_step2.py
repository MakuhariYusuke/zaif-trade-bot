#!/usr/bin/env python3
"""
Test script for Step 2: Feature optimization and Registry extension
"""

from ztb.features.registry import FeatureRegistry
from ztb.utils.data_generation import generate_synthetic_market_data
import time

def test_feature_optimization():
    print('=== Step 2 Dry-run: Feature Optimization and Registry Extension ===')

    # Generate test data
    df = generate_synthetic_market_data(n_samples=10000)
    print(f'Generated {len(df)} samples')

    features_to_test = ['RSI', 'ZScore', 'MACD', 'BB_Width']

    # Test with cache enabled, parallel enabled
    print('\n--- Testing with cache_enabled=True, parallel_enabled=True ---')
    FeatureRegistry.initialize(seed=42, cache_enabled=True, parallel_enabled=True)
    print(f'Cache enabled: {FeatureRegistry.is_cache_enabled()}')
    print(f'Parallel enabled: {FeatureRegistry.is_parallel_enabled()}')

    results_cache_on = {}
    for feature in features_to_test:
        start = time.time()
        result = FeatureRegistry.get(feature)(df)
        end = time.time()
        elapsed = end - start
        nan_rate = result.isna().sum() / len(result) * 100
        results_cache_on[feature] = {
            'time': elapsed,
            'nan_rate': nan_rate,
            'range': (result.min(), result.max())
        }
        print(f'{feature}: {elapsed:.4f}s, NaN: {nan_rate:.2f}%, Range: {result.min():.4f} to {result.max():.4f}')

    # Test with cache disabled, parallel disabled (need to reset registry)
    print('\n--- Testing with cache_enabled=False, parallel_enabled=False ---')
    # Force re-initialize by setting _initialized to False
    FeatureRegistry._initialized = False
    FeatureRegistry.initialize(seed=42, cache_enabled=False, parallel_enabled=False)
    print(f'Cache enabled: {FeatureRegistry.is_cache_enabled()}')
    print(f'Parallel enabled: {FeatureRegistry.is_parallel_enabled()}')

    results_cache_off = {}
    for feature in features_to_test:
        start = time.time()
        result = FeatureRegistry.get(feature)(df)
        end = time.time()
        elapsed = end - start
        nan_rate = result.isna().sum() / len(result) * 100
        results_cache_off[feature] = {
            'time': elapsed,
            'nan_rate': nan_rate,
            'range': (result.min(), result.max())
        }
        print(f'{feature}: {elapsed:.4f}s, NaN: {nan_rate:.2f}%, Range: {result.min():.4f} to {result.max():.4f}')

    # Compare results
    print('\n--- Performance Comparison ---')
    for feature in features_to_test:
        time_diff = results_cache_off[feature]['time'] - results_cache_on[feature]['time']
        print(f'{feature}: Cache speedup: {time_diff:.4f}s')

    print('=== Step 2 Test Completed ===')

if __name__ == "__main__":
    test_feature_optimization()