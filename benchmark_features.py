#!/usr/bin/env python3
"""
Benchmark script for feature computation performance.
特徴量計算のパフォーマンスベンチマーク
"""

import time
import pandas as pd
import numpy as np
from pathlib import Path
import sys

# Add ztb to path
sys.path.insert(0, str(Path(__file__).parent))

from ztb.features import FeatureRegistry
from ztb.utils.data_generation import generate_synthetic_market_data


def benchmark_features():
    """Benchmark feature computation performance"""
    print("🚀 Feature Computation Performance Benchmark")
    print("=" * 50)

    # Generate test data
    print("Generating test data...")
    df = generate_synthetic_market_data(n_samples=10000, seed=42)
    print(f"Generated {len(df)} samples")

    # Features to test
    features = ['RSI', 'ROC', 'OBV', 'ZScore']

    results = []

    for feature_name in features:
        if feature_name not in FeatureRegistry.list():
            print(f"⚠️  Feature {feature_name} not found, skipping")
            continue

        print(f"\nTesting {feature_name}...")

        feature_func = FeatureRegistry.get(feature_name)

        # Warm up
        _ = feature_func(df)

        # Benchmark
        times = []
        for i in range(10):
            start_time = time.time()
            result = feature_func(df)
            end_time = time.time()
            times.append(end_time - start_time)

        avg_time = np.mean(times)
        std_time = np.std(times)
        min_time = np.min(times)
        max_time = np.max(times)

        print(f"  Avg: {avg_time:.4f}s, Std: {std_time:.4f}s, Min: {min_time:.4f}s, Max: {max_time:.4f}s")
        results.append({
            'feature': feature_name,
            'avg_time': avg_time,
            'std_time': std_time,
            'min_time': min_time,
            'max_time': max_time,
            'samples_per_sec': len(df) / avg_time
        })

    # Print summary
    print("\n📊 Performance Summary:")
    print("-" * 70)
    print("<12")
    print("-" * 70)

    for result in results:
        print(f"{result['feature']:<12} {result['avg_time']:.4f}s    {result['samples_per_sec']:>8.0f} samples/sec")
    print("\n✅ Benchmark completed!")


if __name__ == "__main__":
    benchmark_features()