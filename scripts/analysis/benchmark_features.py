#!/usr/bin/env python3
"""
Benchmark script for feature computation performance.
特徴量計算のパフォーマンスベンチマーク
"""

import sys
import time
from pathlib import Path

import numpy as np
import psutil

# Add ztb to path
sys.path.insert(0, str(Path(__file__).parent))

from ztb.features import FeatureRegistry
from ztb.features.feature_engine import compute_features_batch
from ztb.utils.data.data_generation import generate_synthetic_market_data


def benchmark_features():
    """Benchmark feature computation performance"""
    print("🚀 Feature Computation Performance Benchmark")
    print("=" * 50)

    # Generate test data
    print("Generating test data...")
    df = generate_synthetic_market_data(n_samples=10000, seed=42)
    print(f"Generated {len(df)} samples")

    FeatureRegistry.initialize()

    # Import all feature modules to register them
    import ztb.features.momentum
    import ztb.features.scalping
    import ztb.features.time as time_features
    import ztb.features.trend
    import ztb.features.volatility
    import ztb.features.volume

    feature_names = FeatureRegistry.list()
    print(f"Testing {len(feature_names)} features...")

    # Measure initial memory
    process = psutil.Process()
    initial_memory_mb = process.memory_info().rss / 1024 / 1024
    print(f"Initial memory: {initial_memory_mb:.1f} MB")

    # Benchmark full feature computation
    times = []
    memory_peaks = []
    feature_timings = {}

    for i in range(3):  # Run 3 times for averaging
        print(f"\nRun {i+1}/3...")
        start_time = time.time()

        # Compute all features with timing
        features_df, timing_info = compute_features_batch(
            df,
            feature_names,
            verbose=False,
            enable_chunking=True,
            chunk_size=2000,
            return_timing=True,
        )

        end_time = time.time()
        elapsed = end_time - start_time
        times.append(elapsed)

        # Measure memory after computation
        peak_memory_mb = process.memory_info().rss / 1024 / 1024
        memory_peaks.append(peak_memory_mb)

        print(
            f"  Time: {elapsed:.2f}s, Memory: {peak_memory_mb:.1f} MB, Features: {features_df.shape[1]}"
        )

        # Accumulate feature timings
        if timing_info:
            for feature_name, feature_time in timing_info.items():
                if feature_name not in feature_timings:
                    feature_timings[feature_name] = []
                feature_timings[feature_name].append(feature_time)

        # Clean up
        del features_df

    avg_time = np.mean(times)
    std_time = np.std(times)
    avg_memory = np.mean(memory_peaks)
    max_memory = np.max(memory_peaks)

    print("\n📊 Performance Summary:")
    print("-" * 70)
    print(f"Average time: {avg_time:.2f}s ± {std_time:.2f}s")
    print(f"Average memory: {avg_memory:.1f} MB (peak: {max_memory:.1f} MB)")
    print(f"Features computed: {len(feature_names)}")
    print(f"Samples/sec: {len(df) * len(feature_names) / avg_time:,.0f}")
    print("-" * 70)

    # Show top 10 slowest features
    if feature_timings:
        print("\n🐌 Top 10 Slowest Features:")
        print("-" * 40)
        avg_feature_times = {
            name: np.mean(feature_times_list)
            for name, feature_times_list in feature_timings.items()
        }
        sorted_features = sorted(
            avg_feature_times.items(), key=lambda x: x[1], reverse=True
        )
        for i, (feature_name, avg_time) in enumerate(sorted_features[:10]):
            print(f"{i+1:2d}. {feature_name:<30} {avg_time:.4f}s")

    print("\n✅ Benchmark completed!")


if __name__ == "__main__":
    benchmark_features()
