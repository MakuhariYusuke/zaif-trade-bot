#!/usr/bin/env python3
"""
Performance profiling tools for identifying bottlenecks in feature computation.
パフォーマンスプロファイリングツール - 特徴量計算のボトルネック特定
"""

import cProfile
import io
import pstats
import time
from contextlib import contextmanager
from typing import Dict, List, Optional, Tuple

import pandas as pd
import psutil
from ztb.features import FeatureRegistry


class PerformanceProfiler:
    """Performance profiling utilities for feature computation"""

    def __init__(self):
        self.process = psutil.Process()
        self._profile_stats: Optional[pstats.Stats] = None

    @contextmanager
    def profile_context(self, name: str = "operation"):
        """Context manager for profiling code blocks"""
        profiler = cProfile.Profile()
        profiler.enable()

        start_time = time.time()
        start_memory = self.process.memory_info().rss / 1024 / 1024  # MB

        try:
            yield
        finally:
            end_time = time.time()
            end_memory = self.process.memory_info().rss / 1024 / 1024  # MB

            profiler.disable()

            elapsed = end_time - start_time
            memory_delta = end_memory - start_memory

            print(f"[{name}] Time: {elapsed:.3f}s, Memory: {memory_delta:+.2f}MB")

            # Store stats for later analysis
            s = io.StringIO()
            ps = pstats.Stats(profiler, stream=s).sort_stats('cumulative')
            ps.print_stats(20)  # Top 20 functions
            self._profile_stats = ps

    def get_profile_summary(self) -> str:
        """Get profiling summary"""
        if self._profile_stats is None:
            return "No profiling data available"

        s = io.StringIO()
        self._profile_stats.stream = s
        self._profile_stats.print_stats(10)
        return s.getvalue()

    def benchmark_features(
        self,
        df: pd.DataFrame,
        feature_names: Optional[List[str]] = None,
        iterations: int = 3
    ) -> Dict[str, Dict[str, float]]:
        """
        Benchmark individual feature computation performance

        Returns:
            Dict mapping feature names to performance metrics
        """
        if feature_names is None:
            feature_names = FeatureRegistry.list()

        results = {}

        for feature_name in feature_names:
            if feature_name not in FeatureRegistry.list():
                continue

            times = []
            memories = []

            for i in range(iterations):
                start_memory = self.process.memory_info().rss / 1024 / 1024

                start_time = time.time()
                feature_func = FeatureRegistry.get(feature_name)
                result = feature_func(df)
                end_time = time.time()

                end_memory = self.process.memory_info().rss / 1024 / 1024

                times.append(end_time - start_time)
                memories.append(end_memory - start_memory)

            # Calculate statistics
            avg_time = sum(times) / len(times)
            min_time = min(times)
            max_time = max(times)
            avg_memory = sum(memories) / len(memories)

            results[feature_name] = {
                'avg_time_ms': avg_time * 1000,
                'min_time_ms': min_time * 1000,
                'max_time_ms': max_time * 1000,
                'avg_memory_mb': avg_memory,
                'per_sample_us': (avg_time / len(df)) * 1_000_000,  # microseconds per sample
                'output_shape': result.shape if hasattr(result, 'shape') else len(result),
                'output_dtype': str(result.dtype) if hasattr(result, 'dtype') else 'unknown'
            }

        return results

    def identify_bottlenecks(
        self,
        benchmark_results: Dict[str, Dict[str, float]],
        time_threshold_ms: float = 1.0,
        memory_threshold_mb: float = 10.0
    ) -> Dict[str, List[str]]:
        """
        Identify performance bottlenecks based on benchmarks

        Returns:
            Dict with 'slow_features' and 'memory_hungry_features' lists
        """
        slow_features = []
        memory_hungry_features = []

        for feature_name, metrics in benchmark_results.items():
            if metrics['avg_time_ms'] > time_threshold_ms:
                slow_features.append(f"{feature_name} ({metrics['avg_time_ms']:.2f}ms)")

            if metrics['avg_memory_mb'] > memory_threshold_mb:
                memory_hungry_features.append(f"{feature_name} ({metrics['avg_memory_mb']:.2f}MB)")

        return {
            'slow_features': slow_features,
            'memory_hungry_features': memory_hungry_features
        }

    def print_benchmark_report(
        self,
        benchmark_results: Dict[str, Dict[str, float]],
        top_n: int = 10
    ) -> None:
        """Print a formatted benchmark report"""
        print("=" * 80)
        print("FEATURE PERFORMANCE BENCHMARK REPORT")
        print("=" * 80)

        # Sort by average time (slowest first)
        sorted_by_time = sorted(
            benchmark_results.items(),
            key=lambda x: x[1]['avg_time_ms'],
            reverse=True
        )

        print(f"\nTOP {top_n} SLOWEST FEATURES:")
        print("-" * 60)
        for i, (name, metrics) in enumerate(sorted_by_time[:top_n]):
            print(f"{i+1:2d}. {name:<25} {metrics['avg_time_ms']:>8.2f}ms "
                  f"({metrics['per_sample_us']:>6.1f}μs/sample)")

        # Sort by memory usage
        sorted_by_memory = sorted(
            benchmark_results.items(),
            key=lambda x: x[1]['avg_memory_mb'],
            reverse=True
        )

        print(f"\nTOP {top_n} MEMORY-HUNGRIEST FEATURES:")
        print("-" * 60)
        for i, (name, metrics) in enumerate(sorted_by_memory[:top_n]):
            print(f"{i+1:2d}. {name:<25} {metrics['avg_memory_mb']:>8.2f}MB "
                  f"({metrics['output_dtype']})")

        # Summary statistics
        if benchmark_results:
            times = [m['avg_time_ms'] for m in benchmark_results.values()]
            memories = [m['avg_memory_mb'] for m in benchmark_results.values()]

            print(f"\nSUMMARY STATISTICS:")
            print("-" * 60)
            print(f"Total features tested: {len(benchmark_results)}")
            print(f"Average time per feature: {sum(times)/len(times):.2f}ms")
            print(f"Average memory per feature: {sum(memories)/len(memories):.2f}MB")
            print(f"Slowest feature: {sorted_by_time[0][0]} ({sorted_by_time[0][1]['avg_time_ms']:.2f}ms)")
            print(f"Most memory hungry: {sorted_by_memory[0][0]} ({sorted_by_memory[0][1]['avg_memory_mb']:.2f}MB)")
        else:
            print("\nSUMMARY STATISTICS:")
            print("-" * 60)
            print("No features were successfully tested")


def run_performance_analysis(df: pd.DataFrame, feature_subset: Optional[List[str]] = None) -> None:
    """
    Run complete performance analysis on feature computation

    Args:
        df: Test DataFrame with OHLCV data
        feature_subset: Optional list of features to test (None for all)
    """
    print("🚀 Starting Feature Performance Analysis")
    print("=" * 50)

    # Initialize features
    from ztb.features import trend, volatility, momentum, scalping, utils, volume
    FeatureRegistry.initialize()

    profiler = PerformanceProfiler()

    # Run benchmarks
    print("Running benchmarks...")
    benchmark_results = profiler.benchmark_features(df, feature_subset, iterations=3)

    # Print detailed report
    profiler.print_benchmark_report(benchmark_results)

    # Identify bottlenecks
    bottlenecks = profiler.identify_bottlenecks(benchmark_results)
    print("\nBOTTLENECK ANALYSIS:")
    print("-" * 60)

    if bottlenecks['slow_features']:
        print(f"⚠️  Slow features (>1ms): {len(bottlenecks['slow_features'])}")
        for feature in bottlenecks['slow_features'][:5]:  # Show top 5
            print(f"   • {feature}")
    else:
        print("✅ No slow features detected")

    if bottlenecks['memory_hungry_features']:
        print(f"⚠️  Memory hungry features (>10MB): {len(bottlenecks['memory_hungry_features'])}")
        for feature in bottlenecks['memory_hungry_features'][:5]:  # Show top 5
            print(f"   • {feature}")
    else:
        print("✅ No memory hungry features detected")

    print("\n✅ Performance analysis complete!")


if __name__ == "__main__":
    # Example usage
    import numpy as np

    # Generate test data
    np.random.seed(42)
    n_samples = 5000
    dates = pd.date_range('2023-01-01', periods=n_samples, freq='1min')
    df = pd.DataFrame({
        'close': np.random.uniform(100, 200, n_samples),
        'high': np.random.uniform(105, 210, n_samples),
        'low': np.random.uniform(95, 190, n_samples),
        'open': np.random.uniform(98, 202, n_samples),
        'volume': np.random.uniform(1000, 10000, n_samples),
    }, index=dates)

    # Add technical indicators
    df['rsi_14'] = np.random.uniform(20, 80, n_samples)
    df['macd'] = np.random.uniform(-5, 5, n_samples)
    df['macd_hist'] = np.random.uniform(-2, 2, n_samples)
    df['atr_14'] = np.random.uniform(1, 5, n_samples)

    # Initialize features
    from ztb.features import trend, volatility, momentum, scalping, utils, volume
    FeatureRegistry.initialize()

    # Run analysis on key features
    key_features = ['RegimeClustering', 'KalmanFilter', 'KalmanVelocity', 'ADX', 'RSI', 'MACD']
    run_performance_analysis(df, key_features)