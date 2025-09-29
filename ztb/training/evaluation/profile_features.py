#!/usr/bin/env python3
"""
Performance profiling for feature computation
特徴量計算のパフォーマンスプロファイリング
"""

import sys
from pathlib import Path

from line_profiler import LineProfiler
from memory_profiler import memory_usage

# Add ztb to path
sys.path.insert(0, str(Path(__file__).parent))

from ztb.features import FeatureRegistry
from ztb.utils.data_generation import generate_synthetic_market_data

# Global variables for profiling
df = None
rsi_func = None
zscore_func = None
obv_func = None
roc_func = None
macd_func = None
bb_upper_func = None
bb_lower_func = None
bb_width_func = None


def profile_rsi():
    return rsi_func(df)


def profile_zscore():
    return zscore_func(df)


def profile_obv():
    return obv_func(df)


def profile_roc():
    return roc_func(df)


def rsi_wrapper():
    return rsi_func(df)


def zscore_wrapper():
    return zscore_func(df)


def obv_wrapper():
    return obv_func(df)


def profile_macd():
    macd_func = FeatureRegistry.get("MACD")
    return macd_func(df)


def profile_bb_upper():
    bb_upper_func = FeatureRegistry.get("BB_Upper")
    return bb_upper_func(df)


def profile_bb_lower():
    bb_lower_func = FeatureRegistry.get("BB_Lower")
    return bb_lower_func(df)


def profile_bb_width():
    bb_width_func = FeatureRegistry.get("BB_Width")
    return bb_width_func(df)


def profile_feature_computation():
    """Profile feature computation performance"""
    global df, rsi_func, zscore_func, obv_func, roc_func

    print("Starting feature profiling...")

    # Initialize registry
    FeatureRegistry.initialize(cache_enabled=False)

    # Generate test data
    df = generate_synthetic_market_data(n_samples=5000, seed=42)
    print(f"Generated {len(df)} samples for profiling")

    # Get feature functions (local to avoid global issues)
    rsi_func = FeatureRegistry.get("RSI")
    zscore_func = FeatureRegistry.get("ZScore")
    obv_func = FeatureRegistry.get("OBV")
    roc_func = FeatureRegistry.get("ROC")

    # Line profiling
    print("\n=== RSI Profiling ===")
    profiler = LineProfiler()
    profiler.add_function(profile_rsi)
    profiler.run("profile_rsi()")
    profiler.print_stats()

    print("\n=== ZScore Profiling ===")
    profiler = LineProfiler()
    profiler.add_function(profile_zscore)
    profiler.run("profile_zscore()")
    profiler.print_stats()

    print("\n=== OBV Profiling ===")
    profiler = LineProfiler()
    profiler.add_function(profile_obv)
    profiler.run("profile_obv()")
    profiler.print_stats()

    print("\n=== ROC Profiling ===")
    profiler = LineProfiler()
    profiler.add_function(profile_roc)
    profiler.run("profile_roc()")
    profiler.print_stats()

    print("\n=== MACD Profiling ===")
    profiler = LineProfiler()
    profiler.add_function(profile_macd)
    profiler.run("profile_macd()")
    profiler.print_stats()

    print("\n=== BB Upper Profiling ===")
    profiler = LineProfiler()
    profiler.add_function(profile_bb_upper)
    profiler.run("profile_bb_upper()")
    profiler.print_stats()

    print("\n=== BB Width Profiling ===")
    profiler = LineProfiler()
    profiler.add_function(profile_bb_width)
    profiler.run("profile_bb_width()")
    profiler.print_stats()

    # Memory profiling (using tuple format for Windows compatibility)
    print("\n=== Memory Profiling ===")

    def rsi_wrapper():
        return rsi_func(df)

    def zscore_wrapper():
        return zscore_func(df)

    def obv_wrapper():
        return obv_func(df)

    def roc_wrapper():
        return roc_func(df)

    try:
        rsi_mem = memory_usage(rsi_wrapper)
        print(f"RSI memory usage: {rsi_mem}")
    except Exception as e:
        print(f"RSI memory profiling failed: {e}")

    try:
        zscore_mem = memory_usage(zscore_wrapper)
        print(f"ZScore memory usage: {zscore_mem}")
    except Exception as e:
        print(f"ZScore memory profiling failed: {e}")

    try:
        obv_mem = memory_usage(obv_wrapper)
        print(f"OBV memory usage: {obv_mem}")
    except Exception as e:
        print(f"OBV memory profiling failed: {e}")

    try:
        roc_mem = memory_usage(roc_wrapper)
        print(f"ROC memory usage: {roc_mem}")
    except Exception as e:
        print(f"ROC memory profiling failed: {e}")


if __name__ == "__main__":
    profile_feature_computation()
