#!/usr/bin/env python3

"""
Feature computation engine with stability and efficiency improvements.
特徴量計算エンジン - 安定性と効率性の改善
"""

import sys
import time
import psutil
import pandas as pd
from pathlib import Path
from typing import List, Optional

# Add parent directory to sys.path to ensure ztb modules can be imported when running as a script
sys.path.insert(0, str(Path(__file__).parent.parent))

from ztb.features import FeatureRegistry
from ztb.features.utils.rolling import optimize_dataframe_dtypes, generate_intermediate_report
from ztb.utils.data.data_generation import generate_synthetic_market_data


def compute_features_batch(df: pd.DataFrame, feature_names: Optional[List[str]] = None,
                          report_interval: int = 10000, verbose: bool = True) -> pd.DataFrame:
    """
    Compute features in batch with stability and efficiency improvements

    Args:
        df: Input DataFrame with OHLCV data
        feature_names: List of feature names to compute (None for all)
        report_interval: Steps between intermediate reports

    Returns:
        DataFrame with computed features
    """
    # Initialize registry if not already done
    FeatureRegistry.initialize()

    if feature_names is None:
        feature_names = FeatureRegistry.list()

    if verbose:
        print(f"Computing {len(feature_names)} features...")

    results = {}
    feature_times = {}
    nan_rates = {}
    step_count = 0

    for feature_name in feature_names:
        start_time = time.time()

        try:
            feature_func = FeatureRegistry.get(feature_name)
            feature_series = feature_func(df)

            # Store result
            results[feature_name] = feature_series

            # Calculate metrics
            computation_time = time.time() - start_time
            feature_times[feature_name] = computation_time
            nan_rates[feature_name] = feature_series.isna().mean()

            step_count += 1

            # Generate intermediate report
            if step_count % report_interval == 0:
                memory_usage = psutil.Process().memory_info().rss / 1024 / 1024  # MB
                generate_intermediate_report(step_count, feature_times, memory_usage, nan_rates)
                print(f"Generated intermediate report at step {step_count}")

        except Exception as e:
            print(f"Error computing {feature_name}: {e}")
            feature_times[feature_name] = 0.0
            nan_rates[feature_name] = 1.0

    # Combine all features into DataFrame
    features_df = pd.DataFrame(results)

    # Optimize dtypes
    config = FeatureRegistry.get_config()
    if config.get('optimize_dtypes', True):
        features_df = optimize_dataframe_dtypes(features_df)
        print("Optimized DataFrame dtypes")

    return features_df


def run_100k_experiment() -> pd.DataFrame:
    """Run 100k sample experiment with cache disabled"""
    print("🚀 Running 100k Feature Computation Experiment")
    print("=" * 50)

    # Initialize with cache disabled for 100k experiment
    FeatureRegistry.initialize(cache_enabled=False)

    # Generate large dataset
    print("Generating 100k sample dataset...")
    df = generate_synthetic_market_data(n_samples=100000, seed=42)
    print(f"Generated {len(df)} samples")

    # Compute features
    start_time = time.time()
    features_df = compute_features_batch(df, report_interval=10000, verbose=False)
    total_time = time.time() - start_time

    print("\n📊 Experiment Results:")
    print(f"Total computation time: {total_time:.2f}s")
    print(f"Features computed: {len(features_df.columns)}")
    print(f"Average time per sample: {total_time / len(df) * 1000:.2f}ms")
    print(f"Memory usage: {psutil.Process().memory_info().rss / 1024 / 1024:.1f} MB")

    # Verify dtypes are optimized
    dtypes_info = features_df.dtypes.value_counts()
    print(f"\nDataFrame dtypes: {dict(dtypes_info)}")

    return features_df


if __name__ == "__main__":
    # Run 100k experiment
    features_df = run_100k_experiment()

    # Save sample of results
    sample_output = features_df.head(100)
    sample_output.to_csv("reports/feature_computation_sample.csv")
    print("Sample results saved to reports/feature_computation_sample.csv")