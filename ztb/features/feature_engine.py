#!/usr/bin/env python3

"""
Feature computation engine with stability and efficiency improvements.
特徴量計算エンジン - 安定性と効率性の改善
"""

import gc
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd
import psutil

# Add parent directory to sys.path to ensure ztb modules can be imported when running as a script
sys.path.insert(0, str(Path(__file__).parent.parent))

from ztb.features import FeatureRegistry
from ztb.utils.data.data_generation import generate_synthetic_market_data
from ztb.utils.errors import safe_operation

# Optimize GC for better memory management - more aggressive
gc.set_threshold(100, 5, 5)  # Even more aggressive garbage collection


def compute_features_batch(
    df: pd.DataFrame,
    feature_names: Optional[List[str]] = None,
    report_interval: int = 10000,
    verbose: bool = True,
    enable_chunking: Optional[bool] = None,
    chunk_size: Optional[int] = None,
    return_timing: bool = False,
) -> pd.DataFrame | Tuple[pd.DataFrame, Dict[str, float]]:
    """Wrapper that delegates to the FeatureRegistry batch computation."""
    return FeatureRegistry.compute_features_batch(
        df,
        feature_names=feature_names,
        report_interval=report_interval,
        verbose=verbose,
        enable_chunking=enable_chunking,
        chunk_size=chunk_size,
        return_timing=return_timing,
    )


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

    # Compute features with safe operation
    start_time = time.time()
    features_df = safe_operation(
        compute_features_batch,
        fallback=pd.DataFrame(),
        operation_name="100k_feature_computation",
        df=df,
        verbose=True,
        return_timing=False
    )
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
