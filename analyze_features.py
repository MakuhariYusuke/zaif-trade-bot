#!/usr/bin/env python3
"""
Analyze ml-dataset-enhanced.csv for feature statistics
"""

import pandas as pd
import numpy as np
from pathlib import Path

def analyze_features():
    """Analyze features in ml-dataset-enhanced.csv"""

    data_path = Path("ml-dataset-enhanced.csv")
    df = pd.read_csv(data_path)

    print(f"Dataset shape: {df.shape}")
    print(f"Columns: {len(df.columns)}")

    # Exclude non-feature columns
    exclude_cols = [
        "ts", "timestamp", "exchange", "pair", "episode_id",
        "side", "source", "close", "open", "high", "low", "volume"
    ]

    feature_cols = [col for col in df.columns if col not in exclude_cols]
    print(f"Feature columns: {len(feature_cols)}")
    print(f"Feature names: {feature_cols[:10]}...")  # First 10

    # Calculate statistics for numeric features
    numeric_features = df[feature_cols].select_dtypes(include=[np.number]).columns
    print(f"\nNumeric features: {len(numeric_features)}")

    stats = []
    zero_variance_cols = []

    for col in numeric_features:
        series = df[col].dropna()
        if len(series) == 0:
            continue

        mean_val = series.mean()
        std_val = series.std()
        min_val = series.min()
        max_val = series.max()
        n_unique = series.nunique()

        stats.append({
            'feature': col,
            'mean': mean_val,
            'std': std_val,
            'min': min_val,
            'max': max_val,
            'n_unique': n_unique,
            'n_missing': df[col].isnull().sum()
        })

        # Check for zero variance
        if std_val == 0 or std_val < 1e-10:
            zero_variance_cols.append(col)

    # Sort by std ascending to see low variance features first
    stats_df = pd.DataFrame(stats).sort_values('std')

    print("\n=== Feature Statistics (sorted by std ascending) ===")
    pd.set_option('display.max_rows', None)
    pd.set_option('display.float_format', '{:.6f}'.format)
    print(stats_df.to_string(index=False))

    print(f"\n=== Zero Variance Columns ({len(zero_variance_cols)}) ===")
    if zero_variance_cols:
        for col in zero_variance_cols:
            print(f"- {col}")
    else:
        print("No zero variance columns found.")

    # Check for NaN values
    nan_cols = df[feature_cols].isnull().sum()
    nan_cols = nan_cols[nan_cols > 0]
    print(f"\n=== Columns with NaN values ({len(nan_cols)}) ===")
    if len(nan_cols) > 0:
        for col, count in nan_cols.items():
            print(f"- {col}: {count} NaN values")
    else:
        print("No NaN values found in feature columns.")

if __name__ == "__main__":
    analyze_features()