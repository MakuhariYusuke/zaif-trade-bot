#!/usr/bin/env python3
"""
Clean BTC JPY data by removing NaN values and ensuring data integrity.
"""

import pandas as pd
import numpy as np
from pathlib import Path

def clean_btc_data(input_path: str, output_path: str):
    """Clean BTC data by removing NaN values."""
    print(f"Loading data from {input_path}...")

    # Load data
    df = pd.read_csv(input_path)
    print(f"Original data shape: {df.shape}")

    # Check for NaN values
    nan_counts = df.isna().sum()
    print(f"NaN counts per column:\n{nan_counts[nan_counts > 0]}")

    # Remove rows with NaN in critical columns
    critical_columns = ['open', 'high', 'low', 'close', 'volume']
    before_clean = len(df)

    # Drop rows where critical columns are NaN
    df_clean = df.dropna(subset=critical_columns).copy()
    after_clean = len(df_clean)

    print(f"Removed {before_clean - after_clean} rows with NaN in critical columns")
    print(f"Clean data shape: {df_clean.shape}")

    # Forward fill any remaining NaN values in other columns
    df_clean = df_clean.fillna(method='ffill')

    # Ensure we have enough data
    if len(df_clean) < 1000:
        raise ValueError(f"Insufficient data after cleaning: {len(df_clean)} rows")

    # Save cleaned data
    df_clean.to_csv(output_path, index=False)
    print(f"Saved cleaned data to {output_path}")

    # Final validation
    final_nan = df_clean.isna().sum().sum()
    print(f"Final NaN count: {final_nan}")

    return df_clean

if __name__ == "__main__":
    input_file = "data/btc_jpy_yahoo_real_20251021_featured_corrected.csv"
    output_file = "data/btc_jpy_yahoo_real_20251021_featured_corrected_clean.csv"

    clean_btc_data(input_file, output_file)