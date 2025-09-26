#!/usr/bin/env python3
"""
Test script for Step 1: Binance data acquisition
"""

from ztb.data.binance_data import fetch_historical_klines, interpolate_missing_data, save_parquet_chunked, load_parquet_pattern
import tempfile
import os

def test_binance_data_acquisition():
    print('=== Step 1 Dry-run: Binance Data Acquisition ===')

    # Fetch 1 week of data
    print('Fetching 1 week of Binance BTC/USDT 1m data...')
    df = fetch_historical_klines(days=7)
    print(f'Fetched {len(df)} rows')
    print(f'Date range: {df.index.min()} to {df.index.max()}')

    # Check for missing data
    missing_count = df.isnull().sum().sum()
    print(f'Missing values before interpolation: {missing_count}')

    # Interpolate missing data
    df_interp = interpolate_missing_data(df)
    missing_after = df_interp.isnull().sum().sum()
    print(f'Missing values after interpolation: {missing_after}')
    print(f'Rows after interpolation: {len(df_interp)}')

    # Save to Parquet
    with tempfile.TemporaryDirectory() as tmpdir:
        save_path = os.path.join(tmpdir, 'binance_test')
        files = save_parquet_chunked(df_interp, save_path)
        print(f'Saved to {len(files)} files: {[os.path.basename(f) for f in files]}')

        # Load back
        df_loaded = load_parquet_pattern(save_path)
        print(f'Loaded {len(df_loaded)} rows')
        print(f'Data integrity check: {len(df_interp) == len(df_loaded)}')

        # Sample data check
        print('Sample data:')
        print(df_loaded.head(3))

    print('=== Step 1 Test Completed ===')

if __name__ == "__main__":
    test_binance_data_acquisition()