#!/usr/bin/env python3
"""
Yahoo Finance BTC/JPY Data Converter

Convert Yahoo Finance data to match training data format for SAC analysis.
"""

import argparse
import sys
from pathlib import Path

import pandas as pd

# Add project root to path
project_root = next(
    (p for p in Path(__file__).resolve().parents if (p / "pyproject.toml").exists()),
    Path(__file__).resolve().parent,
)
sys.path.insert(0, str(project_root))


def convert_yahoo_finance_data(input_file: str, output_file: str):
    """Convert Yahoo Finance data to training data format."""
    print(f"Loading data from {input_file}...")

    # Load Yahoo Finance data
    df = pd.read_csv(input_file)

    # Check if data has the expected columns
    expected_cols = ['timestamp', 'close', 'high', 'low', 'open', 'volume', 'adj_close']
    if not all(col in df.columns for col in expected_cols):
        print(f"Warning: Missing expected columns. Found: {df.columns.tolist()}")

    # Reorder columns to match training data format: timestamp, open, high, low, close, volume
    df_converted = df[['timestamp', 'open', 'high', 'low', 'close', 'volume']].copy()

    # Convert timestamp to datetime and remove timezone info
    df_converted['timestamp'] = pd.to_datetime(df_converted['timestamp'])
    if df_converted['timestamp'].dt.tz is not None:
        df_converted['timestamp'] = df_converted['timestamp'].dt.tz_localize(None)

    # Remove duplicates and sort by timestamp
    df_converted = df_converted.drop_duplicates(subset=['timestamp']).sort_values('timestamp').reset_index(drop=True)

    # Save converted data
    df_converted.to_csv(output_file, index=False)

    print("✅ データ変換完了:")
    print(f"  元データ: {len(df)} レコード")
    print(f"  変換後: {len(df_converted)} レコード")
    print(f"  保存先: {output_file}")
    print(f"  期間: {df_converted['timestamp'].min()} から {df_converted['timestamp'].max()}")


def main():
    parser = argparse.ArgumentParser(description="Convert Yahoo Finance data to training format")
    parser.add_argument("--input", required=True, help="Input Yahoo Finance CSV file")
    parser.add_argument("--output", required=True, help="Output converted CSV file")

    args = parser.parse_args()

    convert_yahoo_finance_data(args.input, args.output)


if __name__ == "__main__":
    main()
