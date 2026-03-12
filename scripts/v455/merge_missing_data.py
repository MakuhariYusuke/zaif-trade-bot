
import sys
import os
from pathlib import Path
import pandas as pd
import numpy as np

# Add project root to path
try:
    from ztb.utils.path_utils import get_project_root
    project_root = get_project_root()
except ImportError:
    project_root = Path(__file__).resolve().parent.parent.parent
    sys.path.insert(0, str(project_root))

def merge_data():
    target_path = project_root / "data" / "btc_jpy_1m_v454.csv"
    source_path = project_root / "data" / "btc_jpy_1m_dataset.csv"
    
    print(f"Target: {target_path}")
    print(f"Source: {source_path}")
    
    if not target_path.exists() or not source_path.exists():
        print("Error: One of the files does not exist.")
        return

    # Load Target
    print("Loading target...")
    df_target = pd.read_csv(target_path, index_col=0, parse_dates=True)
    # Ensure UTC
    if df_target.index.tz is None:
        df_target.index = df_target.index.tz_localize("UTC")
    else:
        df_target.index = df_target.index.tz_convert("UTC")
        
    print(f"Target rows: {len(df_target)}")
    print(f"Target range: {df_target.index.min()} to {df_target.index.max()}")

    # Load Source
    print("Loading source...")
    df_source = pd.read_csv(source_path, index_col=0, parse_dates=True)
    # Ensure UTC
    if df_source.index.tz is None:
        df_source.index = df_source.index.tz_localize("UTC")
    else:
        df_source.index = df_source.index.tz_convert("UTC")
        
    print(f"Source rows: {len(df_source)}")
    print(f"Source range: {df_source.index.min()} to {df_source.index.max()}")
    
    # Find missing timestamps
    missing_indices = df_source.index.difference(df_target.index)
    
    if missing_indices.empty:
        print("No missing data found in source.")
        return
        
    print(f"Found {len(missing_indices)} missing rows in source.")
    
    # Extract missing rows
    df_missing = df_source.loc[missing_indices]
    
    # Align columns
    # Target has many columns, Source only has OHLCV.
    # We want to keep Target's columns.
    # Reindex df_missing to match df_target columns, filling with NaN
    df_missing_aligned = df_missing.reindex(columns=df_target.columns)
    
    # Fill OHLCV from source (since reindex might have nulled them if names didn't match exactly, but they should match)
    # Let's double check column names case
    # Target columns are likely lowercase (based on previous checks)
    # Source columns: timestamp,close,high,low,open,volume (from Get-Content)
    # Let's ensure lowercase
    df_missing.columns = [c.lower() for c in df_missing.columns]
    
    # Map source columns to target columns
    for col in ["open", "high", "low", "close", "volume"]:
        if col in df_missing.columns and col in df_target.columns:
            df_missing_aligned[col] = df_missing[col]
            
    # Concatenate
    df_merged = pd.concat([df_target, df_missing_aligned])
    
    # Sort
    df_merged = df_merged.sort_index()
    
    # Remove duplicates (just in case)
    df_merged = df_merged[~df_merged.index.duplicated(keep='last')]
    
    print(f"Merged rows: {len(df_merged)}")
    print(f"Merged range: {df_merged.index.min()} to {df_merged.index.max()}")
    
    # Save
    print("Saving merged dataset...")
    df_merged.to_csv(target_path)
    print("Done.")

if __name__ == "__main__":
    merge_data()
