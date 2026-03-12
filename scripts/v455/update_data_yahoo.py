
import sys
import os
from pathlib import Path
import pandas as pd
import yfinance as yf
import numpy as np

# Add project root to path
try:
    from ztb.utils.path_utils import get_project_root
    project_root = get_project_root()
except ImportError:
    project_root = Path(__file__).resolve().parent.parent.parent
    sys.path.insert(0, str(project_root))

def update_dataset():
    data_path = project_root / "data" / "btc_jpy_1m_v454.csv"
    
    print(f"Target file: {data_path}")
    
    if not data_path.exists():
        print("Error: Target file not found.")
        return

    # Load existing data
    print("Loading existing data...")
    df_existing = pd.read_csv(data_path, index_col=0, parse_dates=True)
    
    # Check timestamp index
    if not isinstance(df_existing.index, pd.DatetimeIndex):
        print("Error: Index is not DatetimeIndex. Converting...")
        df_existing.index = pd.to_datetime(df_existing.index, utc=True)
    
    # Ensure UTC
    if df_existing.index.tz is None:
        df_existing.index = df_existing.index.tz_localize("UTC")
    else:
        df_existing.index = df_existing.index.tz_convert("UTC")
        
    last_timestamp = df_existing.index.max()
    print(f"Last timestamp in dataset: {last_timestamp}")
    
    # Download new data
    ticker = "BTC-JPY"
    interval = "1m"
    period = "5d" # Yahoo 1m is usually last 7 days. 5d to be safe and cover gap.
    
    print(f"Downloading {ticker} data (Interval: {interval}, Period: {period})...")
    try:
        df_new = yf.download(ticker, interval=interval, period=period, progress=False)
    except Exception as e:
        print(f"Error downloading data: {e}")
        return

    if df_new.empty:
        print("No data downloaded.")
        return
        
    # Process new data
    # yfinance might return MultiIndex columns if multiple tickers (but here only one)
    # or just standard columns.
    if isinstance(df_new.columns, pd.MultiIndex):
        df_new.columns = [col[0] for col in df_new.columns]
        
    df_new.columns = [c.lower() for c in df_new.columns]
    
    # Ensure UTC
    if df_new.index.tz is None:
        df_new.index = df_new.index.tz_localize("UTC")
    else:
        df_new.index = df_new.index.tz_convert("UTC")
        
    # Filter new data (only keep data after last_timestamp)
    new_rows = df_new[df_new.index > last_timestamp]
    
    if new_rows.empty:
        print("No new data found after the last timestamp.")
        return
        
    print(f"Found {len(new_rows)} new rows.")
    
    # Align columns
    # The existing dataset has many columns. The new one only has OHLCV.
    # We will append new rows, leaving other columns as NaN.
    
    # Ensure basic OHLCV columns match
    required_cols = ["open", "high", "low", "close", "volume"]
    for col in required_cols:
        if col not in new_rows.columns:
            print(f"Error: Missing column {col} in downloaded data.")
            return
            
    # Concatenate
    # We only take the required columns from new_rows to avoid index conflicts if any
    # But actually we want to append to the big dataframe.
    # pd.concat will align columns and fill missing with NaN.
    
    df_updated = pd.concat([df_existing, new_rows])
    
    # Sort just in case
    df_updated = df_updated.sort_index()
    
    # Remove duplicates
    df_updated = df_updated[~df_updated.index.duplicated(keep='last')]
    
    print(f"Updated dataset shape: {df_updated.shape}")
    print(f"New last timestamp: {df_updated.index.max()}")
    
    # Save
    print("Saving updated dataset...")
    df_updated.to_csv(data_path)
    print("Done.")

if __name__ == "__main__":
    update_dataset()
