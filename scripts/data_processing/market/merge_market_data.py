
import pandas as pd
import os

def merge_market_data():
    historical_path = "data/btc_jpy_1m_dataset.csv"
    latest_path = "data/btc_jpy_1m_latest.csv"
    output_path = "data/btc_jpy_1m_dataset.csv"
    backup_path = "data/btc_jpy_1m_dataset.bak"
    
    print(f"Merging {historical_path} and {latest_path}...")
    
    if not os.path.exists(historical_path):
        print(f"Error: Historical data not found at {historical_path}")
        return
        
    if not os.path.exists(latest_path):
        print(f"Error: Latest data not found at {latest_path}")
        return
        
    # Read data
    df_hist = pd.read_csv(historical_path, parse_dates=["timestamp"])
    df_latest = pd.read_csv(latest_path, parse_dates=["timestamp"])
    
    # Ensure timezone awareness compatibility
    # If one is tz-aware and other is not, convert to UTC
    if df_hist["timestamp"].dt.tz is None:
        df_hist["timestamp"] = df_hist["timestamp"].dt.tz_localize("UTC")
    else:
        df_hist["timestamp"] = df_hist["timestamp"].dt.tz_convert("UTC")
        
    if df_latest["timestamp"].dt.tz is None:
        df_latest["timestamp"] = df_latest["timestamp"].dt.tz_localize("UTC")
    else:
        df_latest["timestamp"] = df_latest["timestamp"].dt.tz_convert("UTC")

    print(f"Historical data: {len(df_hist)} rows ({df_hist['timestamp'].min()} to {df_hist['timestamp'].max()})")
    print(f"Latest data: {len(df_latest)} rows ({df_latest['timestamp'].min()} to {df_latest['timestamp'].max()})")
    
    # Align columns (drop extra columns like adj_close if not in latest)
    common_cols = [c for c in df_hist.columns if c in df_latest.columns]
    # Ensure required columns are present
    required = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
    for c in required:
        if c not in common_cols:
            print(f"Warning: Column {c} missing from intersection. Merging might be incomplete.")
    
    df_hist = df_hist[common_cols]
    df_latest = df_latest[common_cols]
    
    # Concatenate
    df_merged = pd.concat([df_hist, df_latest])
    
    # Drop duplicates (keep last/latest version if overlap)
    df_merged = df_merged.drop_duplicates(subset=["timestamp"], keep="last")
    
    # Sort
    df_merged = df_merged.sort_values("timestamp")
    
    # Reset index
    df_merged = df_merged.reset_index(drop=True)
    
    print(f"Merged data: {len(df_merged)} rows ({df_merged['timestamp'].min()} to {df_merged['timestamp'].max()})")
    
    # Backup existing
    if os.path.exists(output_path):
        import shutil
        shutil.copy2(output_path, backup_path)
        print(f"Backed up existing data to {backup_path}")
    
    # Save
    df_merged.to_csv(output_path, index=False)
    print(f"Saved merged data to {output_path}")

if __name__ == "__main__":
    merge_market_data()
