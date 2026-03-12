import sys
from pathlib import Path

import pandas as pd
import yfinance as yf

# Add project root to path
project_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(project_root))


def update_dataset():
    raw_data_path = project_root / "data" / "btc_jpy_1m_dataset.csv"

    print(f"Loading existing data from {raw_data_path}...")
    if raw_data_path.exists():
        df_existing = pd.read_csv(raw_data_path)
        if "timestamp" in df_existing.columns:
            df_existing["timestamp"] = pd.to_datetime(
                df_existing["timestamp"], utc=True
            )
        else:
            print("Error: 'timestamp' column not found in existing data.")
            return

        last_timestamp = df_existing["timestamp"].max()
        print(f"Last timestamp in dataset: {last_timestamp}")
    else:
        print("Existing dataset not found. Creating new one.")
        df_existing = pd.DataFrame(
            columns=["timestamp", "close", "high", "low", "open", "volume"]
        )
        last_timestamp = None

    # Download new data
    ticker = "BTC-JPY"
    interval = "1m"
    period = "7d"  # Max available for 1m

    print(f"Downloading {ticker} data (Interval: {interval}, Period: {period})...")
    try:
        df_new = yf.download(ticker, interval=interval, period=period)
    except Exception as e:
        print(f"Error downloading data: {e}")
        return

    if df_new.empty:
        print("No data downloaded.")
        return

    # Reset index to make timestamp a column
    df_new = df_new.reset_index()

    # Flatten MultiIndex columns if present
    if isinstance(df_new.columns, pd.MultiIndex):
        # Keep the first level (Price type) and ignore the second level (Ticker) if it exists
        df_new.columns = [
            col[0] if isinstance(col, tuple) else col for col in df_new.columns
        ]

    # Normalize column names
    df_new.columns = [c.lower() for c in df_new.columns]

    # Rename 'datetime' or 'date' to 'timestamp'
    if "datetime" in df_new.columns:
        df_new = df_new.rename(columns={"datetime": "timestamp"})
    elif "date" in df_new.columns:
        df_new = df_new.rename(columns={"date": "timestamp"})

    # Ensure timestamp is UTC
    if "timestamp" in df_new.columns:
        df_new["timestamp"] = pd.to_datetime(df_new["timestamp"], utc=True)

    # Select required columns
    required_cols = ["timestamp", "close", "high", "low", "open", "volume"]

    # Check if all required columns exist
    missing_cols = [col for col in required_cols if col not in df_new.columns]
    if missing_cols:
        print(f"Error: Missing columns in downloaded data: {missing_cols}")
        print(f"Available columns: {df_new.columns}")
        return

    df_new = df_new[required_cols]

    # Filter new data to be after the last timestamp
    if last_timestamp:
        new_rows = df_new[df_new["timestamp"] > last_timestamp]
        print(f"Found {len(new_rows)} new rows.")
    else:
        new_rows = df_new
        print(f"Found {len(new_rows)} rows (new dataset).")

    if new_rows.empty:
        print("No new data to append.")
        return

    # Append new data
    df_updated = pd.concat([df_existing, new_rows], ignore_index=True)

    # Sort by timestamp
    df_updated = df_updated.sort_values("timestamp")

    # Drop duplicates just in case
    df_updated = df_updated.drop_duplicates(subset=["timestamp"], keep="last")

    print(f"Updated dataset shape: {df_updated.shape}")

    # Save
    print(f"Saving to {raw_data_path}...")
    df_updated.to_csv(raw_data_path, index=False)
    print("Done.")


if __name__ == "__main__":
    update_dataset()
