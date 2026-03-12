import os
from datetime import datetime

import pandas as pd
import yfinance as yf


def download_yahoo_data():
    ticker = "BTC-JPY"
    interval = "1m"
    # Yahoo Finance 1m data is typically limited to the last 7 days.
    # We'll try to fetch the last 7 days.
    period = "7d"

    print(f"Downloading {ticker} data (Interval: {interval}, Period: {period})...")

    try:
        # Download data
        df = yf.download(ticker, interval=interval, period=period)

        if df.empty:
            print("No data downloaded. Please check the ticker or internet connection.")
            return

        # Reset index to make timestamp a column
        df = df.reset_index()

        # Rename columns to match the project standard
        # yfinance returns: Date/Datetime, Open, High, Low, Close, Adj Close, Volume
        # We need: timestamp, close, high, low, open, volume, adj_close

        # Check column names
        print("Columns found:", df.columns)

        # Flatten MultiIndex columns if present
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = [
                col[0] if isinstance(col, tuple) else col for col in df.columns
            ]
            print("Flattened columns:", df.columns)

        # Normalize column names
        df.columns = [c.lower() for c in df.columns]

        # Rename 'datetime' or 'date' to 'timestamp'
        if "datetime" in df.columns:
            df = df.rename(columns={"datetime": "timestamp"})
        elif "date" in df.columns:
            df = df.rename(columns={"date": "timestamp"})

        # Ensure required columns exist
        required_columns = ["timestamp", "open", "high", "low", "close", "volume"]
        if not all(col in df.columns for col in required_columns):
            print(f"Missing required columns. Found: {df.columns}")
            return

        # Add adj_close if missing (copy close)
        if "adj_close" not in df.columns:
            if "adj close" in df.columns:
                df = df.rename(columns={"adj close": "adj_close"})
            else:
                df["adj_close"] = df["close"]

        # Select and reorder columns
        output_columns = [
            "timestamp",
            "close",
            "high",
            "low",
            "open",
            "volume",
            "adj_close",
        ]
        df = df[output_columns]

        # Save to CSV
        output_dir = os.path.join(os.getcwd(), "data")
        os.makedirs(output_dir, exist_ok=True)

        timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"btc_jpy_1m_yahoo_{timestamp_str}.csv"
        filepath = os.path.join(output_dir, filename)

        df.to_csv(filepath, index=False)
        print(f"Successfully saved {len(df)} rows to {filepath}")
        print(df.head())
        print(df.tail())

    except Exception as e:
        print(f"An error occurred: {e}")


if __name__ == "__main__":
    download_yahoo_data()
