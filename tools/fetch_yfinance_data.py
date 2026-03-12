import pandas as pd
import yfinance as yf


def fetch_yfinance_data(output_path="data/btc_jpy_1m_yfinance.csv"):
    # BTC-JPY ticker
    ticker = "BTC-JPY"

    print(f"Fetching data for {ticker} from Yahoo Finance...")

    # Fetch 1-minute data for the last 7 days (max allowed by Yahoo for 1m)
    # We can try 'max' but usually it limits 1m data.
    data = yf.download(ticker, period="7d", interval="1m")

    if len(data) == 0:
        print("No data fetched.")
        return

    print(f"Fetched {len(data)} rows.")
    print(data.head())
    print(data.tail())

    # Reset index to get timestamp as column
    data.reset_index(inplace=True)

    # Rename columns to match our dataset format
    # Yahoo columns: Datetime, Open, High, Low, Close, Adj Close, Volume
    # Our format: timestamp, open, high, low, close, volume

    # Check columns
    print("Columns:", data.columns)

    # Handle MultiIndex columns if present (yfinance sometimes returns them)
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)

    # Rename
    rename_map = {
        "Datetime": "timestamp",
        "Open": "open",
        "High": "high",
        "Low": "low",
        "Close": "close",
        "Volume": "volume",
    }
    data.rename(columns=rename_map, inplace=True)

    # Select required columns
    required_cols = ["timestamp", "open", "high", "low", "close", "volume"]
    # Ensure all exist
    for col in required_cols:
        if col not in data.columns:
            print(f"Missing column: {col}")
            # Try case insensitive match
            for c in data.columns:
                if c.lower() == col:
                    data.rename(columns={c: col}, inplace=True)

    final_df = data[required_cols].copy()

    # Convert timestamp to standard format if needed, but yfinance usually gives datetime objects
    # Ensure UTC or consistent timezone
    if final_df["timestamp"].dt.tz is not None:
        final_df["timestamp"] = final_df["timestamp"].dt.tz_convert("UTC")

    # Save
    final_df.to_csv(output_path, index=False)
    print(f"Saved to {output_path}")

    return final_df


if __name__ == "__main__":
    fetch_yfinance_data()
