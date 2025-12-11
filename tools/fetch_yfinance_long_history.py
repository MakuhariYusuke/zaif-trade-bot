import pandas as pd
import yfinance as yf


def fetch_long_history(output_path="data/btc_jpy_5m_yfinance.csv"):
    ticker = "BTC-JPY"
    print(f"Fetching 60 days of 5m data for {ticker}...")

    # Fetch 5-minute data for the last 60 days
    data = yf.download(ticker, period="60d", interval="5m")

    if len(data) == 0:
        print("No data fetched.")
        return

    print(f"Fetched {len(data)} rows.")

    data.reset_index(inplace=True)

    # Handle MultiIndex
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)

    rename_map = {
        "Datetime": "timestamp",
        "Open": "open",
        "High": "high",
        "Low": "low",
        "Close": "close",
        "Volume": "volume",
    }
    data.rename(columns=rename_map, inplace=True)

    # Ensure columns
    required_cols = ["timestamp", "open", "high", "low", "close", "volume"]
    for col in required_cols:
        if col not in data.columns:
            for c in data.columns:
                if c.lower() == col:
                    data.rename(columns={c: col}, inplace=True)

    final_df = data[required_cols].copy()

    if final_df["timestamp"].dt.tz is not None:
        final_df["timestamp"] = final_df["timestamp"].dt.tz_convert("UTC")

    final_df.to_csv(output_path, index=False)
    print(f"Saved to {output_path}")


if __name__ == "__main__":
    fetch_long_history()
