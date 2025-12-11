from pathlib import Path

import numpy as np
import pandas as pd


def analyze_data():
    project_root = Path(__file__).resolve().parents[1]
    data_path = project_root / "data" / "btc_jpy_1m_dataset.csv"

    print(f"Loading data from {data_path}")
    df = pd.read_csv(data_path)

    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        print(f"Date Range: {df['timestamp'].min()} to {df['timestamp'].max()}")

    start_price = df["close"].iloc[0]
    end_price = df["close"].iloc[-1]
    max_price = df["close"].max()
    min_price = df["close"].min()

    print(f"Start Price: {start_price}")
    print(f"End Price: {end_price}")
    print(f"Max Price: {max_price}")
    print(f"Min Price: {min_price}")

    change = (end_price - start_price) / start_price * 100
    print(f"Total Change: {change:.2f}%")

    # Calculate volatility
    df["returns"] = df["close"].pct_change()
    volatility = df["returns"].std()
    print(f"Volatility (std of returns): {volatility:.6f}")

    # Check trend
    df["ma_1000"] = df["close"].rolling(1000).mean()
    df["trend"] = np.where(df["close"] > df["ma_1000"], 1, -1)
    trend_counts = df["trend"].value_counts()
    print("Trend Distribution (vs MA1000):")
    print(trend_counts)

    # Check if there are long periods of downtrend
    print("\nChecking for major downtrends...")
    # Simple check: is the second half significantly lower than the first?
    mid = len(df) // 2
    first_half_mean = df["close"].iloc[:mid].mean()
    second_half_mean = df["close"].iloc[mid:].mean()
    print(f"First Half Mean: {first_half_mean:.2f}")
    print(f"Second Half Mean: {second_half_mean:.2f}")


if __name__ == "__main__":
    analyze_data()
