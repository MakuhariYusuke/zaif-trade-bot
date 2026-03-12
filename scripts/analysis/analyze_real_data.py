import os

import numpy as np
import pandas as pd


def analyze_real_market_data():
    """Analyze real market data for ActionSignalGuide validation"""

    # Load real market data
    data_path = "data/btc_jpy_1m_dataset.csv"
    if not os.path.exists(data_path):
        print(f"Data file not found: {data_path}")
        return None

    df = pd.read_csv(data_path)
    print(f"Loaded {len(df)} rows of 1-minute BTC/JPY data")
    print(f"Data columns: {list(df.columns)}")
    print(f'Date range: {df["timestamp"].min()} to {df["timestamp"].max()}')
    print(f'Price range: {df["close"].min():.0f} - {df["close"].max():.0f} JPY')
    print(f'Average volume: {df["volume"].mean():.0f}')

    # Check for missing values
    missing = df.isnull().sum()
    print(f"Missing values: {missing.sum()}")

    # Convert timestamp to datetime
    df["timestamp"] = pd.to_datetime(df["timestamp"])

    # Basic data quality checks
    print("\nData quality checks:")
    print(f'- Valid OHLC: {(df["high"] >= df["low"]).all()}')
    print(
        f'- Valid OHLCV: {(df["high"] >= df["close"]) & (df["close"] >= df["low"]) & (df["high"] >= df["open"]) & (df["open"] >= df["low"]).all()}'
    )

    # Calculate some basic statistics
    returns = df["close"].pct_change()
    print("\nBasic statistics:")
    print(f"- Mean return: {returns.mean():.6f}")
    print(f"- Std return: {returns.std():.6f}")
    print(f"- Max return: {returns.max():.6f}")
    print(f"- Min return: {returns.min():.6f}")

    # Volatility analysis
    volatility = returns.rolling(60).std() * np.sqrt(
        1440
    )  # Annualized volatility (60 min window)
    print(f"- Current volatility (60min window): {volatility.iloc[-1]:.4f}")

    # Volume analysis
    volume_ma = df["volume"].rolling(60).mean()
    print(f"- Average volume (60min): {volume_ma.iloc[-1]:.0f}")

    # Sample recent data for testing
    recent_data = df.tail(200).copy()  # Last 200 minutes
    print("\nRecent data sample (last 200 minutes):")
    print(
        f'- Time range: {recent_data["timestamp"].min()} to {recent_data["timestamp"].max()}'
    )
    print(
        f'- Price range: {recent_data["close"].min():.0f} - {recent_data["close"].max():.0f}'
    )
    print(
        f'- Volume range: {recent_data["volume"].min():.0f} - {recent_data["volume"].max():.0f}'
    )

    return df


if __name__ == "__main__":
    analyze_real_market_data()
