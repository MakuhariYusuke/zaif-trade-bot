import pandas as pd

from backtest.data_generator import generate_synthetic_data


def analyze_price_trend():
    # Generate the same synthetic data as the backtest
    # The backtest uses n_periods=10000 by default if file not found
    # But wait, the backtest log said "Loaded 35334 rows from data/yahoo_finance/btc_jpy_1m_converted.csv" ?
    # No, the log in the previous turn showed:
    # '2025-11-05 17:10:00', ...
    # And the file read of backtest_results_sac_v446.json showed "total_steps": 7062

    # Let's try to load the actual file if it exists
    data_path = "data/yahoo_finance/btc_jpy_1m_converted.csv"
    try:
        df = pd.read_csv(data_path)
        print(f"Loaded data from {data_path}")
        if "timestamp" in df.columns:
            df["timestamp"] = pd.to_datetime(df["timestamp"])
            df.set_index("timestamp", inplace=True)
    except FileNotFoundError:
        print(f"File not found: {data_path}. Using synthetic data.")
        df = generate_synthetic_data(n_periods=10000)

    # Filter for the backtest period if possible
    # The backtest log showed dates from 2025-11-05 to 2025-11-09
    # Let's see if we can find these dates in the dataframe

    print(f"Data range: {df.index.min()} to {df.index.max()}")

    # If the data is synthetic, it might be generated on the fly with current dates?
    # generate_synthetic_data usually uses a start date.

    # Let's just look at the price change
    start_price = df["close"].iloc[0]
    end_price = df["close"].iloc[-1]

    print(f"Start Price: {start_price}")
    print(f"End Price: {end_price}")

    change = (end_price - start_price) / start_price * 100
    print(f"Price Change: {change:.2f}%")

    # Check for downtrend
    if change < 0:
        print("The market was in a DOWNTREND.")
    else:
        print("The market was in an UPTREND.")


if __name__ == "__main__":
    analyze_price_trend()
