import json

import pandas as pd


def analyze_results(file_path):
    with open(file_path, "r") as f:
        data = json.load(f)

    trades = data.get("trades", [])
    print(f"Total Trades: {len(trades)}")

    if not trades:
        print("No trades found.")
        return

    df = pd.DataFrame(trades)
    print("\nAction Distribution:")
    print(df["type"].value_counts(normalize=True))
    print(df["type"].value_counts())

    print("\nEntry Reasons:")
    if "entry_reason" in df.columns:
        print(df["entry_reason"].value_counts())

    print("\nExit Reasons:")
    if "exit_reason" in df.columns:
        print(df["exit_reason"].value_counts())

    # Calculate average holding time
    df["entry_time"] = pd.to_datetime(df["entry_time"])
    df["exit_time"] = pd.to_datetime(df["exit_time"])
    df["duration"] = df["exit_time"] - df["entry_time"]
    print(f"\nAverage Holding Time: {df['duration'].mean()}")


if __name__ == "__main__":
    analyze_results("backtest_results_sac_v446.json")
