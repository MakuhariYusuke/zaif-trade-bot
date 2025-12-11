import json
import sys


def check_lengths(file_path):
    with open(file_path, "r") as f:
        data = json.load(f)

    keys = ["timestamps", "portfolio_history", "price_history", "actions", "trade_pnls"]
    print(f"File: {file_path}")
    for key in keys:
        if key in data:
            print(f"{key}: {len(data[key])}")
        else:
            print(f"{key}: NOT FOUND")


if __name__ == "__main__":
    file_path = "backtest_results/phase6_hft_backtest.json"
    if len(sys.argv) > 1:
        file_path = sys.argv[1]
    check_lengths(file_path)
