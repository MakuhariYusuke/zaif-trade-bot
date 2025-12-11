import json

file_path = "backtest_results/phase6_hft_backtest.json"
with open(file_path, "r") as f:
    data = json.load(f)
    print(data.keys())
