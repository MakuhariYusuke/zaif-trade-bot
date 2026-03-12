#!/usr/bin/env python3
"""
v441モデルバックテスト実行スクリプト
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).resolve().parent
sys.path.insert(0, str(project_root))

import json

from ztb.backtest.backtest_runner import BacktestRunner


def main():
    print("Starting v441 model backtest...")

    # v441モデルのバックテスト実行
    runner = BacktestRunner(
        model_path="models/sac_model.zip",
        config_path="config/sac_v441_stability_focused_config.json",
        output_dir="backtest_results",
    )

    results = runner.run_backtest()
    print("Backtest completed!")
    print(f"Results saved to: {results}")

    # 結果を読み込んで表示
    if results and Path(results).exists():
        with open(results, "r") as f:
            data = json.load(f)

        print("\n=== BACKTEST RESULTS SUMMARY ===")
        print(f"Total Return: {data.get('total_return', 'N/A')}")
        print(f"Sharpe Ratio: {data.get('sharpe_ratio', 'N/A')}")
        print(f"Max Drawdown: {data.get('max_drawdown', 'N/A')}")
        print(f"Win Rate: {data.get('win_rate', 'N/A')}")
        print(f"Total Trades: {data.get('total_trades', 'N/A')}")


if __name__ == "__main__":
    main()
