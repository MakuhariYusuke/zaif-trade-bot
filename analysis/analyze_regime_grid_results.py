import json
import os

def analyze_regime_grid_results():
    """Analyze the regime grid backtest results"""

    results_file = "backtest_results/regime_grid_results.json"

    if not os.path.exists(results_file):
        print(f"Error: {results_file} not found")
        return

    with open(results_file, "r") as f:
        data = json.load(f)

    print("=== トレンドレジーム バックテスト結果分析 ===")
    print(f"総結果数: {len(data)}")
    print()

    # 各レジームの結果を表示
    for result in data:
        regime = result.get("regime", "unknown")
        total_return = result.get("total_return_pct", 0)
        trades = result.get("total_trades", 0)
        win_rate = result.get("win_rate", 0) * 100
        sharpe = result.get("sharpe_ratio", 0)
        max_dd = result.get("max_drawdown_pct", 0) * 100
        final_balance = result.get("final_balance", 0)

        print(f"レジーム: {regime}")
        print(f"  リターン: {total_return:.2f}%")
        print(f"  最終残高: {final_balance:,.0f}円")
        print(f"  総トレード数: {trades}")
        print(f"  勝率: {win_rate:.1f}%")
        print(f"  シャープレシオ: {sharpe:.3f}")
        print(f"  最大ドローダウン: {max_dd:.2f}%")
        print()

    # サマリー統計
    if data:
        avg_return = sum(r.get("total_return_pct", 0) for r in data) / len(data)
        total_trades = sum(r.get("total_trades", 0) for r in data)
        avg_win_rate = sum(r.get("win_rate", 0) for r in data) / len(data) * 100

        print("=== サマリー統計 ===")
        print(f"平均リターン: {avg_return:.2f}%")
        print(f"総トレード数: {total_trades}")
        print(f"平均勝率: {avg_win_rate:.1f}%")

if __name__ == "__main__":
    analyze_regime_grid_results()