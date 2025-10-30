from ztb.analysis.comparative.analyze_backtest import BacktestAnalyzer

# Use the v443_2_phase2 backtest results
results_path = "backtest_results/v443_2_phase2/rl_20251031_004029/metrics.json"

# Create analyzer and run analysis
analyzer = BacktestAnalyzer(results_path)
analysis = analyzer.analyze_backtest_results()

print("=== V443.2 PHASE 2 BACKTEST ANALYSIS RESULTS ===")
print(f'Total Return: {analysis.get("total_return", "N/A")}')
print(f'Sharpe Ratio: {analysis.get("sharpe_ratio", "N/A")}')
print(f'Max Drawdown: {analysis.get("max_drawdown", "N/A")}')
print(f'Win Rate: {analysis.get("win_rate", "N/A")}')
print(f'Total Trades: {analysis.get("total_trades", "N/A")}')

# Print detailed analysis
print("\n=== DETAILED ANALYSIS ===")
for key, value in analysis.items():
    if key in [
        "total_return",
        "sharpe_ratio",
        "max_drawdown",
        "win_rate",
        "total_trades",
        "volatility",
        "cagr",
    ]:
        print(f"{key}: {value}")
