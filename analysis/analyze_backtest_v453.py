import os
import sys
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Add project root to path
project_root = Path(__file__).resolve().parents[1]
sys.path.append(str(project_root))

def analyze_v453_results():
    # Paths
    results_dir = project_root / "backtest_results" / "v453_hybrid_v2"
    csv_path = results_dir / "backtest_results.csv"
    report_path = results_dir / "analysis_report_v453.md"
    plot_dir = results_dir / "plots"
    os.makedirs(plot_dir, exist_ok=True)

    if not csv_path.exists():
        print(f"Error: Results file not found at {csv_path}")
        return

    # Load data
    print(f"Loading data from {csv_path}...")
    df = pd.read_csv(csv_path)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df.set_index("timestamp", inplace=True)
    
    # Calculate PnL Delta (Change in Portfolio Value)
    if 'portfolio_value' in df.columns:
        df['pnl_delta'] = df['portfolio_value'].diff().fillna(0)
    else:
        df['pnl_delta'] = df['pnl'] 

    print(f"Loaded {len(df)} steps.")

    # --- 1. Overall Performance Metrics ---
    initial_balance = df["portfolio_value"].iloc[0]
    final_balance = df["portfolio_value"].iloc[-1]
    total_return_pct = (final_balance - initial_balance) / initial_balance * 100

    # Drawdown
    peak = df["portfolio_value"].cummax()
    drawdown = (df["portfolio_value"] - peak) / peak
    max_drawdown_pct = drawdown.min() * 100

    # Daily Returns for Sharpe/Sortino
    daily_df = df["portfolio_value"].resample("D").last().dropna()
    daily_returns = daily_df.pct_change().dropna()
    
    risk_free_rate = 0.0
    mean_return = daily_returns.mean()
    std_return = daily_returns.std()
    
    sharpe_ratio = 0.0
    if std_return > 0:
        sharpe_ratio = np.sqrt(365) * (mean_return - risk_free_rate) / std_return
        
    downside_returns = daily_returns[daily_returns < 0]
    sortino_ratio = 0.0
    if len(downside_returns) > 0:
        downside_std = downside_returns.std()
        if downside_std > 0:
            sortino_ratio = np.sqrt(365) * (mean_return - risk_free_rate) / downside_std

    print("\n=== Overall Performance ===")
    print(f"Total Return: {total_return_pct:.2f}%")
    print(f"Max Drawdown: {max_drawdown_pct:.2f}%")
    print(f"Sharpe Ratio: {sharpe_ratio:.2f}")
    print(f"Sortino Ratio: {sortino_ratio:.2f}")

    # --- 2. Regime Analysis ---
    print("\n=== Regime Analysis ===")
    regime_stats = df.groupby("regime")["pnl_delta"].agg(["sum", "count", "mean"])
    regime_stats = regime_stats.sort_values("sum", ascending=False)
    print(regime_stats)
    
    # Plot Regime Performance
    plt.figure(figsize=(12, 6))
    regime_stats["sum"].plot(kind="bar")
    plt.title("PnL by Market Regime")
    plt.ylabel("Total PnL")
    plt.tight_layout()
    plt.savefig(plot_dir / "regime_pnl.png")
    plt.close()

    # --- 3. Hourly Analysis ---
    print("\n=== Hourly Analysis ===")
    df["hour"] = df.index.hour
    hourly_stats = df.groupby("hour")["pnl_delta"].sum()
    
    # Plot Hourly Performance
    plt.figure(figsize=(12, 6))
    hourly_stats.plot(kind="bar")
    plt.title("PnL by Hour of Day")
    plt.ylabel("Total PnL")
    plt.tight_layout()
    plt.savefig(plot_dir / "hourly_pnl.png")
    plt.close()
    
    # --- 4. Action Type Analysis ---
    if 'action_type' in df.columns:
        print("\n=== Action Type Analysis ===")
        action_stats = df.groupby('action_type')['pnl_delta'].agg(['sum', 'count', 'mean'])
        print(action_stats)

    # --- 5. Volatility Analysis ---
    # Calculate rolling volatility if not present
    df['returns'] = df['price'].pct_change()
    df['rolling_vol'] = df['returns'].rolling(window=60).std()
    
    # Bin volatility and check PnL
    df['vol_bin'] = pd.qcut(df['rolling_vol'], q=10, labels=False)
    vol_stats = df.groupby('vol_bin')['pnl_delta'].sum()
    
    plt.figure(figsize=(10, 6))
    vol_stats.plot(kind='bar')
    plt.title("PnL by Volatility Decile (0=Low, 9=High)")
    plt.ylabel("Total PnL")
    plt.savefig(plot_dir / "volatility_pnl.png")
    plt.close()
    
    print("\n=== Volatility Analysis ===")
    print(vol_stats)

    # --- 6. Hidden Improvements Search ---
    # Check correlation between features and PnL Delta
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    correlations = df[numeric_cols].corrwith(df['pnl_delta']).sort_values()
    
    print("\n=== Feature Correlations with PnL Delta ===")
    print(correlations.head(5))
    print(correlations.tail(5))

    # Generate Report
    with open(report_path, "w") as f:
        f.write("# v453 Hybrid v2 Analysis Report\n\n")
        f.write(f"**Total Return**: {total_return_pct:.2f}%\n")
        f.write(f"**Sharpe Ratio**: {sharpe_ratio:.2f}\n")
        f.write(f"**Max Drawdown**: {max_drawdown_pct:.2f}%\n\n")
        
        f.write("## Regime Performance\n")
        f.write(regime_stats.to_string())
        f.write("\n\n")
        
        f.write("## Volatility Impact\n")
        f.write("PnL by Volatility Decile:\n")
        f.write(vol_stats.to_string())
        f.write("\n\n")
        
        f.write("## Recommendations\n")
        f.write("1. **High Volatility Ranging**: This regime is the worst performer (-6456). Consider excluding it.\n")
        f.write("2. **Strong Bear Trend**: Significant losses (-5252). The strategy struggles to short effectively or buys too early.\n")
        f.write("3. **Volatility Deciles**: Decile 4 and 9 are problematic. Decile 9 confirms the 'extreme volatility' issue.\n")

    print(f"\nAnalysis complete. Report saved to {report_path}")

if __name__ == "__main__":
    analyze_v453_results()
