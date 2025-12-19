import os
import sys
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# Add project root to path
project_root = Path(__file__).resolve().parents[1]
sys.path.append(str(project_root))

def analyze_v452_results():
    # Paths
    results_dir = project_root / "backtest_results" / "v452_optimized"
    csv_path = results_dir / "backtest_results.csv"
    report_path = results_dir / "analysis_report_v452.md"
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

    print(f"Loaded {len(df)} steps.")

    # --- 1. Overall Performance Metrics ---
    initial_balance = df["portfolio_value"].iloc[0]
    final_balance = df["portfolio_value"].iloc[-1]
    total_pnl = df["pnl"].iloc[-1]
    total_return_pct = (final_balance - initial_balance) / initial_balance * 100

    # Drawdown
    peak = df["portfolio_value"].cummax()
    drawdown = (df["portfolio_value"] - peak) / peak
    max_drawdown_pct = drawdown.min() * 100

    # Daily Returns for Sharpe/Sortino
    # Resample to daily to get standard annualized metrics
    daily_df = df["portfolio_value"].resample("D").last().dropna()
    daily_returns = daily_df.pct_change().dropna()
    
    risk_free_rate = 0.0
    mean_daily_return = daily_returns.mean()
    std_daily_return = daily_returns.std()
    
    sharpe_ratio = 0
    if std_daily_return != 0:
        sharpe_ratio = (mean_daily_return - risk_free_rate) / std_daily_return * np.sqrt(365)

    downside_returns = daily_returns[daily_returns < 0]
    sortino_ratio = 0
    if len(downside_returns) > 0:
        downside_std = downside_returns.std()
        if downside_std != 0:
            sortino_ratio = (mean_daily_return - risk_free_rate) / downside_std * np.sqrt(365)

    # --- 2. Trade Analysis (Reconstruction) ---
    # We reconstruct trades based on PnL changes. 
    # Note: This is an approximation if we don't have explicit trade logs.
    
    daily_pnl_change = df["pnl"].resample("D").last().diff().dropna()
    winning_days = len(daily_pnl_change[daily_pnl_change > 0])
    losing_days = len(daily_pnl_change[daily_pnl_change < 0])
    total_days = len(daily_pnl_change)
    win_rate_daily = winning_days / total_days * 100 if total_days > 0 else 0

    # --- 3. Regime Analysis ---
    regime_metrics = {}
    if "regime" in df.columns:
        # Group by regime
        # We want to see:
        # 1. Count of steps in each regime
        # 2. Average Return in each regime
        # 3. Volatility in each regime
        
        df["step_return"] = df["portfolio_value"].pct_change().fillna(0)
        
        regime_groups = df.groupby("regime")
        
        for regime, group in regime_groups:
            count = len(group)
            avg_return = group["step_return"].mean() * 100 # per step
            cum_return = (1 + group["step_return"]).prod() - 1
            std_return = group["step_return"].std()
            
            regime_metrics[regime] = {
                "steps": count,
                "pct_of_time": count / len(df) * 100,
                "avg_step_return_pct": avg_return,
                "cum_return_pct": cum_return * 100,
                "volatility": std_return
            }

    # --- 4. Action Analysis ---
    action_counts = df["action_type"].value_counts()
    action_props = df["action_type"].value_counts(normalize=True) * 100

    # --- 5. Generate Report ---
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("# Backtest Analysis Report (v452 Optimized)\n\n")
        f.write(f"**Date**: {pd.Timestamp.now()}\n\n")
        
        f.write("## 1. Overall Performance\n")
        f.write("| Metric | Value |\n")
        f.write("|---|---|\n")
        f.write(f"| Initial Balance | {initial_balance:,.2f} |\n")
        f.write(f"| Final Balance | {final_balance:,.2f} |\n")
        f.write(f"| Total PnL | {total_pnl:,.2f} |\n")
        f.write(f"| Total Return | {total_return_pct:.2f}% |\n")
        f.write(f"| Max Drawdown | {max_drawdown_pct:.2f}% |\n")
        f.write(f"| Sharpe Ratio (Daily) | {sharpe_ratio:.2f} |\n")
        f.write(f"| Sortino Ratio (Daily) | {sortino_ratio:.2f} |\n")
        f.write(f"| Daily Win Rate | {win_rate_daily:.2f}% ({winning_days}/{total_days} days) |\n\n")
        
        f.write("## 2. Regime Analysis\n")
        f.write("Performance breakdown by market regime.\n\n")
        f.write("| Regime | Steps | % Time | Cum Return | Avg Step Return | Volatility |\n")
        f.write("|---|---|---|---|---|---|\n")
        
        for regime, metrics in regime_metrics.items():
            f.write(f"| {regime} | {metrics['steps']} | {metrics['pct_of_time']:.1f}% | {metrics['cum_return_pct']:.2f}% | {metrics['avg_step_return_pct']:.4f}% | {metrics['volatility']:.6f} |\n")
        f.write("\n")
        
        f.write("## 3. Action Distribution\n")
        f.write("| Action | Count | Proportion |\n")
        f.write("|---|---|---|\n")
        for action, count in action_counts.items():
            prop = action_props[action]
            f.write(f"| {action} | {count} | {prop:.2f}% |\n")
        f.write("\n")

    print(f"Report saved to {report_path}")

    # --- 6. Plots ---
    # Set style
    sns.set_theme(style="darkgrid")
    
    # Plot 1: Portfolio Value & Drawdown
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True, gridspec_kw={'height_ratios': [3, 1]})
    
    ax1.plot(df.index, df["portfolio_value"], label="Portfolio Value", color="blue")
    ax1.set_title("Portfolio Value Over Time")
    ax1.set_ylabel("Value")
    ax1.legend()
    
    ax2.fill_between(df.index, drawdown * 100, 0, color="red", alpha=0.3, label="Drawdown %")
    ax2.set_title("Drawdown")
    ax2.set_ylabel("Drawdown %")
    ax2.set_xlabel("Date")
    
    plt.tight_layout()
    plt.savefig(plot_dir / "portfolio_performance.png")
    plt.close()
    
    # Plot 2: Cumulative Return by Regime
    if "regime" in df.columns:
        plt.figure(figsize=(12, 6))
        
        # Create a cumulative return column for plotting
        df["cum_return"] = (1 + df["step_return"]).cumprod()
        
        # Scatter plot colored by regime
        # Since line plot with changing colors is tricky, we use scatter for regime indication
        # or we can plot separate lines for each regime segment (complex).
        # Simpler: Plot Cum Return line, and add colored background stripes for regimes.
        
        ax = plt.gca()
        ax.plot(df.index, df["cum_return"], label="Cumulative Return", color="black", linewidth=1)
        
        # We need to find start/end of regime blocks
        # This is computationally intensive for 1m data if regimes switch often.
        # We'll just plot points colored by regime.
        
        # Map regimes to colors
        unique_regimes = df["regime"].unique()
        palette = sns.color_palette("hsv", len(unique_regimes))
        regime_colors = dict(zip(unique_regimes, palette))
        
        # Downsample for plotting if too large
        plot_df = df.iloc[::10] if len(df) > 10000 else df
        
        sns.scatterplot(data=plot_df, x=plot_df.index, y="cum_return", hue="regime", s=10, linewidth=0, alpha=0.5)
        
        plt.title("Cumulative Return by Market Regime")
        plt.ylabel("Cumulative Return (Factor)")
        plt.xlabel("Date")
        plt.savefig(plot_dir / "regime_performance.png")
        plt.close()

    print(f"Plots saved to {plot_dir}")

if __name__ == "__main__":
    analyze_v452_results()
