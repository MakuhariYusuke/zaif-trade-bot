import os
import sys
from pathlib import Path
import pandas as pd
import numpy as np

# Add project root to path
project_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(project_root))

def analyze_v453_deep_dive():
    results_dir = os.path.join(project_root, "backtest_results", "v453_hybrid_v2")
    csv_path = os.path.join(results_dir, "backtest_results.csv")

    if not os.path.exists(csv_path):
        print(f"Error: Results file not found at {csv_path}")
        return

    print(f"Loading results from {csv_path}...")
    df = pd.read_csv(csv_path)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df["hour"] = df["timestamp"].dt.hour
    
    # Calculate PnL Delta (Change in Portfolio Value)
    df['pnl_delta'] = df['portfolio_value'].diff().fillna(0)
    
    # --- 1. Regime Analysis (Deep Dive) ---
    print("\n=== 1. Regime Performance Analysis ===")
    regime_stats = df.groupby("regime").agg({
        "pnl_delta": ["count", "sum", "mean", "std", "min", "max"],
        "action": lambda x: (x != 0).sum() # Count of non-hold actions (approx)
    })
    regime_stats.columns = ["steps", "total_pnl", "avg_pnl", "std_pnl", "min_pnl", "max_pnl", "active_actions"]
    regime_stats["pnl_per_step"] = regime_stats["total_pnl"] / regime_stats["steps"]
    regime_stats = regime_stats.sort_values("total_pnl", ascending=True)
    print(regime_stats)
    
    # --- 2. High Volatility Ranging Analysis ---
    print("\n=== 2. High Volatility Ranging Deep Dive ===")
    hvr_df = df[df['regime'] == 'high_volatility_ranging'].copy()
    if not hvr_df.empty:
        print(f"Total Steps: {len(hvr_df)}")
        print(f"Total PnL: {hvr_df['pnl_delta'].sum():.2f}")
        print(f"Average Volatility (if available): {hvr_df['price'].pct_change().std() if 'price' in hvr_df.columns else 'N/A'}")
        
        # Hourly breakdown of HVR
        hvr_hourly = hvr_df.groupby('hour')['pnl_delta'].sum().sort_values()
        print("\nWorst Hours for High Volatility Ranging:")
        print(hvr_hourly.head(5))
    
    # --- 3. Trade Duration Analysis ---
    # We need to reconstruct trades from the log to analyze duration
    # This is an approximation if we don't have trade IDs
    # Assuming 'action_type' has ENTRY/EXIT or we infer from position changes
    # For now, let's look at consecutive non-zero positions if possible, or just skip if complex.
    # Instead, let's look at "Holding Period" impact.
    
    # --- 4. Entry Efficiency ---
    # Look at price change 5, 15, 60 mins after an ENTRY
    if 'action_type' in df.columns:
        print("\n=== 4. Entry Efficiency Analysis ===")
        entries = df[df['action_type'].isin(['BUY', 'SELL'])].copy()
        entries['next_price_5m'] = df['price'].shift(-5)
        entries['next_price_15m'] = df['price'].shift(-15)
        entries['next_price_60m'] = df['price'].shift(-60)
        
        # Calculate return for the direction taken
        # If BUY, return = (next - current) / current
        # If SELL, return = (current - next) / current
        
        def calc_return(row, horizon_col):
            if pd.isna(row[horizon_col]): return np.nan
            ret = (row[horizon_col] - row['price']) / row['price']
            if row['action_type'] == 'SELL':
                ret = -ret
            return ret
            
        entries['ret_5m'] = entries.apply(lambda x: calc_return(x, 'next_price_5m'), axis=1)
        entries['ret_15m'] = entries.apply(lambda x: calc_return(x, 'next_price_15m'), axis=1)
        entries['ret_60m'] = entries.apply(lambda x: calc_return(x, 'next_price_60m'), axis=1)
        
        print("Average Return after Entry:")
        print(f"5 min: {entries['ret_5m'].mean()*100:.4f}%")
        print(f"15 min: {entries['ret_15m'].mean()*100:.4f}%")
        print(f"60 min: {entries['ret_60m'].mean()*100:.4f}%")
        
        print("\nEntries by Regime (Count):")
        print(entries['regime'].value_counts())
        
        print("\nAvg 60m Return by Regime:")
        print(entries.groupby('regime')['ret_60m'].mean().sort_values() * 100)

    # --- 5. Drawdown Analysis ---
    print("\n=== 5. Drawdown Analysis ===")
    df['cum_max'] = df['portfolio_value'].cummax()
    df['drawdown'] = (df['portfolio_value'] - df['cum_max']) / df['cum_max']
    max_dd = df['drawdown'].min()
    print(f"Max Drawdown: {max_dd*100:.2f}%")
    
    # Find the period of max drawdown
    dd_idx = df['drawdown'].idxmin()
    dd_date = df.loc[dd_idx, 'timestamp']
    print(f"Max Drawdown Date: {dd_date}")
    print(f"Regime at Max Drawdown: {df.loc[dd_idx, 'regime']}")
    
    # --- 6. Consecutive Losses ---
    # Identify streaks of negative PnL
    df['loss'] = df['pnl_delta'] < 0
    # ... (simple streak logic)
    
    print("\nAnalysis Complete.")

if __name__ == "__main__":
    analyze_v453_deep_dive()
