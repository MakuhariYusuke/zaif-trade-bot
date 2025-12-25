import pandas as pd
import matplotlib.pyplot as plt
import os
from pathlib import Path

def compare_strategies(v454_path: str, v455_path: str):
    print(f"Comparing v454 ({v454_path}) vs v455 ({v455_path})")
    
    if not os.path.exists(v454_path):
        print(f"v454 results not found at {v454_path}")
        return
    if not os.path.exists(v455_path):
        print(f"v455 results not found at {v455_path}")
        return
        
    df_454 = pd.read_csv(v454_path)
    df_455 = pd.read_csv(v455_path)
    
    # Calculate Metrics
    def calc_metrics(df, name):
        pv = df['portfolio_value']
        ret = pv.pct_change().dropna()
        total_ret = (pv.iloc[-1] / pv.iloc[0]) - 1.0
        sharpe = ret.mean() / ret.std() * (252**0.5) * 1440 # Approx for 1m data? No, 1m data -> 525600 mins/yr
        # Actually Sharpe on 1m data is noisy.
        # Let's just use Total Return and Max Drawdown
        cum_max = pv.cummax()
        dd = (pv - cum_max) / cum_max
        max_dd = dd.min()
        
        print(f"--- {name} ---")
        print(f"Total Return: {total_ret*100:.2f}%")
        print(f"Max Drawdown: {max_dd*100:.2f}%")
        print(f"Final Value: {pv.iloc[-1]:.0f}")
        return pv

    pv_454 = calc_metrics(df_454, "v454")
    pv_455 = calc_metrics(df_455, "v455")
    
    # Plot
    plt.figure(figsize=(12, 6))
    plt.plot(pv_454, label='v454')
    plt.plot(pv_455, label='v455')
    plt.title('Strategy Comparison: v454 vs v455')
    plt.legend()
    plt.grid(True)
    
    out_path = "analysis_results/comparison_v454_v455.png"
    os.makedirs("analysis_results", exist_ok=True)
    plt.savefig(out_path)
    print(f"Plot saved to {out_path}")

if __name__ == "__main__":
    # Example paths - adjust as needed
    project_root = Path(__file__).resolve().parent.parent
    v454_file = os.path.join(project_root, "backtest_results", "v451_optimized", "backtest_results.csv") # Assuming v451 is baseline/v454
    v455_file = os.path.join(project_root, "backtest_results", "v455", "shadow_results.csv")
    
    compare_strategies(v454_file, v455_file)
