import os
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from ztb.analysis.common.plot_utils import save_plot
from ztb.metrics.metrics import sharpe_ratio
from ztb.utils.data_utils import load_csv_data


def compare_strategies(v454_path: str, v455_path: str) -> None:
    print(f"Comparing v454 ({v454_path}) vs v455 ({v455_path})")

    if not os.path.exists(v454_path):
        print(f"v454 results not found at {v454_path}")
        return
    if not os.path.exists(v455_path):
        print(f"v455 results not found at {v455_path}")
        return

    df_454 = load_csv_data(v454_path)
    df_455 = load_csv_data(v455_path)

    def calc_metrics(df: pd.DataFrame, name: str) -> pd.Series:
        """Calculate metrics for a strategy DataFrame."""
        # Assuming df has 'portfolio_value' column
        pv = df['portfolio_value'] if 'portfolio_value' in df.columns else df.iloc[:, -1]
        ret = pv.pct_change().fillna(0)
        
        total_ret = (pv.iloc[-1] / pv.iloc[0]) - 1.0
        sharpe = sharpe_ratio(ret, period_per_year=525600)  # 1m data: 525600 periods per year
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
    save_plot(out_path)
    print(f"Plot saved to {out_path}")

if __name__ == "__main__":
    # Example paths - adjust as needed
    project_root = Path(__file__).resolve().parent.parent
    v454_file = os.path.join(project_root, "backtest_results", "v451_optimized", "backtest_results.csv") # Assuming v451 is baseline/v454
    v455_file = os.path.join(project_root, "backtest_results", "v455", "shadow_results.csv")

    compare_strategies(v454_file, v455_file)
