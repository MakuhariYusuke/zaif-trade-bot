import json
import os
import sys
from pathlib import Path

import pandas as pd

# Add project root to path
project_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(project_root))

from ztb.analysis.market_regime_classifier import MarketRegimeClassifier


def analyze_results():
    # Load results
    results_path = os.path.join(
        project_root, "backtest_results", "v451_optimized", "backtest_results.csv"
    )
    if not os.path.exists(results_path):
        print("Results file not found.")
        return

    results_df = pd.read_csv(results_path)

    # Load original data to reconstruct regimes
    data_path = os.path.join(project_root, "data", "btc_jpy_1m_v451.csv")
    market_data = pd.read_csv(data_path, index_col=0, parse_dates=True)

    # Align data lengths
    # The backtest might have started from a specific index or used a subset
    # Assuming the backtest ran from the beginning (after warmup)
    # HeavyTradingEnv usually has a warmup period.
    # Let's assume the results correspond to the end of the dataset or start after warmup.
    # But without exact indices, it's hard.
    # However, the backtest script didn't specify start/end, so it likely used the whole dataset minus warmup.

    # Let's try to match by length from the beginning
    # The backtest likely started from the beginning of the dataset (after warmup)
    n_steps = len(results_df)
    # Assuming a small warmup or starting from index 0 for simplicity in alignment
    # If there is a warmup, the regimes might be slightly shifted, but for distribution analysis it's okay.
    aligned_data = market_data.iloc[:n_steps].copy()

    # Initialize Regime Classifier
    config_path = os.path.join(
        project_root, "config", "v451", "sac_v451_optimized.json"
    )
    with open(config_path, "r") as f:
        full_config = json.load(f)
    regime_config = full_config["training"]["environment"]["config"][
        "advanced_market_regime"
    ]["regime_classifier_config"]
    classifier = MarketRegimeClassifier(regime_config)

    # Detect regimes (this might be slow for 10k steps, but acceptable)
    print("Detecting regimes for analysis...")
    regimes = []

    # We need a window for detection, so we need data before the aligned segment
    # We'll use the full market_data but only store results for the aligned part

    # Actually, let's just do a quick approximation or use the classifier on the aligned data
    # taking into account the lookback.

    # Better: Iterate through the aligned data indices
    start_idx = len(market_data) - n_steps

    for i in range(n_steps):
        current_idx = start_idx + i
        # We need to pass the full dataframe and the current integer index
        result = classifier.detect_regime(market_data, current_idx)
        # Handle both object and tuple return types just in case, but based on error it is an object
        if hasattr(result, "primary_regime"):
            regime = result.primary_regime
        elif isinstance(result, tuple):
            regime = result[0]
        else:
            regime = result

        regimes.append(regime.value if hasattr(regime, "value") else str(regime))

        if i % 1000 == 0:
            print(f"Processed {i}/{n_steps} steps")

    results_df["regime"] = regimes

    # Discretize actions for analysis
    # Using a simple threshold of 0.05 as a proxy for the dynamic threshold
    threshold = 0.05
    results_df["discrete_action"] = 0
    results_df.loc[results_df["action"] > threshold, "discrete_action"] = 1
    results_df.loc[results_df["action"] < -threshold, "discrete_action"] = -1

    # Analysis
    print("\n--- Analysis Report ---")
    print(f"Total Steps: {len(results_df)}")
    print(f"Final Portfolio Value: {results_df['portfolio_value'].iloc[-1]:.2f}")

    # Action Distribution
    # 0: Hold, 1: Buy, -1: Sell
    action_counts = results_df["discrete_action"].value_counts().sort_index()
    print("\nAction Distribution (Discrete):")
    print(action_counts)

    # Regime Distribution
    print("\nRegime Distribution:")
    print(results_df["regime"].value_counts())

    # Action by Regime
    print("\nAction Distribution by Regime (Normalized):")
    ct = pd.crosstab(
        results_df["regime"], results_df["discrete_action"], normalize="index"
    )
    print(ct)

    # Check specifically for Buy Breakout
    target_regimes = ["buy_breakout", "strong_bull_trend", "high_volatility_ranging"]
    print("\nTarget Regime Analysis:")
    for reg in target_regimes:
        if reg in ct.index:
            print(f"\nRegime: {reg}")
            print(ct.loc[reg])

            # Check if we are buying more than selling
            # Action -1 is Sell, 1 is Buy
            buy_rate = ct.loc[reg].get(1, 0)
            sell_rate = ct.loc[reg].get(-1, 0)
            print(f"Buy Rate: {buy_rate:.2%}")
            print(f"Sell Rate: {sell_rate:.2%}")

            if reg == "buy_breakout":
                if buy_rate > sell_rate:
                    print("SUCCESS: Buying dominates in this bullish regime.")
                else:
                    print("FAILURE: Still selling more or holding.")


if __name__ == "__main__":
    analyze_results()
