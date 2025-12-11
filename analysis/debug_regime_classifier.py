import os
import sys
from pathlib import Path

import pandas as pd
from tqdm import tqdm

# Add project root to path
project_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(project_root))

from ztb.analysis.market_regime_classifier import MarketRegimeClassifier


def analyze_regime_distribution():
    data_path = os.path.join(project_root, "data", "btc_jpy_1m_v451.csv")
    print(f"Loading data from {data_path}...")
    df = pd.read_csv(data_path)

    # Ensure datetime index if needed, though classifier uses integer index
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.set_index("timestamp")

    print("Initializing MarketRegimeClassifier...")
    classifier = MarketRegimeClassifier()

    print("Detecting regimes (sampling every 100 steps)...")

    results = []

    # Sample data to save time, but keep enough history for lookback
    # We need at least 100 points lookback.
    # Analyze the last 20000 steps to match backtest
    start_idx = len(df) - 20000
    if start_idx < 200:
        start_idx = 200

    step_size = 100
    indices = range(start_idx, len(df), step_size)

    print(
        f"Analyzing last {len(df) - start_idx} steps (indices {start_idx} to {len(df)})..."
    )

    for i in tqdm(indices):
        try:
            # detect_regime expects the dataframe and the current integer index
            # We pass the full dataframe but tell it to look at index i
            # Note: detect_regime uses iloc internally, so we pass the integer position

            result = classifier.detect_regime(df, current_index=i)

            metrics = result.metrics

            results.append(
                {
                    "timestamp": df.index[i],
                    "regime": result.primary_regime.name,
                    "confidence": result.confidence,
                    "volatility": metrics.volatility,
                    "trend_strength": metrics.trend_strength,
                    "bull_strength": metrics.bull_strength,
                    "bear_strength": metrics.bear_strength,
                    "price_range_ratio": metrics.price_range_ratio,
                    "adx": metrics.adx,
                    "rsi": metrics.rsi,
                }
            )

        except Exception as e:
            print(f"Error at index {i}: {e}")
            # import traceback
            # traceback.print_exc()

    results_df = pd.DataFrame(results)

    print("\n=== Regime Distribution ===")
    print(results_df["regime"].value_counts())

    print("\n=== Metrics Statistics ===")
    print(results_df.describe())

    print("\n=== Thresholds vs Actuals ===")
    print("Thresholds (from classifier):")
    for k, v in classifier.thresholds.items():
        print(f"  {k}: {v}")

    print("\nActual Volatility Stats:")
    print(results_df["volatility"].describe())

    print("\nActual Trend Strength Stats:")
    print(results_df["trend_strength"].describe())


if __name__ == "__main__":
    analyze_regime_distribution()
