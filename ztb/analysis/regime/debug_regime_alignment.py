#!/usr/bin/env python3
"""
Debug script for regime analysis alignment.
"""

from pathlib import Path

from ztb.io.data_loader import DataLoader

from ztb.io.json_io import read_json
from ztb.analysis.comparative.regime_performance_analyzer import (
    RegimePerformanceAnalyzer,
)
from ztb.analysis.specialized.market.market_regime_classifier import (
    MarketRegimeClassifier,
)

def main():
    # Load backtest results
    backtest_file = (
        Path("optimization_results")
        / "20251020_171431"
        / "extended_backtest_results.json"
    )
    backtest_results = read_json(backtest_file)

    # Load market data
    market_data = DataLoader.load_csv_strict("data/btc_jpy_extended_dataset.csv")

    # Filter data by date range (same as config)
    market_data["timestamp"] = pd.to_datetime(market_data["timestamp"])
    start_date = pd.to_datetime("2022-01-01")
    end_date = pd.to_datetime("2024-01-04")
    market_data = market_data[
        (market_data["timestamp"] >= start_date)
        & (market_data["timestamp"] <= end_date)
    ]

    print(f"Loaded {len(market_data)} market data points after filtering")

    # Create classifier and analyzer
    classifier = MarketRegimeClassifier()
    analyzer = RegimePerformanceAnalyzer()

    print(f"Loaded {len(backtest_results.get('trades', []))} trades")

    # Classify market conditions
    market_conditions = classifier.classify_market_conditions(market_data)
    print(f"Classified {len(market_conditions)} market conditions")

    if market_conditions:
        print(f"First condition timestamp: {market_conditions[0].timestamp}")
        print(f"Last condition timestamp: {market_conditions[-1].timestamp}")

    if backtest_results.get("trades"):
        first_trade = backtest_results["trades"][0]
        print(f"First trade timestamp: {first_trade.get('timestamp')}")

    # Try alignment
    aligned_data = analyzer._align_backtest_with_conditions(
        backtest_results, market_conditions
    )
    print(f"Aligned {len(aligned_data)} trades with conditions")

if __name__ == "__main__":
    main()
