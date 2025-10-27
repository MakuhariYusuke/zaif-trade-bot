#!/usr/bin/env python3
"""
SAC v435 Scalping Analysis Script
"""

import json
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from sac_v435_detailed_analysis import SACv435Analyzer


def main():
    print("🚀 SAC v435 Scalping Analysis")
    print("=" * 50)

    # Load existing backtest results
    results_file = "backtest_results_v435.json"
    with open(results_file, "r") as f:
        data = json.load(f)

    print(f"✅ Loaded backtest data from {results_file}")

    # Create analyzer instance
    analyzer = SACv435Analyzer(results_file)

    # Run comprehensive analysis
    print("\n📊 Running P-Average Analysis...")
    try:
        p_avg_results = analyzer.calculate_p_average_returns()
        print("✅ P-Average analysis completed")
        print(json.dumps(p_avg_results, indent=2, default=str))
    except Exception as e:
        print(f"❌ P-Average analysis failed: {e}")

    print("\n📊 Running Trading Interval Analysis...")
    try:
        interval_results = analyzer.analyze_trading_intervals()
        print("✅ Trading interval analysis completed")
        print(json.dumps(interval_results, indent=2, default=str))
    except Exception as e:
        print(f"❌ Trading interval analysis failed: {e}")

    print("\n📊 Running Risk Metrics Analysis...")
    try:
        risk_results = analyzer.calculate_risk_metrics()
        print("✅ Risk metrics analysis completed")
        print(json.dumps(risk_results, indent=2, default=str))
    except Exception as e:
        print(f"❌ Risk metrics analysis failed: {e}")

    print("\n📊 Running Statistical Comparison...")
    try:
        stat_results = analyzer.perform_statistical_comparison()
        print("✅ Statistical comparison completed")
        print(json.dumps(stat_results, indent=2, default=str))
    except Exception as e:
        print(f"❌ Statistical comparison failed: {e}")


if __name__ == "__main__":
    main()
