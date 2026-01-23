#!/usr/bin/env python3
"""
Test BTC analysis in comprehensive report
"""

import json

from ztb.analysis.comparative.analyze_backtest import BacktestAnalyzer


def test_btc_in_report():
    """Test BTC analysis in comprehensive report"""
    try:
        # Load the comparison results and extract one model
        with open("results/v445_comparison_backtest_20251109_161601.json", "r") as f:
            comparison_data = json.load(f)

        # Use v445.4 model data for testing
        model_data = comparison_data["detailed_results"]["v445.4_ultra_aggressive"]

        # Create analyzer with individual model data
        analyzer = BacktestAnalyzer.__new__(
            BacktestAnalyzer
        )  # Create without calling __init__
        analyzer.data = model_data
        analyzer._validate_data = lambda: None  # Skip validation for this test

        # Mock required attributes
        class MockMonitor:
            def __enter__(self):
                return self

            def __exit__(self, *args):
                pass

        analyzer.performance_monitor = MockMonitor()
        analyzer.results_path = type("MockPath", (), {"name": "test_model"})()
        analyzer.training_data = None

        # Generate comprehensive report
        report = analyzer.generate_comprehensive_report()

        print("=== Comprehensive Report BTC Analysis Test ===")

        if "BTCパフォーマンス分析" in report:
            print("✅ BTC analysis section found in comprehensive report!")

            # Extract BTC section
            btc_section_start = report.find("=== BTCパフォーマンス分析 ===")
            if btc_section_start != -1:
                btc_section_end = report.find("\n\n", btc_section_start)
                if btc_section_end == -1:
                    btc_section_end = len(report)
                btc_section = report[btc_section_start:btc_section_end]
                print("\nBTC Analysis Section:")
                print(btc_section)
        else:
            print("❌ BTC analysis section not found in comprehensive report")
            print("Report preview:")
            print(report[:1000] + "..." if len(report) > 1000 else report)

    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    test_btc_in_report()
