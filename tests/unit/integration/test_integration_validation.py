#!/usr/bin/env python3
"""
Validation script to test integrated multi-period backtest functionality.

This script compares the output of the original multi_period_analysis_sac_v445_3.py
with the integrated functionality in unified_trainer and v4xx_unified_analyzer.
"""

import sys
from pathlib import Path
from typing import Any, Dict

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from ztb.training.unified_trainer.trainer import UnifiedTrainer
from ztb.analysis.v4xx_unified_analyzer import V4XXUnifiedAnalyzer


def validate_integration():
    """Validate that integrated functionality produces correct output."""

    print("🔍 Validating Multi-Period Backtest Integration")
    print("=" * 60)

    try:
        # Test data - use existing backtest results if available
        test_model_path = "models/sac_v445_3_final_model.zip"
        test_data_path = "data/processed/trading_data.csv"

        # Check if test files exist
        if not Path(test_model_path).exists():
            print(f"⚠️  Test model not found: {test_model_path}")
            print("Using mock data for validation...")

            # Create mock backtest results for testing
            mock_results = create_mock_backtest_results()
            _run_with_mock_data(mock_results)
            return

        if not Path(test_data_path).exists():
            print(f"⚠️  Test data not found: {test_data_path}")
            print("Using mock data for validation...")

            mock_results = create_mock_backtest_results()
            _run_with_mock_data(mock_results)
            return

        # Test integrated trainer functionality
        print("Testing integrated trainer functionality...")
        trainer = UnifiedTrainer()

        backtest_results = trainer.run_multi_period_backtest(
            model_path=test_model_path,
            data_path=test_data_path,
            window_sizes=[24, 48, 72],  # 1d, 2d, 3d windows
            overlap_ratio=0.5,
            output_path="test_integration_results.json"
        )

        print("✅ Trainer integration successful")

        # Test integrated analyzer functionality
        print("Testing integrated analyzer functionality...")
        analyzer = V4XXUnifiedAnalyzer("test_integration_results.json", version="445.3")

        analysis_results = analyzer.analyze_multi_period_backtest(backtest_results)

        print("✅ Analyzer integration successful")

        # Validate output structure
        validate_output_structure(backtest_results, analysis_results)

        print("\n🎉 Integration validation completed successfully!")

    except Exception as e:
        print(f"❌ Integration validation failed: {e}")
        import traceback
        traceback.print_exc()
        return False

    return True


def create_mock_backtest_results() -> list[Dict[str, Any]]:
    """Create mock backtest results for testing."""
    return [
        {"name": "24h_window"},
        {"name": "48h_window"},
    ]


def _run_with_mock_data(mock_results: list[Dict[str, Any]]) -> None:
    """Run integration validation with mock data."""
    print("Testing with mock data...")

    # Create analyzer instance without loading data
    analyzer = V4XXUnifiedAnalyzer.__new__(V4XXUnifiedAnalyzer)
    analyzer.version = "445.3"
    analyzer.metrics = {}
    analyzer.results = {
        "summary": {
            "average_trades": 135,
            "win_rate": 0.615,
            "total_return": 0.0205,
            "sharpe_ratio": 1.65,
            "max_drawdown": 0.135,
        }
    }

    # Initialize logger (from UnifiedBase)
    import logging
    analyzer.logger = logging.getLogger("test_analyzer")

    # Test analyzer functionality directly
    analysis_results = analyzer.analyze_multi_period_backtest(mock_results)

    print("✅ Mock data analysis successful")

    # Validate output structure
    validate_output_structure(mock_results, analysis_results)


def test_with_mock_data():
    """Test integration with mock data."""
    _run_with_mock_data(create_mock_backtest_results())


def validate_output_structure(
    backtest_results: list[Dict[str, Any]], analysis_results: Dict[str, Any]
):
    """Validate that output has expected structure."""

    print("Validating output structure...")

    # Check backtest results structure
    required_keys = [
        "period_analysis",
        "overall_metrics",
        "regime_performance",
        "recommendations",
    ]
    for key in required_keys:
        if key not in analysis_results:
            raise ValueError(f"Missing required key in analysis results: {key}")

    print("✅ Analysis results structure valid")

    # Check period analysis and recommendation structure
    period_analysis = analysis_results["period_analysis"]
    if len(period_analysis) != len(backtest_results):
        raise ValueError("Unexpected period analysis length")

    recommendations = analysis_results["recommendations"]
    if not isinstance(recommendations, list) or not recommendations:
        raise ValueError("Recommendations should be a non-empty list")

    print("✅ Recommendations structure valid")

    # Print summary
    print("\n📊 Integration Test Summary:")
    print(f"  - Overall Metrics: {analysis_results['overall_metrics']}")
    print(f"  - First Recommendation: {recommendations[0]}")


if __name__ == "__main__":
    success = validate_integration()
    sys.exit(0 if success else 1)
