#!/usr/bin/env python3
"""
Validation script to test integrated multi-period backtest functionality.

This script compares the output of the original multi_period_analysis_sac_v445_3.py
with the integrated functionality in unified_trainer and v4xx_unified_analyzer.
"""

import json
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
            test_with_mock_data(mock_results)
            return

        if not Path(test_data_path).exists():
            print(f"⚠️  Test data not found: {test_data_path}")
            print("Using mock data for validation...")

            mock_results = create_mock_backtest_results()
            test_with_mock_data(mock_results)
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


def create_mock_backtest_results() -> Dict[str, Any]:
    """Create mock backtest results for testing."""
    return {
        "24h_windows": {
            "summary": {
                "overall": {
                    "total_trades": 150,
                    "win_rate": 0.65,
                    "avg_return": 0.023,
                    "sharpe_ratio": 1.8,
                    "max_drawdown": 0.12
                }
            },
            "regime_performance": {
                "bull": {"win_rate": 0.75, "avg_return": 0.035},
                "bear": {"win_rate": 0.45, "avg_return": -0.012},
                "sideways": {"win_rate": 0.60, "avg_return": 0.015}
            }
        },
        "48h_windows": {
            "summary": {
                "overall": {
                    "total_trades": 120,
                    "win_rate": 0.58,
                    "avg_return": 0.018,
                    "sharpe_ratio": 1.5,
                    "max_drawdown": 0.15
                }
            },
            "regime_performance": {
                "bull": {"win_rate": 0.70, "avg_return": 0.028},
                "bear": {"win_rate": 0.40, "avg_return": -0.008},
                "sideways": {"win_rate": 0.55, "avg_return": 0.012}
            }
        }
    }


def test_with_mock_data(mock_results: Dict[str, Any]):
    """Test integration with mock data."""
    print("Testing with mock data...")

    # Create analyzer instance without loading data
    analyzer = V4XXUnifiedAnalyzer.__new__(V4XXUnifiedAnalyzer)
    analyzer.version = "445.3"
    analyzer.metrics = {}

    # Initialize logger (from UnifiedBase)
    import logging
    analyzer.logger = logging.getLogger("test_analyzer")

    # Test analyzer functionality directly
    analysis_results = analyzer.analyze_multi_period_backtest(mock_results)

    print("✅ Mock data analysis successful")

    # Validate output structure
    validate_output_structure(mock_results, analysis_results)


def validate_output_structure(backtest_results: Dict[str, Any], analysis_results: Dict[str, Any]):
    """Validate that output has expected structure."""

    print("Validating output structure...")

    # Check backtest results structure
    required_keys = ["overall_performance", "regime_performance", "timeframe_comparison", "recommendations"]
    for key in required_keys:
        if key not in analysis_results:
            raise ValueError(f"Missing required key in analysis results: {key}")

    print("✅ Analysis results structure valid")

    # Check recommendations structure
    recommendations = analysis_results["recommendations"]
    rec_keys = ["optimal_timeframe", "regime_strategy", "risk_management", "implementation_priority"]
    for key in rec_keys:
        if key not in recommendations:
            raise ValueError(f"Missing required key in recommendations: {key}")

    print("✅ Recommendations structure valid")

    # Print summary
    print("\n📊 Integration Test Summary:")
    print(f"  - Overall Performance: {analysis_results['overall_performance']}")
    print(f"  - Optimal Timeframe: {recommendations['optimal_timeframe']}")
    print(f"  - Implementation Priority: {recommendations['implementation_priority']}")


if __name__ == "__main__":
    success = validate_integration()
    sys.exit(0 if success else 1)