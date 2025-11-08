#!/usr/bin/env python3
"""
Test script for refactored analysis functions
"""

import numpy as np
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).resolve().parent
sys.path.insert(0, str(project_root))

def test_analyze_backtest_detailed():
    """Test analyze_backtest_detailed.py refactoring"""
    try:
        from ztb.analysis.analyze_backtest_detailed import analyze_portfolio_performance

        # Create test data
        values = np.cumprod(1 + np.random.normal(0.001, 0.02, 100))
        result = analyze_portfolio_performance(values)

        print('✓ analyze_backtest_detailed.py refactoring test:')
        print(f'  Sharpe ratio: {result["sharpe_ratio"]:.4f}')
        print(f'  Max drawdown: {result["max_drawdown_pct"]:.2f}%')
        return True
    except Exception as e:
        print(f'✗ analyze_backtest_detailed.py test failed: {e}')
        return False

def test_advanced_sac_analysis():
    """Test advanced_sac_v434_1_analysis.py refactoring"""
    try:
        # This would require loading actual backtest data, so just test import
        from ztb.analysis.advanced_sac_v434_1_analysis import sharpe_ratio
        print('✓ advanced_sac_v434_1_analysis.py import test passed')
        return True
    except Exception as e:
        print(f'✗ advanced_sac_v434_1_analysis.py test failed: {e}')
        return False

def test_auto_feature_generator():
    """Test auto_feature_generator.py refactoring"""
    try:
        from ztb.analysis.auto_feature_generator import AutoFeatureGenerator

        # Create test returns
        returns = np.random.normal(0.001, 0.02, 50)
        gen = AutoFeatureGenerator()
        metrics = gen._calculate_basic_metrics(returns)

        print('✓ auto_feature_generator.py refactoring test:')
        print(f'  Sharpe ratio: {metrics["sharpe"]:.4f}')
        print(f'  Max drawdown: {metrics["max_drawdown"]:.4f}')
        return True
    except Exception as e:
        print(f'✗ auto_feature_generator.py test failed: {e}')
        return False

if __name__ == "__main__":
    print("Testing refactored analysis functions...")
    print()

    results = []
    results.append(test_analyze_backtest_detailed())
    results.append(test_advanced_sac_analysis())
    results.append(test_auto_feature_generator())

    print()
    print(f"Results: {sum(results)}/{len(results)} tests passed")

    if all(results):
        print("🎉 All refactoring tests passed!")
    else:
        print("❌ Some tests failed")
        sys.exit(1)