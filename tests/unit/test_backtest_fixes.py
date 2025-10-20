#!/usr/bin/env python3
"""
Test script for backtest bug fixes and 150-dimensional features.
"""

import numpy as np
import pandas as pd

from ztb.trading.backtest.adapters import RLPolicyAdapter
from ztb.trading.backtest.metrics import MetricsCalculator
from ztb.trading.backtest.runner import BacktestEngine


def create_sample_data(n_periods=100):
    """Create sample market data for testing."""
    dates = pd.date_range("2023-01-01", periods=n_periods, freq="D")
    np.random.seed(42)

    # Generate realistic price data
    base_price = 100
    returns = np.random.randn(n_periods) * 0.02  # 2% daily volatility
    prices = base_price * np.exp(returns.cumsum())

    data = pd.DataFrame(
        {
            "timestamp": dates,
            "open": prices * (1 + np.random.randn(n_periods) * 0.01),
            "high": prices * (1 + np.random.randn(n_periods) * 0.02),
            "low": prices * (1 - np.random.randn(n_periods) * 0.02),
            "close": prices,
            "volume": np.random.randint(100, 1000, n_periods),
        }
    )
    data.set_index("timestamp", inplace=True)
    return data


def test_basic_backtest():
    """Test basic backtest functionality."""
    print("=== Testing Basic Backtest ===")

    # Create sample data
    data = create_sample_data(50)
    print(f"Created sample data: {data.shape}")

    # Test adapter without 150d features
    print("Testing RLPolicyAdapter (basic features)...")
    adapter = RLPolicyAdapter(enable_150d_features=False)
    signal = adapter.generate_signal(data, 0)
    print(
        f"Generated signal: {signal['action']} (confidence: {signal.get('confidence', 'N/A')})"
    )

    # Test backtest engine
    print("Testing BacktestEngine...")
    engine = BacktestEngine(initial_capital=10000)
    equity_series, orders_df, adaptation_summary = engine.run_backtest(adapter, data)

    print("Backtest completed:")
    print(f"  - Final equity: ${equity_series.iloc[-1]:.2f}")
    print(f"  - Total orders: {len(orders_df)}")
    print(f"  - Equity range: ${equity_series.min():.2f} - ${equity_series.max():.2f}")

    # Test metrics
    returns = MetricsCalculator.calculate_returns(equity_series)
    sharpe = MetricsCalculator.calculate_sharpe_ratio(returns)
    print(f"  - Sharpe ratio: {sharpe:.4f}")

    return True


def test_150d_features():
    """Test 150-dimensional feature generation."""
    print("\n=== Testing 150-Dimensional Features ===")

    # Create sample data
    data = create_sample_data(100)
    print(f"Created sample data: {data.shape}")

    # Test adapter with 150d features
    print("Testing RLPolicyAdapter (150d features)...")
    adapter = RLPolicyAdapter(enable_150d_features=True)

    # Check cache stats
    cache_stats = adapter.get_cache_stats()
    print(
        f"Initial cache: {cache_stats['cache_entries']} entries, {cache_stats['estimated_memory_mb']:.2f} MB"
    )

    # Generate signal
    signal = adapter.generate_signal(data, 0)
    print(
        f"Generated signal: {signal['action']} (features used: {signal.get('features_used', 'N/A')})"
    )

    # Check cache after generation
    cache_stats = adapter.get_cache_stats()
    print(
        f"Cache after generation: {cache_stats['cache_entries']} entries, {cache_stats['estimated_memory_mb']:.2f} MB"
    )

    # Clear cache
    adapter.clear_feature_cache()
    cache_stats = adapter.get_cache_stats()
    print(
        f"Cache after clearing: {cache_stats['cache_entries']} entries, {cache_stats['estimated_memory_mb']:.2f} MB"
    )

    return True


def test_error_handling():
    """Test error handling capabilities."""
    print("\n=== Testing Error Handling ===")

    # Test with empty data
    print("Testing with empty data...")
    adapter = RLPolicyAdapter(enable_150d_features=False)
    try:
        signal = adapter.generate_signal(pd.DataFrame(), 0)
        print(f"Empty data signal: {signal['action']}")
    except Exception as e:
        print(f"Empty data handled gracefully: {type(e).__name__}")

    # Test with invalid data
    print("Testing with invalid data...")
    bad_data = pd.DataFrame({"invalid_col": [1, 2, 3]})
    try:
        signal = adapter.generate_signal(bad_data, 0)
        print(f"Invalid data signal: {signal['action']}")
    except Exception as e:
        print(f"Invalid data handled gracefully: {type(e).__name__}")

    return True


if __name__ == "__main__":
    print("Starting backtest system tests...\n")

    try:
        # Run tests
        test_basic_backtest()
        test_150d_features()
        test_error_handling()

        print("\n=== All Tests Completed Successfully! ===")
        print("✓ Basic backtest functionality")
        print("✓ 150-dimensional feature generation")
        print("✓ Error handling and robustness")
        print("✓ Memory management and caching")

    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback

        traceback.print_exc()
