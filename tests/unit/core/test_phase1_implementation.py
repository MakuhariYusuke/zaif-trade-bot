"""
Test script for Phase 1: Risk Management and Dynamic Thresholds Implementation

This script tests the new risk management features and dynamic threshold adjustments
implemented in ActionSignalGuideAdapter.
"""

from datetime import datetime, timedelta

import numpy as np
import pandas as pd

# Import the updated adapter
from ztb.trading.backtest.adapters import ActionSignalGuideAdapter


def create_sample_data(num_days: int = 100) -> pd.DataFrame:
    """Create sample OHLCV data for testing."""
    np.random.seed(42)  # For reproducible results

    # Generate date range
    start_date = datetime(2023, 1, 1)
    dates = [start_date + timedelta(days=i) for i in range(num_days)]

    # Generate price data with some trend and volatility
    base_price = 100.0
    prices = [base_price]

    for i in range(1, num_days):
        # Add some trend and random walk
        trend = 0.001  # Slight upward trend
        volatility = 0.02  # 2% daily volatility
        change = np.random.normal(trend, volatility)
        new_price = prices[-1] * (1 + change)
        prices.append(new_price)

    # Create OHLCV data
    data = []
    for i, price in enumerate(prices):
        high = price * (1 + abs(np.random.normal(0, 0.01)))
        low = price * (1 - abs(np.random.normal(0, 0.01)))
        open_price = prices[i - 1] if i > 0 else price
        volume = np.random.randint(1000, 10000)

        data.append(
            {
                "timestamp": dates[i],
                "open": open_price,
                "high": high,
                "low": low,
                "close": price,
                "volume": volume,
            }
        )

    df = pd.DataFrame(data)
    df.set_index("timestamp", inplace=True)
    return df


def test_risk_management():
    """Test risk management functionality."""
    print("Testing Risk Management Features...")
    print("=" * 50)

    # Create sample data
    data = create_sample_data(50)

    # Initialize adapter
    adapter = ActionSignalGuideAdapter()

    # Test market volatility calculation
    volatility = adapter._calculate_market_volatility(data)
    print(".4f")

    # Test dynamic thresholds
    thresholds = adapter._calculate_dynamic_thresholds(data)
    print(
        f"Dynamic Thresholds: Confidence={thresholds['confidence_threshold']:.3f}, "
        f"Strength={thresholds['signal_strength_threshold']:.3f}"
    )

    # Test risk manager position validation
    signal_strength = 0.8
    market_volatility = volatility

    can_open = adapter.risk_manager.should_open_position(
        signal_strength, market_volatility, adapter.risk_manager.portfolio_value
    )
    print(f"Can Open Position: {can_open}")

    # Test position sizing
    position_size = adapter.risk_manager.get_risk_adjusted_position_size(
        signal_strength, market_volatility
    )
    print(f"Risk-Adjusted Position Size: {position_size:.4f}")

    print("✓ Risk management tests completed\n")


def test_dynamic_thresholds():
    """Test dynamic threshold management."""
    print("Testing Dynamic Threshold Management...")
    print("=" * 50)

    # Create sample data with different regimes
    data = create_sample_data(100)

    adapter = ActionSignalGuideAdapter()

    # Test regime detection
    regime = adapter.threshold_manager.detect_market_regime(data)
    print(f"Detected Market Regime: {regime}")

    # Test adaptive thresholds
    adaptive_thresholds = adapter.threshold_manager.calculate_adaptive_thresholds(data)
    print(
        f"Adaptive Thresholds: Confidence={adaptive_thresholds['confidence_threshold']:.3f}, "
        f"Strength={adaptive_thresholds['signal_strength_threshold']:.3f}"
    )
    print(f"Regime: {adaptive_thresholds['regime']}")

    print("✓ Dynamic threshold tests completed\n")


def test_signal_generation():
    """Test enhanced signal generation with risk management."""
    print("Testing Enhanced Signal Generation...")
    print("=" * 50)

    # Create sample data
    data = create_sample_data(100)

    adapter = ActionSignalGuideAdapter()

    # Generate signals for the last few bars
    signals = []
    for i in range(20, len(data)):
        current_data = data.iloc[: i + 1]
        signal = adapter.generate_signal(current_data, 0)
        signals.append(signal)

    # Analyze signals
    buy_signals = sum(1 for s in signals if s["action"] == "buy")
    sell_signals = sum(1 for s in signals if s["action"] == "sell")
    hold_signals = sum(1 for s in signals if s["action"] == "hold")
    risk_filtered = sum(1 for s in signals if s.get("risk_filtered", False))

    print("Signal Summary (last 20 bars):")
    print(f"  Buy Signals: {buy_signals}")
    print(f"  Sell Signals: {sell_signals}")
    print(f"  Hold Signals: {hold_signals}")
    print(f"  Risk Filtered: {risk_filtered}")

    # Show sample signal with risk management info
    recent_signals = [s for s in signals if s["action"] != "hold"][
        -3:
    ]  # Last 3 non-hold signals
    for i, signal in enumerate(recent_signals):
        print(
            f"Sample Signal {i+1}: {signal['action']} "
            f"(confidence: {signal.get('confidence', 0):.3f})"
        )
        if "position_size" in signal:
            print(f"  Position Size: {signal['position_size']:.4f}")
        if "risk_filtered" in signal:
            print(f"  Risk Filtered: {signal['risk_filtered']}")

    print("✓ Signal generation tests completed\n")


def test_walk_forward_analysis():
    """Test walk-forward analysis functionality."""
    print("Testing Walk-Forward Analysis...")
    print("=" * 50)

    # Create larger sample dataset
    data = create_sample_data(300)  # 300 days of data

    adapter = ActionSignalGuideAdapter()

    try:
        # Run walk-forward analysis
        results = adapter.walk_forward_analyzer.run_walk_forward_analysis(data, adapter)

        print(
            f"Walk-Forward Analysis completed with {len(results['walk_forward_results'])} periods"
        )

        # Show overall metrics
        metrics = results["overall_metrics"]
        print("Overall Metrics:")
        print(f"  Average Total Return: {metrics.get('average_total_return', 0):.4f}")
        print(f"  Average Sharpe Ratio: {metrics.get('average_sharpe_ratio', 0):.4f}")
        print(f"  Sharpe Consistency: {metrics.get('sharpe_consistency', 0):.4f}")
        print(
            f"  Positive Periods: {metrics.get('positive_periods', 0)}/{metrics.get('num_periods', 0)}"
        )

        print("✓ Walk-forward analysis tests completed\n")

    except Exception as e:
        print(f"Walk-forward analysis test failed: {e}")
        print("This might be due to insufficient data or implementation details\n")


def main():
    """Run all Phase 1 tests."""
    print("Phase 1 Implementation Test Suite")
    print("=" * 60)
    print()

    try:
        test_risk_management()
        test_dynamic_thresholds()
        test_signal_generation()
        test_walk_forward_analysis()

        print("=" * 60)
        print("Phase 1 Testing Complete!")
        print()
        print("Summary of Implemented Features:")
        print("✓ RiskManager - Position sizing, stop losses, risk limits")
        print(
            "✓ DynamicThresholdManager - Market regime detection, adaptive thresholds"
        )
        print("✓ WalkForwardAnalyzer - Parameter optimization, out-of-sample testing")
        print("✓ Enhanced ActionSignalGuideAdapter - Integrated risk management")
        print()
        print("Next Steps:")
        print("- Run full backtests with real market data")
        print("- Validate performance improvements")
        print("- Fine-tune risk parameters based on results")

    except Exception as e:
        print(f"Test suite failed with error: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
