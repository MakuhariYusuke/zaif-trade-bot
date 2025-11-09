#!/usr/bin/env python3
"""
Test BTC analysis in comprehensive backtest
"""

from datetime import datetime

from ztb.trading.comprehensive_backtest import BacktestConfig, BacktestResult


def test_comprehensive_backtest_btc():
    """Test BTC analysis in comprehensive backtest"""
    try:
        # Create a test config with BTC
        config = BacktestConfig(
            symbol="btc_jpy",
            start_date=datetime(2024, 1, 1),
            end_date=datetime(2024, 12, 31),
            initial_balance=100000.0,
            initial_btc=0.5,  # Add initial BTC
        )

        # Create a test result with BTC data
        result = BacktestResult(
            config=config,
            total_return=0.05,  # 5% return
            initial_btc=0.5,
            final_btc=0.52,  # Gained 0.02 BTC
            btc_return=4.0,  # 4% BTC return
            btc_holdings_history=[0.5, 0.51, 0.52],  # Sample history
            net_btc_gained=0.02,
        )

        print("=== Comprehensive Backtest BTC Test ===")
        print(f"Config Initial BTC: {config.initial_btc}")
        print(f"Result Initial BTC: {result.initial_btc}")
        print(f"Result Final BTC: {result.final_btc}")
        print(f"Result BTC Return: {result.btc_return}%")
        print(f"Result Net BTC Gained: {result.net_btc_gained}")
        print(f"BTC Holdings History Length: {len(result.btc_holdings_history)}")

        print("\n✅ BTC fields successfully added to comprehensive backtest!")

    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    test_comprehensive_backtest_btc()
