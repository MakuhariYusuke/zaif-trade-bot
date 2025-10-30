#!/usr/bin/env python3
"""
Action Signal Guide Backtest Script

This script performs backtesting of the ActionSignalGuide strategy in isolation
to evaluate its profitability and signal generation capabilities.
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from backtest.config import get_backtest_config, get_engine_config
from backtest.data_generator import generate_synthetic_data
from backtest.results_runner import display_backtest_results, display_signal_statistics
from ztb.trading.backtest.adapters import ActionSignalGuideAdapter
from ztb.trading.backtest.metrics import MetricsCalculator
from ztb.trading.backtest.runner import BacktestEngine


def run_action_signal_guide_backtest():
    """Run the ActionSignalGuide backtest."""
    print("=== Action Signal Guide Backtest ===")
    print("Generating synthetic data...")

    # Generate data
    data = generate_synthetic_data(5000)
    print(f"Generated {len(data)} data points")

    # Configure ActionSignalGuide
    config = get_backtest_config()

    # Create adapter
    adapter = ActionSignalGuideAdapter(config=config)

    # Backtest configuration
    backtest_config = get_engine_config()

    print("Running backtest...")
    print(f"Initial capital: ${backtest_config['initial_capital']:,.2f}")
    print(f"Data period: {data.index[0]} to {data.index[-1]}")
    print(f"Total bars: {len(data)}")

    # Run backtest
    try:
        engine = BacktestEngine(
            initial_capital=backtest_config["initial_capital"],
            commission_bps=backtest_config["commission"] * 100,  # Convert to bps
            enable_risk=backtest_config["enable_risk_management"],
            max_position_size=backtest_config["max_position_size"],
        )
        results = engine.run_backtest(strategy=adapter, data=data)

        equity_curve, orders, adaptation_history = results

        # Calculate metrics
        metrics_calculator = MetricsCalculator()
        performance_metrics = metrics_calculator.calculate_all_metrics(
            equity_curve=equity_curve,
            orders=orders,
            initial_capital=engine.initial_capital,
            risk_free_rate=0.02,
            slippage_bps=engine.slippage_bps,
        )

        # Convert to dict for display
        performance_metrics_dict = {
            "final_capital": equity_curve.iloc[-1],
            "total_return": performance_metrics.total_return,
            "annual_return": performance_metrics.annualized_return,
            "max_drawdown": performance_metrics.max_drawdown,
            "sharpe_ratio": performance_metrics.sharpe_ratio,
            "win_rate": performance_metrics.win_rate,
            "total_trades": performance_metrics.total_trades,
            "profit_factor": performance_metrics.profit_factor,
        }

        # Display results
        display_backtest_results(performance_metrics_dict)

        # Signal statistics
        display_signal_statistics(adapter)

        return results, performance_metrics

    except Exception as e:
        print(f"Backtest failed: {e}")
        import traceback

        traceback.print_exc()
        return None, None


if __name__ == "__main__":
    results, metrics = run_action_signal_guide_backtest()
