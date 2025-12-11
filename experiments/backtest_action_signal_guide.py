#!/usr/bin/env python3
"""
Action Signal Guide Backtest Script

Tests the Action Signal Guide strategy in isolation to evaluate its profitability.
"""

import json
import sys
from datetime import datetime
from pathlib import Path

# Add project root to path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

import pandas as pd

from ztb.trading.backtest.adapters import ActionSignalGuideAdapter
from ztb.trading.backtest.metrics import MetricsCalculator
from ztb.trading.backtest.report import ReportGenerator
from ztb.trading.backtest.runner import BacktestEngine


def load_sample_data() -> pd.DataFrame:
    """Load sample OHLCV data for backtesting."""
    # Try to load from existing data files
    data_paths = [
        "data/sample_ohlcv.csv",
        "tests/test_synthetic_dataset.csv",
        "test_synthetic_dataset.csv",
    ]

    for path in data_paths:
        if Path(path).exists():
            print(f"Loading data from {path}")
            df = pd.read_csv(path)
            # Ensure required columns exist
            required_cols = ["timestamp", "open", "high", "low", "close", "volume"]
            if all(col in df.columns for col in required_cols):
                # Convert timestamp if needed
                if "timestamp" in df.columns:
                    df["timestamp"] = pd.to_datetime(df["timestamp"])
                    df.set_index("timestamp", inplace=True)
                return df

    # Generate synthetic data if no data file found
    print("No data file found, generating synthetic data...")
    np.random.seed(42)
    dates = pd.date_range(
        "2023-01-01", periods=5000, freq="h"
    )  # Increased from 1000 to 5000

    # Generate realistic price series
    base_price = 50000.0
    prices = []
    for i in range(len(dates)):
        trend = 0.00005 * i  # Reduced trend for more realistic data
        noise = np.random.normal(0, 0.015)  # Reduced volatility
        seasonal = 0.01 * np.sin(2 * np.pi * i / 24)  # Daily seasonality
        price = base_price * (1 + trend + noise + seasonal)
        prices.append(max(price, 1000.0))  # Ensure positive prices

    # Create OHLCV data
    data = []
    for i, price in enumerate(prices):
        high = price * (1 + abs(np.random.normal(0, 0.01)))
        low = price * (1 - abs(np.random.normal(0, 0.01)))
        open_price = prices[i - 1] if i > 0 else price
        close = price
        volume = np.random.randint(100, 1000)

        data.append(
            {
                "timestamp": dates[i],
                "open": open_price,
                "high": high,
                "low": low,
                "close": close,
                "volume": volume,
            }
        )

    df = pd.DataFrame(data)
    df.set_index("timestamp", inplace=True)
    return df


def run_action_signal_guide_backtest():
    """Run backtest for Action Signal Guide strategy."""
    print("=== Action Signal Guide Backtest ===")
    print("Testing Action Signal Guide strategy profitability...")

    # Load data
    data = load_sample_data()
    print(f"Loaded {len(data)} data points")

    # Initialize components
    adapter = ActionSignalGuideAdapter()
    # Adjust hyperparameters for more signals
    adapter.update_hyperparameters(
        {
            "confidence_threshold": 0.3,  # Lower threshold from 0.5 to 0.3
            "signal_strength_threshold": 0.1,  # Lower threshold from 0.3 to 0.1
            "max_signals_per_bar": 5,  # Increased from 3 to 5
            "force_accept_signals": True,  # Diagnostic: force acceptance of signals
        }
    )
    engine = BacktestEngine(
        initial_capital=100000.0,  # Increased capital from 10k to 100k
        slippage_bps=5.0,
        commission_bps=0.1,
        enable_risk=True,  # Enable risk management
        risk_profile="conservative",  # Use conservative risk profile
    )

    metrics_calculator = MetricsCalculator()
    report_generator = ReportGenerator()

    # Run backtest
    print("Running backtest...")
    start_time = datetime.now()

    # For debugging, limit to first 50 steps
    debug_limit = None  # Remove debug limit for full backtest
    print("Running full backtest (no debug limit)")

    results = engine.run_backtest(
        strategy=adapter, data=data if debug_limit is None else data[:debug_limit]
    )

    equity_curve, orders, adaptation_history, signal_performance = results

    end_time = datetime.now()
    duration = end_time - start_time
    print(f"Backtest completed in {duration}")

    # Calculate metrics
    metrics = metrics_calculator.calculate_all_metrics(
        equity_curve=equity_curve,
        orders=orders,
        initial_capital=engine.initial_capital,
        risk_free_rate=0.02,
        slippage_bps=engine.slippage_bps,
    )
    print("\n=== BACKTEST RESULTS ===")
    print(f"Initial Capital: ${engine.initial_capital:,.2f}")
    print(f"Final Portfolio Value: ${equity_curve.iloc[-1]:,.2f}")
    print(f"Total Return: {metrics.total_return * 100:.2f}%")
    print(f"Annualized Return: {metrics.annualized_return * 100:.2f}%")
    print(f"CAGR: {metrics.cagr * 100:.2f}%")
    print(f"Total Trades: {metrics.total_trades}")
    print(f"Win Rate: {metrics.win_rate * 100:.2f}%")
    print(f"Sharpe Ratio: {metrics.sharpe_ratio:.4f}")
    print(f"Sortino Ratio: {metrics.sortino_ratio:.4f}")
    print(f"Calmar Ratio: {metrics.calmar_ratio:.4f}")
    print(f"Max Drawdown: {metrics.max_drawdown * 100:.2f}%")
    print(f"Volatility: {metrics.volatility * 100:.2f}%")
    print(f"Profit Factor: {metrics.profit_factor:.2f}")
    print(f"Avg Win: ${metrics.avg_win:.2f}")
    print(f"Avg Loss: ${metrics.avg_loss:.2f}")
    print(f"Turnover: {metrics.turnover:.2f}")

    # Generate report
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_path = f"backtest_results/action_signal_guide_backtest_{timestamp}.json"

    report_data = {
        "backtest_type": "action_signal_guide_only",
        "timestamp": timestamp,
        "duration_seconds": duration.total_seconds(),
        "data_points": len(data),
        "initial_capital": engine.initial_capital,
        "final_portfolio_value": float(equity_curve.iloc[-1]),
        "metrics": {
            "total_return_pct": metrics.total_return * 100,
            "annualized_return_pct": metrics.annualized_return * 100,
            "cagr_pct": metrics.cagr * 100,
            "total_trades": metrics.total_trades,
            "win_rate_pct": metrics.win_rate * 100,
            "sharpe_ratio": metrics.sharpe_ratio,
            "sortino_ratio": metrics.sortino_ratio,
            "calmar_ratio": metrics.calmar_ratio,
            "max_drawdown_pct": metrics.max_drawdown * 100,
            "volatility_pct": metrics.volatility * 100,
            "profit_factor": metrics.profit_factor,
            "avg_win": metrics.avg_win,
            "avg_loss": metrics.avg_loss,
            "turnover": metrics.turnover,
            "estimated_slippage_bps": metrics.estimated_slippage_bps,
        },
        "config": {
            "slippage_bps": engine.slippage_bps,
            "commission_bps": engine.commission_bps,
            "strategy_hyperparameters": adapter.hyperparameters,
        },
    }

    # Save report
    Path("backtest_results").mkdir(exist_ok=True)
    with open(report_path, "w") as f:
        json.dump(report_data, f, indent=2, default=str)

    print(f"\nReport saved to: {report_path}")

    # Summary
    total_return_pct = metrics.total_return * 100
    if total_return_pct > 0:
        print(f"✅ Action Signal Guide shows POSITIVE returns: {total_return_pct:.2f}%")
        print("The strategy appears profitable on its own!")
    else:
        print(f"❌ Action Signal Guide shows NEGATIVE returns: {total_return_pct:.2f}%")
        print("The strategy may need tuning or is not profitable in isolation.")

    return metrics


if __name__ == "__main__":
    import numpy as np

    try:
        results = run_action_signal_guide_backtest()
    except Exception as e:
        print(f"Error running backtest: {e}")
        import traceback

        traceback.print_exc()
