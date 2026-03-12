#!/usr/bin/env python3
"""
Action Signal Guide Backtest Script

This script performs backtesting of the ActionSignalGuide strategy in isolation
to evaluate its profitability and signal generation capabilities.
"""

import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

# Add project root to path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from backtest.config import get_backtest_config, get_engine_config
from backtest.data_generator import generate_synthetic_data
from backtest.results_runner import display_backtest_results, display_signal_statistics
from ztb.trading.backtest.adapters import ActionSignalGuideAdapter
from ztb.metrics.metrics import MetricsCalculator


class SimpleBacktestEngine:
    """Simple backtest engine for Action Signal Guide testing."""

    def __init__(
        self, initial_capital: float = 10000.0, commission: float = 0.001
    ) -> None:
        self.initial_capital = initial_capital
        self.commission = commission
        self.capital = initial_capital
        self.position = 0  # -1 (short), 0 (flat), 1 (long)
        self.entry_price = 0.0
        self.trades: List[Dict[str, Any]] = []

    def run_backtest(
        self, strategy_adapter: "ActionSignalGuideAdapter", data: pd.DataFrame
    ) -> Tuple[pd.Series, pd.DataFrame]:
        """Run backtest with strategy adapter."""
        equity_curve = [self.initial_capital]
        orders = []

        # Pre-compute all signals for efficiency
        print("Pre-computing signals for backtest...")
        all_signals = strategy_adapter.generate_signals_batch(data)
        print(f"Pre-computed {len(all_signals)} signals")

        for i in range(len(data)):
            row = data.iloc[i]

            # Get pre-computed signal
            signal = all_signals[i]

            # Execute trade if signal
            if signal["action"] in ["buy", "sell"]:
                price = row["close"]

                # Calculate commission
                commission_amount = price * self.commission

                if signal["action"] == "buy" and self.position <= 0:
                    # Buy signal - go long
                    if self.position < 0:  # Close short position first
                        # Calculate P&L for closing short
                        pnl = self.entry_price - price
                        pnl_amount = (
                            pnl * abs(self.position) * price
                        )  # Simplified position sizing
                        self.capital += pnl_amount - commission_amount

                        # Record closing trade
                        orders.append(
                            {
                                "timestamp": row.name,
                                "action": "close_short",
                                "price": price,
                                "pnl": pnl_amount,
                                "capital_after": self.capital,
                            }
                        )

                    # Open long position
                    self.position = 1
                    self.entry_price = price
                    self.capital -= commission_amount

                    orders.append(
                        {
                            "timestamp": row.name,
                            "action": "buy",
                            "price": price,
                            "capital_after": self.capital,
                        }
                    )

                elif signal["action"] == "sell" and self.position >= 0:
                    # Sell signal - go short
                    if self.position > 0:  # Close long position first
                        # Calculate P&L for closing long
                        pnl = price - self.entry_price
                        pnl_amount = (
                            pnl * self.position * price
                        )  # Simplified position sizing
                        self.capital += pnl_amount - commission_amount

                        # Record closing trade
                        orders.append(
                            {
                                "timestamp": row.name,
                                "action": "close_long",
                                "price": price,
                                "pnl": pnl_amount,
                                "capital_after": self.capital,
                            }
                        )

                    # Open short position
                    self.position = -1
                    self.entry_price = price
                    self.capital -= commission_amount

                    orders.append(
                        {
                            "timestamp": row.name,
                            "action": "sell",
                            "price": price,
                            "capital_after": self.capital,
                        }
                    )

            # Record equity
            equity_curve.append(self.capital)

        return pd.Series(equity_curve[:-1], index=data.index), pd.DataFrame(orders)


def run_action_signal_guide_backtest() -> Tuple[Optional[Any], Optional[Any]]:
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
        engine = SimpleBacktestEngine(
            initial_capital=backtest_config["initial_capital"],
            commission=backtest_config["commission"],
        )
        equity_curve, orders = engine.run_backtest(adapter, data)

        # Calculate metrics
        metrics_calculator = MetricsCalculator()
        performance_metrics = metrics_calculator.calculate_all_metrics(
            equity_curve=equity_curve,
            orders=orders,
            initial_capital=engine.initial_capital,
            risk_free_rate=0.02,
            slippage_bps=0.5,  # 0.05%
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

        # Save results to JSON file
        import json
        from datetime import datetime

        signal_stats = adapter.get_signal_statistics()

        results_data = {
            "timestamp": datetime.now().isoformat(),
            "config": {
                "initial_capital": engine.initial_capital,
                "commission": engine.commission,
                "data_points": len(data),
                "dynamic_thresholds": {
                    "confidence_threshold": 0.7,
                    "signal_strength_threshold": 0.4,
                },  # Fixed values for JSON serialization
            },
            "performance_metrics": performance_metrics_dict,
            "signal_statistics": signal_stats,
            "equity_curve": equity_curve.tolist()
            if hasattr(equity_curve, "tolist")
            else list(equity_curve),
            "orders": [
                {
                    k: (v.isoformat() if hasattr(v, "isoformat") else v)
                    for k, v in order.items()
                }
                for order in (
                    orders.to_dict("records") if hasattr(orders, "to_dict") else orders
                )
                if isinstance(order, dict)
            ],
        }

        output_file = project_root / "backtest_results_action_signal_guide.json"
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(results_data, f, indent=2, ensure_ascii=False)

        print(f"\nResults saved to: {output_file}")

        return (equity_curve, orders, None), performance_metrics

    except Exception as e:
        # Enhanced error handling for backtest failures
        error_type = type(e).__name__
        error_msg = str(e)
        import traceback

        print(f"Backtest failed with {error_type}: {error_msg}")

        # Classify error for better diagnostics
        if "memory" in error_msg.lower():
            error_category = "memory_error"
            print("ERROR: Memory error during backtest execution")
        elif "timeout" in error_msg.lower():
            error_category = "timeout_error"
            print("ERROR: Backtest execution timed out")
        elif "json" in error_msg.lower():
            error_category = "serialization_error"
            print("ERROR: Failed to serialize results to JSON")
        elif "signal" in error_msg.lower():
            error_category = "signal_generation_error"
            print("ERROR: Signal generation failed during backtest")
        else:
            error_category = "backtest_execution_error"
            print(f"ERROR: Unexpected backtest failure: {error_type}")

        # Print full traceback for debugging
        print("\nFull traceback:")
        traceback.print_exc()

        # Attempt to save partial results if possible
        try:
            import json
            from datetime import datetime

            error_results = {
                "error": True,
                "error_category": error_category,
                "error_message": error_msg,
                "error_type": error_type,
                "timestamp": datetime.now().isoformat(),
                "partial_data": {
                    "data_points": len(data) if "data" in locals() else 0,
                    "adapter_initialized": adapter is not None
                    if "adapter" in locals()
                    else False,
                },
            }

            error_file = project_root / "backtest_error_action_signal_guide.json"
            with open(error_file, "w", encoding="utf-8") as f:
                json.dump(error_results, f, indent=2, ensure_ascii=False)
            print(f"Error information saved to: {error_file}")

        except Exception as save_error:
            print(f"Failed to save error information: {save_error}")

        return None, None


if __name__ == "__main__":
    results, metrics = run_action_signal_guide_backtest()
