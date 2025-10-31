#!/usr/bin/env python3
"""
Individual Pattern Recognizer Validation Script

This script performs backtesting of individual ActionSignalGuide pattern recognizers
to evaluate their profitability and signal generation capabilities separately.
"""

import logging
import sys
from pathlib import Path
from typing import Any, Dict, List

# Debug mode configuration
DEBUG_MODE = False  # Set to False for full testing
DEBUG_DATA_LENGTH = 1000 if DEBUG_MODE else 5000
DEBUG_LOG_LEVEL = "WARNING" if DEBUG_MODE else "INFO"

# Patterns to test in debug mode (focus on problematic ones)
DEBUG_PATTERNS = ["harmonic", "dow_theory", "fibonacci", "oscillator"] if DEBUG_MODE else None

# Configure logging for debug mode
if DEBUG_MODE:
    logging.basicConfig(level=getattr(logging, DEBUG_LOG_LEVEL), format='%(levelname)s: %(message)s')
else:
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

# Add project root to path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from backtest.config import get_backtest_config_for_pattern, get_engine_config
from backtest.data_generator import generate_synthetic_data
from backtest.results_runner import display_backtest_results, display_signal_statistics
from ztb.trading.backtest.adapters import ActionSignalGuideAdapter
from ztb.trading.backtest.metrics import MetricsCalculator
from ztb.trading.backtest.runner import BacktestEngine

# Available pattern recognizers
PATTERN_RECOGNIZERS = [
    "candlestick",
    "fibonacci",
    "gann",
    "wave",
    "harmonic",
    "oscillator",
    "volume",
    "bollinger",
    "adx",
    "granville",
    "heikin_ashi",
    "dow_theory",
]

# Debug mode configuration
DEBUG_MODE = True  # Set to False for full testing
DEBUG_DATA_LENGTH = 1000 if DEBUG_MODE else 5000
DEBUG_LOG_LEVEL = "WARNING" if DEBUG_MODE else "INFO"

# Patterns to test in debug mode (focus on problematic ones)
DEBUG_PATTERNS = ["harmonic", "dow_theory", "fibonacci", "oscillator"] if DEBUG_MODE else None


def run_individual_pattern_backtest(pattern_name: str) -> Dict[str, Any]:
    """Run backtest for a specific pattern recognizer."""
    print(f"\n=== Testing {pattern_name.upper()} Pattern Recognizer ===")

    # Generate data (reuse the same data for fair comparison)
    data = generate_synthetic_data(DEBUG_DATA_LENGTH)  # Use debug data length

    # Configure ActionSignalGuide for specific pattern
    config = get_backtest_config_for_pattern(pattern_name)

    # Create adapter
    adapter = ActionSignalGuideAdapter(config=config)

    # Backtest configuration
    backtest_config = get_engine_config()

    print(f"Testing {pattern_name} patterns...")
    print(f"Data period: {data.index[0]} to {data.index[-1]}")
    print(f"Total bars: {len(data)}")

    # Run backtest
    try:
        engine = BacktestEngine(
            initial_capital=backtest_config["initial_capital"],
            commission_bps=backtest_config["commission"] * 100,
            enable_risk=backtest_config["enable_risk_management"],
            max_position_size=backtest_config["max_position_size"],
        )

        # Start backtest from index 200 to ensure sufficient data for pattern recognition
        backtest_data = data.iloc[200:].copy()
        print(f"Starting backtest from index 200, using {len(backtest_data)} data points")

        results = engine.run_backtest(strategy=adapter, data=backtest_data)

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

        return {
            "pattern": pattern_name,
            "performance": performance_metrics_dict,
            "signal_stats": adapter.get_signal_statistics(),
            "orders": len(orders),
            "success": True,
        }

    except Exception as e:
        print(f"Backtest failed for {pattern_name}: {e}")
        import traceback

        traceback.print_exc()
        return {"pattern": pattern_name, "success": False, "error": str(e)}


def run_all_individual_tests() -> List[Dict[str, Any]]:
    """Run backtests for all individual pattern recognizers."""
    results = []

    # Use debug patterns if in debug mode
    patterns_to_test = DEBUG_PATTERNS if DEBUG_PATTERNS else PATTERN_RECOGNIZERS

    print("=== Individual Pattern Recognizer Validation ===")
    print(f"Testing {len(patterns_to_test)} pattern recognizers individually")
    if DEBUG_MODE:
        print(f"DEBUG MODE: Testing only problematic patterns: {patterns_to_test}")

    for pattern in patterns_to_test:
        result = run_individual_pattern_backtest(pattern)
        results.append(result)

    return results


def analyze_results(results: List[Dict[str, Any]]) -> None:
    """Analyze and display comparative results."""
    print("\n" + "=" * 80)
    print("COMPARATIVE ANALYSIS")
    print("=" * 80)

    successful_results = [r for r in results if r.get("success", False)]

    if not successful_results:
        print("No successful tests to analyze.")
        return

    # Sort by total return
    sorted_results = sorted(
        successful_results, key=lambda x: x["performance"]["total_return"], reverse=True
    )

    print(
        f"{'Pattern':<15} {'Total Return':<12} {'Win Rate':<9} {'Profit Factor':<13} {'Trades':<7} {'Signals':<8}"
    )
    print("-" * 80)
    print(f"{'(%)':<15} {'(%)':<12} {'(%)':<9} {'':<13} {'Count':<7} {'Count':<8}")
    print("-" * 80)

    for result in sorted_results:
        perf = result["performance"]
        signals = result["signal_stats"]
        print(
            f"{result['pattern'].upper():<15} {perf['total_return']:<12.2f} {perf['win_rate']:<9.2f} {perf['profit_factor']:<13.2f} {perf['total_trades']:<7} {signals['total_signals']:<8}"
        )

    # Best and worst performers
    best = sorted_results[0]
    worst = sorted_results[-1]

    print("\nBEST PERFORMER:")
    print(f"  Pattern: {best['pattern'].upper()}")
    print(f"  Total Return: {best['performance']['total_return']:.2f}%")
    print(f"  Win Rate: {best['performance']['win_rate']:.2f}%")
    print(f"  Profit Factor: {best['performance']['profit_factor']:.2f}")

    print("\nWORST PERFORMER:")
    print(f"  Pattern: {worst['pattern'].upper()}")
    print(f"  Total Return: {worst['performance']['total_return']:.2f}%")
    print(f"  Win Rate: {worst['performance']['win_rate']:.2f}%")
    print(f"  Profit Factor: {worst['performance']['profit_factor']:.2f}")
    print("\nSUMMARY STATISTICS:")
    total_returns = [r["performance"]["total_return"] for r in successful_results]
    win_rates = [r["performance"]["win_rate"] for r in successful_results]
    profit_factors = [r["performance"]["profit_factor"] for r in successful_results]
    
    print(f"  Average Total Return: {sum(total_returns)/len(total_returns):.2f}%")
    print(f"  Average Win Rate: {sum(win_rates)/len(win_rates):.2f}%")
    print(f"  Average Profit Factor: {sum(profit_factors)/len(profit_factors):.2f}")
    print(f"  Best Total Return: {max(total_returns):.2f}%")
    print(f"  Worst Total Return: {min(total_returns):.2f}%")
    print(f"  Patterns Tested: {len(successful_results)}")


if __name__ == "__main__":
    results = run_all_individual_tests()
    analyze_results(results)
