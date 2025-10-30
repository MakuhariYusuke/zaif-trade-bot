#!/usr/bin/env python3
"""
Action Signal Guide Backtest Results Runner

This module handles the execution and reporting of ActionSignalGuide backtests.
"""

import json
from pathlib import Path
from typing import Any, Dict, Optional, Tuple


def display_backtest_results(performance_metrics: Dict[str, Any]) -> None:
    """
    Display backtest results in a formatted manner.

    Args:
        results: Raw backtest results
        performance_metrics: Calculated performance metrics
    """
    print("\n=== Backtest Results ===")
    print(f"Final capital: ${performance_metrics['final_capital']:,.2f}")
    print(f"Total return: {performance_metrics['total_return']:.2%}")
    print(f"Annual return: {performance_metrics['annual_return']:.2%}")
    print(f"Max drawdown: {performance_metrics['max_drawdown']:.2%}")
    print(f"Sharpe ratio: {performance_metrics['sharpe_ratio']:.2f}")
    print(f"Win rate: {performance_metrics['win_rate']:.2%}")
    print(f"Total trades: {performance_metrics['total_trades']}")
    print(f"Profit factor: {performance_metrics['profit_factor']:.2f}")


def display_signal_statistics(adapter) -> None:
    """
    Display signal generation statistics.

    Args:
        adapter: The ActionSignalGuideAdapter instance
    """
    signal_stats = adapter.get_signal_statistics()
    print("\nSignal Statistics:")
    print(f"Total signals generated: {signal_stats['total_signals']}")
    print(f"Buy signals: {signal_stats['buy_signals']}")
    print(f"Sell signals: {signal_stats['sell_signals']}")
    print(f"Hold signals: {signal_stats['hold_signals']}")


def save_results_to_file(
    results: Dict[str, Any],
    performance_metrics: Dict[str, Any],
    signal_stats: Dict[str, Any],
    output_dir: str = "backtest_results",
) -> None:
    """
    Save backtest results to JSON files.

    Args:
        results: Raw backtest results
        performance_metrics: Calculated performance metrics
        signal_stats: Signal generation statistics
        output_dir: Directory to save results
    """
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)

    # Save performance metrics
    metrics_file = output_path / "action_signal_guide_performance.json"
    with open(metrics_file, "w") as f:
        json.dump(performance_metrics, f, indent=2, default=str)

    # Save signal statistics
    signals_file = output_path / "action_signal_guide_signals.json"
    with open(signals_file, "w") as f:
        json.dump(signal_stats, f, indent=2)

    # Save raw results
    results_file = output_path / "action_signal_guide_results.json"
    with open(results_file, "w") as f:
        json.dump(results, f, indent=2, default=str)

    print(f"\nResults saved to {output_path}/")


def generate_summary_report(
    performance_metrics: Dict[str, Any], signal_stats: Dict[str, Any]
) -> str:
    """
    Generate a summary report of the backtest.

    Args:
        performance_metrics: Calculated performance metrics
        signal_stats: Signal generation statistics

    Returns:
        Formatted summary report as string
    """
    report = []
    report.append("# Action Signal Guide Backtest Report")
    report.append("")

    report.append("## Performance Metrics")
    report.append(f"- **Final Capital**: ${performance_metrics['final_capital']:,.2f}")
    report.append(f"- **Total Return**: {performance_metrics['total_return']:.2%}")
    report.append(f"- **Annual Return**: {performance_metrics['annual_return']:.2%}")
    report.append(f"- **Max Drawdown**: {performance_metrics['max_drawdown']:.2%}")
    report.append(f"- **Sharpe Ratio**: {performance_metrics['sharpe_ratio']:.2f}")
    report.append(f"- **Win Rate**: {performance_metrics['win_rate']:.2%}")
    report.append(f"- **Total Trades**: {performance_metrics['total_trades']}")
    report.append(f"- **Profit Factor**: {performance_metrics['profit_factor']:.2f}")
    report.append("")

    report.append("## Signal Statistics")
    report.append(f"- **Total Signals**: {signal_stats['total_signals']}")
    report.append(f"- **Buy Signals**: {signal_stats['buy_signals']}")
    report.append(f"- **Sell Signals**: {signal_stats['sell_signals']}")
    report.append(f"- **Hold Signals**: {signal_stats['hold_signals']}")
    report.append("")

    # Generate conclusion
    total_return = performance_metrics["total_return"]
    win_rate = performance_metrics["win_rate"]
    total_signals = signal_stats["total_signals"]

    report.append("## Conclusion")
    if total_return > 0:
        report.append(
            "✅ **Positive Results**: The Action Signal Guide generated positive returns."
        )
    else:
        report.append(
            "❌ **Negative Results**: The Action Signal Guide generated negative returns."
        )

    if win_rate > 0.5:
        report.append("✅ **Good Win Rate**: Win rate above 50%.")
    else:
        report.append("⚠️ **Poor Win Rate**: Win rate below 50%.")

    if total_signals > 0:
        report.append(
            "✅ **Signal Generation**: Successfully generated trading signals."
        )
    else:
        report.append("❌ **No Signals**: No trading signals were generated.")

    return "\n".join(report)


def run_and_report_backtest(
    backtest_function, save_results: bool = True
) -> Tuple[Optional[Dict], Optional[Dict]]:
    """
    Run a backtest and generate reports.

    Args:
        backtest_function: Function that runs the backtest and returns (results, metrics)
        save_results: Whether to save results to files

    Returns:
        Tuple of (results, performance_metrics)
    """
    try:
        results, performance_metrics = backtest_function()

        if results is None or performance_metrics is None:
            print("Backtest failed - no results to report")
            return None, None

        # Display results
        display_backtest_results(results, performance_metrics)

        # Get signal statistics (assuming adapter is available in global scope or passed)
        # This would need to be modified based on how the adapter is accessed
        # display_signal_statistics(adapter)

        # Save results if requested
        if save_results:
            # signal_stats = adapter.get_signal_statistics()
            # save_results_to_file(results, performance_metrics, signal_stats)
            pass  # Placeholder for now

        # Generate summary report
        # report = generate_summary_report(performance_metrics, signal_stats)
        # print("\n" + "="*50)
        # print("SUMMARY REPORT")
        # print("="*50)
        # print(report)

        return results, performance_metrics

    except Exception as e:
        print(f"Error in backtest execution: {e}")
        import traceback

        traceback.print_exc()
        return None, None
