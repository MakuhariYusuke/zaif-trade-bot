"""
Analysis Display Manager

Manages display and visualization of analysis results.
Specialized version of DisplayManager for analysis tasks.
Separated to follow Single Responsibility Principle.
"""

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from ztb.analysis.common.plot_utils import save_plot
from ztb.utils.logging_utils import get_logger

logger = get_logger(__name__)

class AnalysisDisplayManager:
    """
    Manages display and visualization of analysis results.

    Responsibilities:
    - Displaying analysis results in various formats
    - Creating plots and charts for analysis data
    - Formatting output for different display modes
    - Managing analysis-specific visualizations
    """

    def __init__(self, output_dir: str = "analysis_results"):
        self.logger = get_logger(__name__)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)

        # set up plotting style
        plt.style.use("default")
        if hasattr(sns, "set_palette"):
            sns.set_palette("husl")

    def display_backtest_results(
        self,
        results: dict[str, Any],
        title: str = "Backtest Results",
        show_plots: bool = True,
        save_plots: bool = True,
    ) -> None:
        """
        Display backtest analysis results.

        Args:
            results: Backtest results dictionary
            title: Title for the display
            show_plots: Whether to display plots
            save_plots: Whether to save plots to files
        """
        self.logger.info(f"Displaying backtest results: {title}")

        print(f"\n{'='*60}")
        print(f"📊 {title.upper()}")
        print(f"{'='*60}")

        # Display key metrics
        self._display_key_metrics(results)

        # Display performance summary
        self._display_performance_summary(results)

        # Create plots
        if show_plots or save_plots:
            self._create_backtest_plots(results, title, show_plots, save_plots)

    def _display_key_metrics(self, results: dict[str, Any]) -> None:
        """Display key performance metrics."""
        key_metrics = [
            "total_return_pct",
            "annualized_return_pct",
            "sharpe_ratio",
            "max_drawdown_pct",
            "win_rate",
            "total_trades",
            "avg_trade_return_pct",
        ]

        print("\n🎯 Key Metrics:")
        print("-" * 40)

        for metric in key_metrics:
            if metric in results:
                value = results[metric]
                if isinstance(value, float):
                    if "pct" in metric:
                        print(f"  {metric.replace('_', ' ').title()}: {value:.2f}%")
                    elif "ratio" in metric:
                        print(f"  {metric.replace('_', ' ').title()}: {value:.4f}")
                    else:
                        print(f"  {metric.replace('_', ' ').title()}: {value:.4f}")
                else:
                    print(f"  {metric.replace('_', ' ').title()}: {value}")

    def _display_performance_summary(self, results: dict[str, Any]) -> None:
        """Display performance summary."""
        print("\n📈 Performance Summary:")
        print("-" * 40)

        # Risk-adjusted returns
        if "sharpe_ratio" in results:
            sharpe = results["sharpe_ratio"]
            if sharpe > 1.0:
                print(f"  Sharpe Ratio: {sharpe:.4f} ✅ (Good)")
            elif sharpe > 0.5:
                print(f"  Sharpe Ratio: {sharpe:.4f} ⚠️ (Moderate)")
            else:
                print(f"  Sharpe Ratio: {sharpe:.4f} ❌ (Poor)")

        # Drawdown analysis
        if "max_drawdown_pct" in results:
            dd = results["max_drawdown_pct"]
            if dd < 10:
                print(f"  Max Drawdown: {dd:.2f}% ✅ (Low)")
            elif dd < 20:
                print(f"  Max Drawdown: {dd:.2f}% ⚠️ (Moderate)")
            else:
                print(f"  Max Drawdown: {dd:.2f}% ❌ (High)")

        # Win rate analysis
        if "win_rate" in results:
            wr = results["win_rate"]
            if wr > 0.6:
                print(f"  Win Rate: {wr:.1%} ✅ (Good)")
            elif wr > 0.5:
                print(f"  Win Rate: {wr:.1%} ⚠️ (Moderate)")
            else:
                print(f"  Win Rate: {wr:.1%} ❌ (Poor)")

    def _create_backtest_plots(
        self, results: dict[str, Any], title: str, show_plots: bool, save_plots: bool
    ) -> None:
        """Create backtest visualization plots."""
        try:
            # Portfolio value over time
            self._plot_portfolio_value(results, title, show_plots, save_plots)

            # Drawdown chart
            self._plot_drawdown(results, title, show_plots, save_plots)

            # Monthly returns heatmap
            self._plot_monthly_returns(results, title, show_plots, save_plots)

            # Trade analysis
            self._plot_trade_analysis(results, title, show_plots, save_plots)

        except Exception as e:
            self.logger.warning(f"Failed to create backtest plots: {e}")

    def _plot_portfolio_value(
        self, results: dict[str, Any], title: str, show_plots: bool, save_plots: bool
    ) -> None:
        """Plot portfolio value over time."""
        if "portfolio_history" not in results or "timestamps" not in results:
            return

        portfolio = np.array(results["portfolio_history"])
        timestamps = pd.to_datetime(results["timestamps"])

        plt.figure(figsize=(12, 6))
        plt.plot(timestamps, portfolio, "b-", linewidth=1.5)
        plt.title(f"Portfolio Value Over Time - {title}")
        plt.xlabel("Date")
        plt.ylabel("Portfolio Value")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()

        if save_plots:
            filename = (
                self.output_dir
                / f"portfolio_value_{title.lower().replace(' ', '_')}.png"
            )
            save_plot(filename)
            self.logger.info(f"Saved portfolio value plot to {filename}")

        if show_plots:
            plt.show()
        else:
            plt.close()

    def _plot_drawdown(
        self, results: dict[str, Any], title: str, show_plots: bool, save_plots: bool
    ) -> None:
        """Plot drawdown chart."""
        if "portfolio_history" not in results:
            return

        portfolio = np.array(results["portfolio_history"])
        peak = np.maximum.accumulate(portfolio)
        drawdown = (portfolio - peak) / peak * 100

        plt.figure(figsize=(12, 6))
        plt.fill_between(range(len(drawdown)), 0, drawdown, color="red", alpha=0.3)
        plt.plot(drawdown, "r-", linewidth=1)
        plt.title(f"Portfolio Drawdown - {title}")
        plt.xlabel("Time Steps")
        plt.ylabel("Drawdown (%)")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()

        if save_plots:
            filename = (
                self.output_dir / f"drawdown_{title.lower().replace(' ', '_')}.png"
            )
            save_plot(filename)
            self.logger.info(f"Saved drawdown plot to {filename}")

        if show_plots:
            plt.show()
        else:
            plt.close()

    def _plot_monthly_returns(
        self, results: dict[str, Any], title: str, show_plots: bool, save_plots: bool
    ) -> None:
        """Plot monthly returns heatmap."""
        if "monthly_returns" not in results:
            return

        monthly_returns = results["monthly_returns"]

        # Convert to DataFrame for heatmap
        if isinstance(monthly_returns, dict):
            df = pd.DataFrame(monthly_returns)
        else:
            return

        plt.figure(figsize=(10, 6))
        sns.heatmap(
            df.T,
            annot=True,
            fmt=".1f",
            cmap="RdYlGn",
            center=0,
            cbar_kws={"label": "Monthly Return (%)"},
        )
        plt.title(f"Monthly Returns Heatmap - {title}")
        plt.tight_layout()

        if save_plots:
            filename = (
                self.output_dir
                / f"monthly_returns_{title.lower().replace(' ', '_')}.png"
            )
            save_plot(filename)
            self.logger.info(f"Saved monthly returns plot to {filename}")

        if show_plots:
            plt.show()
        else:
            plt.close()

    def _plot_trade_analysis(
        self, results: dict[str, Any], title: str, show_plots: bool, save_plots: bool
    ) -> None:
        """Plot trade analysis charts."""
        if "trade_analysis" not in results:
            return

        trade_data = results["trade_analysis"]

        # Create subplots for trade analysis
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        fig.suptitle(f"Trade Analysis - {title}")

        # Trade returns distribution
        if "returns" in trade_data:
            returns = trade_data["returns"]
            axes[0, 0].hist(returns, bins=50, alpha=0.7, color="blue")
            axes[0, 0].set_title("Trade Returns Distribution")
            axes[0, 0].set_xlabel("Return (%)")
            axes[0, 0].set_ylabel("Frequency")

        # Trade duration
        if "durations" in trade_data:
            durations = trade_data["durations"]
            axes[0, 1].hist(durations, bins=30, alpha=0.7, color="green")
            axes[0, 1].set_title("Trade Duration Distribution")
            axes[0, 1].set_xlabel("Duration (steps)")
            axes[0, 1].set_ylabel("Frequency")

        # Win/Loss pie chart
        if "win_count" in trade_data and "loss_count" in trade_data:
            win_count = trade_data["win_count"]
            loss_count = trade_data["loss_count"]
            axes[1, 0].pie(
                [win_count, loss_count],
                labels=["Wins", "Losses"],
                autopct="%1.1f%%",
                colors=["green", "red"],
            )
            axes[1, 0].set_title("Win/Loss Ratio")

        # Cumulative returns
        if "cumulative_returns" in trade_data:
            cum_returns = trade_data["cumulative_returns"]
            axes[1, 1].plot(cum_returns, "b-")
            axes[1, 1].set_title("Cumulative Trade Returns")
            axes[1, 1].set_xlabel("Trade Number")
            axes[1, 1].set_ylabel("Cumulative Return (%)")
            axes[1, 1].grid(True, alpha=0.3)

        plt.tight_layout()

        if save_plots:
            filename = (
                self.output_dir
                / f"trade_analysis_{title.lower().replace(' ', '_')}.png"
            )
            save_plot(filename)
            self.logger.info(f"Saved trade analysis plot to {filename}")

        if show_plots:
            plt.show()
        else:
            plt.close()

    def display_comparison_results(
        self,
        comparisons: list[dict[str, Any]],
        metric_names: list[str],
        titles: list[str],
        show_plots: bool = True,
        save_plots: bool = True,
    ) -> None:
        """
        Display comparison results between different analyses.

        Args:
            comparisons: list of comparison result dictionaries
            metric_names: list of metric names to compare
            titles: Titles for each comparison
            show_plots: Whether to display plots
            save_plots: Whether to save plots
        """
        print(f"\n{'='*60}")
        print("ANALYSIS COMPARISON RESULTS")
        print(f"{'='*60}")

        # Display comparison table
        self._display_comparison_table(comparisons, metric_names, titles)

        # Create comparison plots
        if show_plots or save_plots:
            self._create_comparison_plots(
                comparisons, metric_names, titles, show_plots, save_plots
            )

    def _display_comparison_table(
        self,
        comparisons: list[dict[str, Any]],
        metric_names: list[str],
        titles: list[str],
    ) -> None:
        """Display comparison table."""
        print(f"{'Analysis':<25} {' | '.join(f'{m[:12]:<12}' for m in metric_names)}")
        print("-" * (25 + len(metric_names) * 14))

        for i, (comparison, title) in enumerate(zip(comparisons, titles)):
            metric_values = []
            for metric in metric_names:
                if metric in comparison:
                    value = comparison[metric]
                    if isinstance(value, float):
                        if "pct" in metric or "return" in metric:
                            metric_values.append(f"{value:<12.2f}")
                        else:
                            metric_values.append(f"{value:<12.4f}")
                    else:
                        metric_values.append(f"{value:<12}")
                else:
                    metric_values.append(f"{'N/A':<12}")

            print(f"{title:<25} {' | '.join(metric_values)}")

        print()

    def _create_comparison_plots(
        self,
        comparisons: list[dict[str, Any]],
        metric_names: list[str],
        titles: list[str],
        show_plots: bool,
        save_plots: bool,
    ) -> None:
        """Create comparison plots."""
        n_metrics = len(metric_names)
        if n_metrics == 0:
            return

        # Create bar plot for each metric
        if n_metrics == 1:
            fig, axes = plt.subplots(1, 1, figsize=(8, 6))
            axes = [axes]
        else:
            fig, axes = plt.subplots(1, n_metrics, figsize=(6 * n_metrics, 6))

        for i, metric in enumerate(metric_names):
            values = []
            for comparison in comparisons:
                if metric in comparison:
                    values.append(comparison[metric])
                else:
                    values.append(0)

            # Convert to numpy array to avoid matplotlib recursion issues
            values = np.array(values, dtype=float)
            x_positions = np.arange(len(titles))

            bars = axes[i].bar(x_positions, values)
            axes[i].set_title(f'{metric.replace("_", " ").title()} Comparison')
            axes[i].set_ylabel(metric.replace("_", " ").title())
            axes[i].set_xticks(x_positions)
            axes[i].set_xticklabels(titles, rotation=45, ha="right")

            # Add value labels on bars
            for bar, value in zip(bars, values):
                height = bar.get_height()
                axes[i].text(
                    bar.get_x() + bar.get_width() / 2.0,
                    height,
                    f"{value:.2f}" if isinstance(value, float) else str(value),
                    ha="center",
                    va="bottom",
                )

            axes[i].tick_params(axis="x", rotation=45)

        plt.tight_layout()

        if save_plots:
            filename = self.output_dir / "analysis_comparison.png"
            save_plot(filename)
            self.logger.info(f"Saved comparison plot to {filename}")

        if show_plots:
            plt.show()
        else:
            plt.close()
