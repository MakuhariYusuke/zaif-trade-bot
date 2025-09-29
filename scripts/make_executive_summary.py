#!/usr/bin/env python3
"""
Executive Summary Generator for Trading Strategies

Generates a one-page executive summary comparing RL vs SMA vs Buy&Hold strategies.
"""

import json
import argparse
from pathlib import Path
from typing import Dict, List, Any, Optional
from datetime import datetime
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


class ExecutiveSummaryGenerator:
    """Generates executive summary from trading results."""

    def __init__(self, results_dir: Path):
        self.results_dir = results_dir
        self.strategies = ['rl', 'sma_fast_slow', 'buy_hold']
        self.results: Dict[str, Dict[str, Any]] = {}

    def load_strategy_results(self, strategy: str) -> Optional[Dict[str, Any]]:
        """Load results for a specific strategy."""
        # Find latest result directory for strategy
        pattern = f"{strategy}_*"
        matching_dirs = list(self.results_dir.glob(pattern))

        if not matching_dirs:
            print(f"No results found for strategy: {strategy}")
            return None

        # Use latest directory
        latest_dir = max(matching_dirs, key=lambda x: x.stat().st_mtime)
        summary_file = latest_dir / 'summary.json'

        if not summary_file.exists():
            print(f"Summary file not found: {summary_file}")
            return None

        with open(summary_file, 'r') as f:
            data = json.load(f)

        # Add directory info
        data['result_dir'] = str(latest_dir)
        data['generated_at'] = datetime.now().isoformat()

        return data

    def load_all_results(self):
        """Load results for all strategies."""
        # Scan all subdirectories for summary.json files
        for result_dir in self.results_dir.glob("*"):
            if not result_dir.is_dir():
                continue

            summary_file = result_dir / 'summary.json'
            if not summary_file.exists():
                continue

            with open(summary_file, 'r') as f:
                data = json.load(f)

            # Infer strategy from directory name or data
            strategy = self._infer_strategy(result_dir, data)
            if strategy:
                data['result_dir'] = str(result_dir)
                data['generated_at'] = datetime.now().isoformat()
                self.results[strategy] = data
                print(f"Loaded results for {strategy}: {result_dir}")

    def _infer_strategy(self, result_dir: Path, data: Dict[str, Any]) -> Optional[str]:
        """Infer strategy name from directory or data."""
        dir_name = result_dir.name.lower()

        # Check directory name patterns
        if 'sma' in dir_name or 'fast_slow' in dir_name:
            return 'sma_fast_slow'
        elif 'buy_hold' in dir_name or 'buyhold' in dir_name:
            return 'buy_hold'
        elif 'rl' in dir_name:
            return 'rl'

        # Check if we have trade_log to infer from signals
        trade_log_file = result_dir / 'trade_log.json'
        if trade_log_file.exists():
            try:
                with open(trade_log_file, 'r') as f:
                    trade_log = json.load(f)

                # For now, assume all are sma_fast_slow if we can't determine
                # In practice, you'd check the signals or have metadata
                return 'sma_fast_slow'
            except:
                pass

        return None

    def calculate_metrics(self) -> Dict[str, Any]:
        """Calculate comparative metrics."""
        metrics = {}

        for strategy, data in self.results.items():
            trade_log = data.get('trade_log', [])
            pnl_series = data.get('pnl_series', [])

            # Basic metrics
            total_trades = len(trade_log)
            total_pnl = sum(t.get('pnl', 0) for t in trade_log)
            win_rate = sum(1 for t in trade_log if t.get('pnl', 0) > 0) / max(total_trades, 1) * 100

            # Risk metrics
            if pnl_series:
                pnl_values = [p.get('pnl', 0) for p in pnl_series]
                max_drawdown = min(pnl_values) if pnl_values else 0
                volatility = pd.Series(pnl_values).std() if pnl_values else 0
            else:
                max_drawdown = 0
                volatility = 0

            # Sharpe ratio (simplified)
            if volatility > 0:
                sharpe_ratio = total_pnl / volatility
            else:
                sharpe_ratio = 0

            # Load statistical significance from metrics.json if available
            metrics_file = Path(data.get('result_dir', '')) / '..' / 'metrics.json'
            dsr = None
            pvalue_bootstrap = None
            if metrics_file.exists():
                try:
                    with open(metrics_file, 'r') as f:
                        metrics_data = json.load(f)
                    dsr = metrics_data.get('deflated_sharpe_ratio')
                    pvalue_bootstrap = metrics_data.get('pvalue_bootstrap')
                except:
                    pass

            metrics[strategy] = {
                'total_trades': total_trades,
                'total_pnl': total_pnl,
                'win_rate': win_rate,
                'max_drawdown': max_drawdown,
                'volatility': volatility,
                'sharpe_ratio': sharpe_ratio,
                'deflated_sharpe_ratio': dsr,
                'pvalue_bootstrap': pvalue_bootstrap,
                'trades_per_day': total_trades / max(data.get('duration_minutes', 60) / (24 * 60), 1)
            }

        return metrics

    def calculate_budget_analysis(self, metrics: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate budget and ROI analysis for strategies."""
        budget = {}

        # Constants
        TRADING_COST_PERCENT = 0.005  # 0.5% per trade
        ANNUAL_TRADING_DAYS = 365
        INITIAL_CAPITAL = 10000
        DEVELOPMENT_COST = 500000  # Amortized over 2 years
        INFRASTRUCTURE_COST_YEARLY = 50000

        for strategy, m in metrics.items():
            daily_pnl = m['total_pnl'] / max(m['trades_per_day'], 1) * m['trades_per_day']
            annual_pnl = daily_pnl * ANNUAL_TRADING_DAYS

            # Trading costs (0.5% of notional per trade)
            avg_trade_size = INITIAL_CAPITAL * 0.1  # Assume 10% of capital per trade
            cost_per_trade = avg_trade_size * TRADING_COST_PERCENT
            annual_trading_costs = cost_per_trade * m['trades_per_day'] * ANNUAL_TRADING_DAYS

            # Net profit after costs
            net_annual_profit = annual_pnl - annual_trading_costs - INFRASTRUCTURE_COST_YEARLY

            # Break-even analysis
            total_fixed_costs = DEVELOPMENT_COST / 2  # Amortized over 2 years
            break_even_trades = total_fixed_costs / max(m['total_pnl'] / max(m['total_trades'], 1), 1)

            # Payback period
            monthly_profit = net_annual_profit / 12
            payback_period_months = total_fixed_costs / max(monthly_profit, 1) if monthly_profit > 0 else float('inf')

            # ROI calculation
            total_investment = DEVELOPMENT_COST + INITIAL_CAPITAL
            annual_roi = (net_annual_profit / total_investment) * 100

            budget[strategy] = {
                'annual_pnl': annual_pnl,
                'trading_costs': annual_trading_costs,
                'net_profit': net_annual_profit,
                'break_even_trades': break_even_trades,
                'payback_period_months': payback_period_months,
                'roi_percentage': annual_roi,
                'total_investment': total_investment
            }

        return budget

    def generate_summary_text(self, metrics: Dict[str, Any]) -> str:
        """Generate executive summary text."""
        lines = []
        lines.append("# Trading Strategy Executive Summary")
        lines.append("")
        lines.append(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        lines.append("")

        # Performance overview
        lines.append("## Performance Overview")
        lines.append("")
        lines.append("| Strategy | Total P&L | Win Rate | Max Drawdown | Sharpe Ratio | Deflated Sharpe | P-Value | Trades/Day |")
        lines.append("|----------|-----------|----------|--------------|--------------|----------------|---------|------------|")

        for strategy in self.strategies:
            if strategy in metrics:
                m = metrics[strategy]
                dsr = f"{m['deflated_sharpe_ratio']:.2f}" if m['deflated_sharpe_ratio'] is not None else "N/A"
                pval = f"{m['pvalue_bootstrap']:.3f}" if m['pvalue_bootstrap'] is not None else "N/A"
                lines.append(f"| {strategy.upper()} | ¥{m['total_pnl']:.0f} | {m['win_rate']:.1f}% | ¥{m['max_drawdown']:.0f} | {m['sharpe_ratio']:.2f} | {dsr} | {pval} | {m['trades_per_day']:.1f} |")
            else:
                lines.append(f"| {strategy.upper()} | N/A | N/A | N/A | N/A | N/A | N/A | N/A |")

        lines.append("")

        # Key insights
        lines.append("## Key Insights")
        lines.append("")

        if metrics:
            # Best performer
            best_pnl = max((m['total_pnl'], s) for s, m in metrics.items())
            lines.append(f"- **Best P&L:** {best_pnl[1].upper()} (¥{best_pnl[0]:.0f})")

            # Best win rate
            best_win = max((m['win_rate'], s) for s, m in metrics.items())
            lines.append(f"- **Best Win Rate:** {best_win[1].upper()} ({best_win[0]:.1f}%)")

            # Lowest drawdown
            best_dd = min((abs(m['max_drawdown']), s) for s, m in metrics.items())
            lines.append(f"- **Lowest Drawdown:** {best_dd[1].upper()} (¥{best_dd[0]:.0f})")

        lines.append("")

        # Budget justification
        lines.append("## Budget Justification")
        lines.append("")
        lines.append("### Cost-Benefit Analysis")
        lines.append("")

        # Calculate budget metrics
        budget_analysis = self.calculate_budget_analysis(metrics)

        lines.append("| Strategy | Annual P&L | Trading Costs | Net Profit | Break-even Trades | Payback Period |")
        lines.append("|----------|-------------|---------------|------------|-------------------|----------------|")

        for strategy in self.strategies:
            if strategy in budget_analysis:
                b = budget_analysis[strategy]
                lines.append(f"| {strategy.upper()} | ¥{b['annual_pnl']:.0f} | ¥{b['trading_costs']:.0f} | ¥{b['net_profit']:.0f} | {b['break_even_trades']:.0f} | {b['payback_period_months']:.1f} months |")
            else:
                lines.append(f"| {strategy.upper()} | N/A | N/A | N/A | N/A | N/A |")

        lines.append("")
        lines.append("### Assumptions")
        lines.append("- Trading costs: 0.5% per trade (maker/taker fees + slippage)")
        lines.append("- Annual trading days: 365")
        lines.append("- Initial capital: ¥10,000")
        lines.append("- Development cost: ¥500,000 (amortized over 2 years)")
        lines.append("- Infrastructure cost: ¥50,000/year")
        lines.append("")

        lines.append("### ROI Projections")
        lines.append("")

        for strategy in self.strategies:
            if strategy in budget_analysis:
                b = budget_analysis[strategy]
                roi = b['roi_percentage']
                status = "✅ Profitable" if roi > 0 else "❌ Not profitable"
                lines.append(f"- **{strategy.upper()}:** {roi:.1f}% ROI - {status}")

        lines.append("")
        lines.append("## Recommendations")
        lines.append("")
        lines.append("Based on the results:")
        lines.append("")
        lines.append("1. **Production Candidate:** [Strategy with best risk-adjusted returns]")
        lines.append("2. **Further Testing:** [Strategy needing more evaluation]")
        lines.append("3. **Risk Considerations:** [Key risk factors to monitor]")
        lines.append("")
        lines.append("## Data Sources")
        lines.append("")

        for strategy, data in self.results.items():
            lines.append(f"- **{strategy.upper()}:** {data.get('result_dir', 'N/A')}")
            lines.append(f"  - Generated: {data.get('generated_at', 'N/A')}")
            lines.append(f"  - Duration: {data.get('duration_minutes', 'N/A')} minutes")

        return "\n".join(lines)

    def generate_chart(self, output_path: Path):
        """Generate comparison chart."""
        if not self.results:
            print("No results to chart")
            return

        # Prepare data for plotting
        strategies = []
        pnl_values = []
        win_rates = []

        for strategy, data in self.results.items():
            trade_log = data.get('trade_log', [])
            total_pnl = sum(t.get('pnl', 0) for t in trade_log)
            win_rate = sum(1 for t in trade_log if t.get('pnl', 0) > 0) / max(len(trade_log), 1) * 100

            strategies.append(strategy.upper())
            pnl_values.append(total_pnl)
            win_rates.append(win_rate)

        # Create subplots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

        # P&L comparison
        bars1 = ax1.bar(strategies, pnl_values, color=['blue', 'green', 'red'])
        ax1.set_title('Total P&L by Strategy')
        ax1.set_ylabel('P&L (JPY)')
        ax1.grid(True, alpha=0.3)

        # Add value labels
        for bar, value in zip(bars1, pnl_values):
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                    f'¥{value:.0f}', ha='center', va='bottom')

        # Win rate comparison
        bars2 = ax2.bar(strategies, win_rates, color=['lightblue', 'lightgreen', 'lightcoral'])
        ax2.set_title('Win Rate by Strategy')
        ax2.set_ylabel('Win Rate (%)')
        ax2.grid(True, alpha=0.3)

        # Add value labels
        for bar, value in zip(bars2, win_rates):
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                    f'{value:.1f}%', ha='center', va='bottom')

        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()

        print(f"Chart saved to: {output_path}")

    def generate_report(self, output_file: Path, include_chart: bool = True):
        """Generate complete executive summary report."""
        print("Loading strategy results...")
        self.load_all_results()

        if not self.results:
            print("No results found. Cannot generate summary.")
            return

        print(f"Found results for {len(self.results)} strategies: {list(self.results.keys())}")

        # Calculate metrics
        metrics = self.calculate_metrics()

        # Generate summary text
        summary_text = self.generate_summary_text(metrics)

        # Write to file
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(summary_text)

        print(f"Executive summary saved to: {output_file}")

        # Generate chart
        if include_chart:
            chart_file = output_file.with_suffix('.png')
            self.generate_chart(chart_file)


def main():
    parser = argparse.ArgumentParser(description='Generate executive summary from trading results')
    parser.add_argument('--results-dir', default='results/paper',
                       help='Directory containing strategy results (default: results/paper)')
    parser.add_argument('--output', default='executive_summary.md',
                       help='Output file for summary (default: executive_summary.md)')
    parser.add_argument('--no-chart', action='store_true',
                       help='Skip chart generation')

    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    output_file = Path(args.output)

    if not results_dir.exists():
        print(f"Results directory not found: {results_dir}")
        return 1

    generator = ExecutiveSummaryGenerator(results_dir)
    generator.generate_report(output_file, include_chart=not args.no_chart)

    return 0


if __name__ == '__main__':
    exit(main())