#!/usr/bin/env python3
"""
SAC v430 Backtest Analysis - Profitability and Win Rate Analysis
"""

import os
import sys
from pathlib import Path
from typing import Dict, List

from ztb.trading.environment.constants import (
    ACTION_BUY,
    ACTION_SELL,
    continuous_to_discrete_action,
)

# Add project root to path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

import numpy as np

from ztb.io.json_io import read_json, write_json

def load_backtest_results(results_path: str) -> Dict:
    """Load backtest results from JSON file."""
    return read_json(results_path)


def analyze_trades(
    portfolio_history: List[float], actions_history: List[float]
) -> List[Dict]:
    """Analyze trades from portfolio and action history."""

    trades = []
    current_position = 0
    entry_price = None
    entry_step = None

    for step, (portfolio_value, continuous_action) in enumerate(
        zip(portfolio_history, actions_history)
    ):
        # Convert continuous action to discrete
        discrete_action = continuous_to_discrete_action(continuous_action)

        # Track position changes
        old_position = current_position

        if discrete_action == ACTION_BUY:
            if current_position <= 0:  # Opening long or closing short
                if current_position < 0:  # Closing short position
                    exit_price = portfolio_value  # Approximate exit price
                    pnl = entry_price - exit_price if entry_price else 0
                    trades.append(
                        {
                            "entry_step": entry_step,
                            "exit_step": step,
                            "position_type": "short",
                            "entry_price": entry_price,
                            "exit_price": exit_price,
                            "pnl": pnl,
                            "duration": step - entry_step if entry_step else 0,
                        }
                    )
                # Open long position
                current_position = 1
                entry_price = portfolio_value
                entry_step = step

        elif discrete_action == ACTION_SELL:
            if current_position >= 0:  # Opening short or closing long
                if current_position > 0:  # Closing long position
                    exit_price = portfolio_value  # Approximate exit price
                    pnl = exit_price - entry_price if entry_price else 0
                    trades.append(
                        {
                            "entry_step": entry_step,
                            "exit_step": step,
                            "position_type": "long",
                            "entry_price": entry_price,
                            "exit_price": exit_price,
                            "pnl": pnl,
                            "duration": step - entry_step if entry_step else 0,
                        }
                    )
                # Open short position
                current_position = -1
                entry_price = portfolio_value
                entry_step = step

        # HOLD action doesn't change position

    # Close any open position at the end
    if current_position != 0 and entry_price is not None:
        exit_price = portfolio_history[-1]
        if current_position > 0:
            pnl = exit_price - entry_price
            position_type = "long"
        else:
            pnl = entry_price - exit_price
            position_type = "short"

        trades.append(
            {
                "entry_step": entry_step,
                "exit_step": len(portfolio_history) - 1,
                "position_type": position_type,
                "entry_price": entry_price,
                "exit_price": exit_price,
                "pnl": pnl,
                "duration": len(portfolio_history) - 1 - entry_step
                if entry_step
                else 0,
            }
        )

    return trades


def calculate_win_rate(trades: List[Dict]) -> Dict:
    """Calculate win rate and related metrics."""

    if not trades:
        return {
            "total_trades": 0,
            "winning_trades": 0,
            "losing_trades": 0,
            "win_rate": 0.0,
            "avg_win": 0.0,
            "avg_loss": 0.0,
            "profit_factor": 0.0,
            "total_pnl": 0.0,
            "max_drawdown": 0.0,
        }

    winning_trades = [t for t in trades if t["pnl"] > 0]
    losing_trades = [t for t in trades if t["pnl"] < 0]

    total_pnl = sum(t["pnl"] for t in trades)
    win_rate = len(winning_trades) / len(trades) * 100

    avg_win = np.mean([t["pnl"] for t in winning_trades]) if winning_trades else 0
    avg_loss = np.mean([t["pnl"] for t in losing_trades]) if losing_trades else 0

    total_wins = sum(t["pnl"] for t in winning_trades)
    total_losses = abs(sum(t["pnl"] for t in losing_trades))
    profit_factor = total_wins / total_losses if total_losses > 0 else float("inf")

    # Calculate drawdown
    cumulative_pnl = np.cumsum([t["pnl"] for t in trades])
    running_max = np.maximum.accumulate(cumulative_pnl)
    drawdowns = running_max - cumulative_pnl
    max_drawdown = np.max(drawdowns) if len(drawdowns) > 0 else 0

    return {
        "total_trades": len(trades),
        "winning_trades": len(winning_trades),
        "losing_trades": len(losing_trades),
        "win_rate": win_rate,
        "avg_win": avg_win,
        "avg_loss": avg_loss,
        "profit_factor": profit_factor,
        "total_pnl": total_pnl,
        "max_drawdown": max_drawdown,
    }


def analyze_profitability_over_time(
    portfolio_history: List[float], timestamps: List[int]
) -> Dict:
    """Analyze profitability over different time periods."""

    portfolio_values = np.array(portfolio_history)
    initial_value = portfolio_values[0]

    # Calculate returns over different periods
    periods = {
        "1h": 60,  # Assuming 1 step = 1 minute
        "6h": 360,
        "12h": 720,
        "1d": 1440,
        "3d": 4320,
        "1w": 10080,
        "1m": 43200,  # Approximate
        "total": len(portfolio_values) - 1,
    }

    profitability = {}

    for period_name, steps in periods.items():
        if steps >= len(portfolio_values):
            continue

        # Calculate periodic returns
        periodic_values = portfolio_values[::steps][:10]  # First 10 periods
        if len(periodic_values) < 2:
            continue

        returns = []
        for i in range(1, len(periodic_values)):
            ret = (
                (periodic_values[i] - periodic_values[i - 1])
                / periodic_values[i - 1]
                * 100
            )
            returns.append(ret)

        profitability[period_name] = {
            "avg_return_pct": np.mean(returns),
            "max_return_pct": np.max(returns),
            "min_return_pct": np.min(returns),
            "volatility_pct": np.std(returns),
            "periods_analyzed": len(returns),
        }

    # Overall performance
    final_value = portfolio_values[-1]
    total_return_pct = (final_value - initial_value) / initial_value * 100
    total_steps = len(portfolio_values) - 1

    profitability["overall"] = {
        "initial_value": initial_value,
        "final_value": final_value,
        "total_return_pct": total_return_pct,
        "total_steps": total_steps,
        "avg_return_per_step": total_return_pct / total_steps,
    }

    return profitability


def analyze_win_rate_causes(trades: List[Dict], actions_history: List[float]) -> Dict:
    """Analyze potential causes of low win rate."""

    if not trades:
        return {"analysis": "No trades to analyze"}

    # Analyze trade characteristics
    long_trades = [t for t in trades if t["position_type"] == "long"]
    short_trades = [t for t in trades if t["position_type"] == "short"]

    long_win_rate = (
        len([t for t in long_trades if t["pnl"] > 0]) / len(long_trades) * 100
        if long_trades
        else 0
    )
    short_win_rate = (
        len([t for t in short_trades if t["pnl"] > 0]) / len(short_trades) * 100
        if short_trades
        else 0
    )

    # Analyze trade duration
    winning_durations = [t["duration"] for t in trades if t["pnl"] > 0]
    losing_durations = [t["duration"] for t in trades if t["pnl"] < 0]

    avg_win_duration = np.mean(winning_durations) if winning_durations else 0
    avg_loss_duration = np.mean(losing_durations) if losing_durations else 0

    # Analyze consecutive actions (potential overtrading)
    action_changes = sum(
        1
        for i in range(1, len(actions_history))
        if actions_history[i] != actions_history[i - 1]
    )
    action_change_rate = action_changes / len(actions_history) * 100

    return {
        "long_vs_short_performance": {
            "long_trades": len(long_trades),
            "short_trades": len(short_trades),
            "long_win_rate": long_win_rate,
            "short_win_rate": short_win_rate,
        },
        "trade_duration_analysis": {
            "avg_winning_trade_duration": avg_win_duration,
            "avg_losing_trade_duration": avg_loss_duration,
            "duration_ratio": avg_win_duration / avg_loss_duration
            if avg_loss_duration > 0
            else float("inf"),
        },
        "action_pattern_analysis": {
            "total_action_changes": action_changes,
            "action_change_rate_pct": action_change_rate,
            "assessment": "High action change rate may indicate overtrading"
            if action_change_rate > 20
            else "Normal action patterns",
        },
    }


def main():
    """Main analysis function."""

    print("🔬 SAC v430 Backtest Analysis")
    print("=" * 60)

    # Load backtest results
    results_path = "results/sac_v430_backtest_results.json"
    if not os.path.exists(results_path):
        print(f"❌ Backtest results not found: {results_path}")
        return 1

    print(f"📊 Loading results from: {results_path}")
    results = load_backtest_results(results_path)

    portfolio_history = results["portfolio_history"]
    actions_history = results["actions_history"]
    timestamps = results.get("timestamps", list(range(len(portfolio_history))))

    print(f"✅ Loaded {len(portfolio_history)} steps of backtest data")

    # Analyze trades
    print("\n🔍 Analyzing trades...")
    trades = analyze_trades(portfolio_history, actions_history)
    print(f"📈 Identified {len(trades)} trades")

    # Calculate win rate
    print("\n📊 Calculating win rate metrics...")
    win_rate_stats = calculate_win_rate(trades)

    # Analyze profitability over time
    print("\n💰 Analyzing profitability over time...")
    profitability = analyze_profitability_over_time(portfolio_history, timestamps)

    # Analyze win rate causes
    print("\n🔍 Analyzing win rate causes...")
    win_rate_causes = analyze_win_rate_causes(trades, actions_history)

    # Print results
    print("\n" + "=" * 60)
    print("📊 BACKTEST ANALYSIS RESULTS")
    print("=" * 60)

    print("\n🎯 WIN RATE ANALYSIS:")
    print(f"   Total Trades: {win_rate_stats['total_trades']}")
    print(f"   Winning Trades: {win_rate_stats['winning_trades']}")
    print(f"   Losing Trades: {win_rate_stats['losing_trades']}")
    print(f"   Win Rate: {win_rate_stats['win_rate']:.1f}%")
    print(f"   Average Win: {win_rate_stats['avg_win']:.2f}")
    print(f"   Average Loss: {win_rate_stats['avg_loss']:.2f}")
    print(f"   Profit Factor: {win_rate_stats['profit_factor']:.2f}")
    print(f"   Total PnL: {win_rate_stats['total_pnl']:.2f}")
    print(f"   Max Drawdown: {win_rate_stats['max_drawdown']:.2f}")

    print("\n💰 PROFITABILITY ANALYSIS:")
    print(f"   Initial Portfolio: {profitability['overall']['initial_value']:.2f}")
    print(f"   Final Portfolio: {profitability['overall']['final_value']:.2f}")
    print(f"   Total Return: {profitability['overall']['total_return_pct']:.2f}%")
    print(f"   Total Steps: {profitability['overall']['total_steps']}")
    for period, data in profitability.items():
        if period == "overall":
            continue
        if "avg_return_pct" in data:
            print(
                f"   {period}: {data['avg_return_pct']:.2f}% avg return ({data['periods_analyzed']} periods)"
            )

    print("\n🔍 WIN RATE CAUSE ANALYSIS:")
    if "long_vs_short_performance" in win_rate_causes:
        lvsp = win_rate_causes["long_vs_short_performance"]
        print(
            f"   Long Trades: {lvsp['long_trades']}, Win Rate: {lvsp['long_win_rate']:.1f}%"
        )
        print(
            f"   Short Trades: {lvsp['short_trades']}, Win Rate: {lvsp['short_win_rate']:.1f}%"
        )

    if "trade_duration_analysis" in win_rate_causes:
        tda = win_rate_causes["trade_duration_analysis"]
        print(
            f"   Avg Winning Trade Duration: {tda['avg_winning_trade_duration']:.1f} steps"
        )
        print(
            f"   Avg Losing Trade Duration: {tda['avg_losing_trade_duration']:.1f} steps"
        )

    if "action_pattern_analysis" in win_rate_causes:
        apa = win_rate_causes["action_pattern_analysis"]
        print(f"   Action Change Rate: {apa['action_change_rate_pct']:.1f}%")
        print(f"   Assessment: {apa['assessment']}")

    print("\n" + "=" * 60)

    # Save detailed analysis
    analysis_results = {
        "win_rate_stats": win_rate_stats,
        "profitability": profitability,
        "win_rate_causes": win_rate_causes,
        "trades": trades[:10],  # First 10 trades as examples
    }

    analysis_path = "results/sac_v430_backtest_analysis.json"
    write_json(analysis_path, analysis_results, indent=2, ensure_ascii=False)

    print(f"💾 Detailed analysis saved to: {analysis_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
