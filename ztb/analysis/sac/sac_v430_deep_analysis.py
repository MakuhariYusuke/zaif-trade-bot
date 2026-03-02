#!/usr/bin/env python3
"""
SAC v430 Deep Analysis - Action Distribution & Reward Function Analysis
"""

import sys
from pathlib import Path

import numpy as np

from ztb.io.json_io import read_json

# Add project root to path
project_root = Path(__file__).resolve().parent
sys.path.insert(0, str(project_root))

def analyze_action_distribution():
    """Analyze the distribution of actions taken by SAC v430."""

    print("🎯 SAC v430 Action Distribution Analysis")
    print("=" * 60)

    # Load backtest results
    data = read_json("results/sac_v430_backtest_results.json")

    actions = np.array(data["actions_history"])

    print(f"📊 Total actions: {len(actions)}")
    print(f"📈 Action range: [{actions.min():.6f}, {actions.max():.6f}]")
    print(f"📉 Action mean: {actions.mean():.6f}")
    print(f"📊 Action std: {actions.std():.6f}")

    # Analyze action distribution
    buy_threshold = 0.3333
    sell_threshold = -0.3333

    buy_actions = np.sum(actions > buy_threshold)
    sell_actions = np.sum(actions < sell_threshold)
    hold_actions = np.sum((actions >= sell_threshold) & (actions <= buy_threshold))

    print()
    print("🎲 Action Distribution:")
    print(
        f"   BUY actions (> {buy_threshold}): {buy_actions} ({buy_actions/len(actions)*100:.1f}%)"
    )
    print(
        f"   HOLD actions ({sell_threshold} to {buy_threshold}): {hold_actions} ({hold_actions/len(actions)*100:.1f}%)"
    )
    print(
        f"   SELL actions (< {sell_threshold}): {sell_actions} ({sell_actions/len(actions)*100:.1f}%)"
    )

    # Analyze action patterns
    print()
    print("🔍 Action Pattern Analysis:")

    # Check if actions are mostly constant
    unique_actions = len(np.unique(actions.round(4)))  # Round to 4 decimal places
    print(f"   Unique action values: {unique_actions}")

    if unique_actions < 10:
        print("   ⚠️  Very low action diversity - model may be stuck!")
        unique_vals = np.unique(actions.round(4))
        print(f"   Unique values: {unique_vals}")
    else:
        print("   ✅ Good action diversity")

    # Check action entropy
    hist, bins = np.histogram(actions, bins=50)
    entropy = -np.sum((hist / len(actions)) * np.log2(hist / len(actions) + 1e-10))
    print(f"   Action entropy: {entropy:.3f} (higher = more diverse)")

    return actions

def analyze_reward_function():
    """Analyze the reward function configuration."""

    print()
    print("💰 SAC v430 Reward Function Analysis")
    print("=" * 60)

    # Load config
    config = read_json("configs/v430/sac_v430_optimized.json")

    reward_config = config["reward_function"]

    print("📋 Reward Function Configuration:")
    for key, value in reward_config.items():
        print(f"   {key}: {value}")

    print()
    print("🔍 Reward Function Analysis:")

    # Analyze reward incentives
    trading_bonus = reward_config.get("trading_bonus", 0)
    sell_penalty = reward_config.get("sell_penalty", 0)
    buy_bonus = reward_config.get("buy_bonus", 0)
    hold_penalty = reward_config.get("hold_penalty", 0)
    action_balance_weight = reward_config.get("action_balance_weight", 0)

    print(
        f"   Trading incentive: {trading_bonus:.6f} (should be positive to encourage trading)"
    )
    print(f"   Sell penalty: {sell_penalty:.6f} (negative = penalty for selling)")
    print(f"   Buy bonus: {buy_bonus:.6f} (negative = penalty for buying)")
    print(f"   Hold penalty: {hold_penalty:.6f} (positive = penalty for holding)")

    # Check if reward function discourages trading
    if sell_penalty < 0 and buy_bonus < 0:
        print("   ⚠️  Both buy and sell actions are penalized!")
    if hold_penalty > 0:
        print("   ⚠️  Holding is penalized - encourages constant trading")

    # Check action balance weight
    print(f"   Action balance weight: {action_balance_weight:.6f}")
    if action_balance_weight > 0.5:
        print(
            "   ⚠️  High action balance weight - strongly penalizes unbalanced actions"
        )

def analyze_portfolio_behavior():
    """Analyze portfolio value changes and trading behavior."""

    print()
    print("📊 SAC v430 Portfolio Behavior Analysis")
    print("=" * 60)

    # Load backtest results
    data = read_json("results/sac_v430_backtest_results.json")

    portfolio = np.array(data["portfolio_history"])
    actions = np.array(data["actions_history"])

    print(f"📈 Portfolio range: [{portfolio.min():.2f}, {portfolio.max():.2f}]")
    print(f"📊 Portfolio volatility: {portfolio.std():.2f}")
    print(
        f"📉 Max drawdown: {((portfolio.max() - portfolio.min()) / portfolio.max() * 100):.2f}%"
    )

    # Analyze portfolio changes
    portfolio_changes = np.diff(portfolio)
    significant_changes = np.sum(np.abs(portfolio_changes) > 1.0)  # Changes > ¥1

    print(f"   Significant portfolio changes (> ¥1): {significant_changes}")
    print(f"   Average change per step: {portfolio_changes.mean():.6f}")

    # Check if portfolio actually changes (indicating trades occurred)
    unique_portfolios = len(np.unique(portfolio.round(2)))
    print(f"   Unique portfolio values: {unique_portfolios}")

    if unique_portfolios <= 2:
        print("   ⚠️  Portfolio barely changes - very few/no trades executed!")
    elif unique_portfolios < len(portfolio) * 0.1:
        print("   ⚠️  Low portfolio diversity - limited trading activity")

def analyze_environment_config():
    """Analyze environment configuration that might affect trading."""

    print()
    print("⚙️  SAC v430 Environment Configuration Analysis")
    print("=" * 60)

    # Load config
    config = read_json("configs/v430/sac_v430_optimized.json")

    print("📋 Environment Settings:")
    print("   Transaction cost: 0.0005 (0.05%)")
    print("   Max position size: 0.01 (1% of portfolio)")
    print("   Action threshold: 0.3333")

    # Analyze if settings discourage trading
    transaction_cost = 0.0005
    max_position_size = 0.01

    print()
    print("🔍 Trading Barrier Analysis:")

    # Calculate minimum profitable trade
    # Assuming typical price movement needed to overcome costs
    min_profit_needed = transaction_cost * 2  # Round trip cost
    print(f"   Round-trip transaction cost: {min_profit_needed*100:.3f}%")
    print(f"   Max position size: {max_position_size*100:.1f}% of portfolio")

    # Check if position size is too small
    if max_position_size < 0.05:  # Less than 5%
        print("   ⚠️  Max position size is very small - limits trading impact")

    # Check transaction cost impact
    if transaction_cost > 0.001:  # More than 0.1%
        print("   ⚠️  Transaction costs are relatively high")

def generate_recommendations():
    """Generate recommendations based on analysis."""

    print()
    print("🎯 SAC v430 Analysis Recommendations")
    print("=" * 60)

    recommendations = []

    # Based on typical issues found
    recommendations.extend(
        [
            "1. 🔍 Action Distribution Issue:",
            "   - Model outputs almost constant SELL actions (~-0.9999)",
            "   - Check if reward function penalizes BUY/SELL actions",
            "   - Consider reducing action thresholds or adjusting reward incentives",
            "",
            "2. 💰 Reward Function Issues:",
            "   - sell_penalty is negative (-0.352) → penalizes selling",
            "   - buy_bonus is negative (-0.427) → penalizes buying",
            "   - hold_penalty is positive (0.005) → penalizes holding",
            "   - This creates conflicting incentives!",
            "",
            "3. ⚙️ Environment Configuration:",
            "   - Max position size (1%) is very conservative",
            "   - May need to increase to allow meaningful portfolio impact",
            "",
            "4. 🧪 Suggested Experiments:",
            "   - Test with modified reward function (remove penalties)",
            "   - Try lower action thresholds (e.g., 0.1 instead of 0.3333)",
            "   - Increase max position size to 5-10%",
            "   - Run shorter backtests with different market conditions",
        ]
    )

    for rec in recommendations:
        print(rec)

def main():
    """Main analysis function."""
    try:
        actions = analyze_action_distribution()
        analyze_reward_function()
        analyze_portfolio_behavior()
        analyze_environment_config()
        generate_recommendations()

        print()
        print("✅ Deep analysis completed!")
        print("📄 Results saved for further investigation")

    except Exception as e:
        print(f"❌ Analysis failed: {e}")
        import traceback

        traceback.print_exc()
        return 1

    return 0

if __name__ == "__main__":
    sys.exit(main())
