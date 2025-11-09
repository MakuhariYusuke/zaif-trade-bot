#!/usr/bin/env python3
"""
Comprehensive Backtest Comparison: v445.3 vs v445.4
Compare Strong Selling Optimized vs Ultra Aggressive Selling
Enhanced with BTC tracking and data quality analysis
"""

import json
import os
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from stable_baselines3 import PPO

from ztb.trading.environment.heavy_env.core import HeavyTradingEnv
from ztb.trading.environment.utils.config import EnvironmentConfig


def analyze_data_quality(df: pd.DataFrame) -> dict:
    """Analyze data quality and market conditions."""
    analysis = {}

    # Price trend analysis
    price_changes = df["close"].pct_change()
    analysis["total_return"] = (df["close"].iloc[-1] / df["close"].iloc[0] - 1) * 100
    analysis["positive_days"] = (price_changes > 0).sum()
    analysis["negative_days"] = (price_changes < 0).sum()
    analysis["trend_ratio"] = analysis["positive_days"] / max(
        analysis["negative_days"], 1
    )

    # Volatility analysis
    analysis["volatility"] = (
        price_changes.std() * np.sqrt(252) * 100
    )  # Annualized volatility
    analysis["max_drawdown"] = (
        (df["close"] - df["close"].expanding().max()) / df["close"].expanding().max()
    ).min() * 100

    # Market regime detection
    analysis["is_uptrend_only"] = analysis["trend_ratio"] > 5.0  # Very bullish bias
    analysis["is_downtrend_only"] = analysis["trend_ratio"] < 0.2  # Very bearish bias
    analysis["is_balanced"] = 0.5 <= analysis["trend_ratio"] <= 2.0

    return analysis


def load_config(config_path: str) -> dict:
    """Load configuration from JSON file."""
    with open(config_path, "r", encoding="utf-8") as f:
        return json.load(f)
    with open(config_path, "r", encoding="utf-8") as f:
        return json.load(f)


def create_environment(config: dict) -> HeavyTradingEnv:
    """Create and configure the trading environment."""
    # Load data
    data_path = config["training"]["data_config"]["data_path"]
    df = pd.read_csv(data_path)

    # Create environment config
    env_config_dict = config["training"]["environment"].copy()
    env_config_dict["use_continuous_actions"] = True

    # Extract reward_scaling from reward_settings if nested
    if "reward_settings" in env_config_dict and isinstance(
        env_config_dict["reward_settings"], dict
    ):
        if "reward_scaling" in env_config_dict["reward_settings"]:
            env_config_dict["reward_scaling"] = float(
                env_config_dict["reward_settings"]["reward_scaling"]
            )
        elif "reward_scale" in env_config_dict["reward_settings"]:
            env_config_dict["reward_scaling"] = float(
                env_config_dict["reward_settings"]["reward_scale"]
            )

    # Remove reward_settings to avoid conflicts
    if "reward_settings" in env_config_dict:
        del env_config_dict["reward_settings"]

    # Convert initial_balance to initial_portfolio_value if needed
    if "initial_balance" in env_config_dict:
        env_config_dict["initial_portfolio_value"] = env_config_dict.pop(
            "initial_balance"
        )

    # Remove fields that don't exist in EnvironmentConfig
    fields_to_remove = [
        "feature_engineering",
        "market_regime_detection",
        "risk_management",
        "multi_timeframe_integration",
        "behavior_optimization",
    ]
    for field in fields_to_remove:
        env_config_dict.pop(field, None)

    env_config = EnvironmentConfig(**env_config_dict)
    env = HeavyTradingEnv(df=df, config=env_config, use_continuous_actions=True)

    return env


def run_backtest(model_path: str, config_path: str, model_name: str) -> dict:
    """Run backtest for a single model with enhanced BTC tracking."""
    print(f"\n🔄 Running backtest for {model_name}")
    print("-" * 50)

    # Load config and create environment
    config = load_config(config_path)
    env = create_environment(config)

    # Analyze data quality
    data_analysis = analyze_data_quality(env.df)
    print("📊 Data Quality Analysis:")
    print(f"   Total Return: {data_analysis['total_return']:.2f}%")
    print(
        f"   Positive/Negative Days: {data_analysis['positive_days']}/{data_analysis['negative_days']}"
    )
    print(f"   Trend Ratio: {data_analysis['trend_ratio']:.2f}")
    print(f"   Annualized Volatility: {data_analysis['volatility']:.1f}%")
    print(
        f"⚠️  WARNING: Data shows {'uptrend-only' if data_analysis['is_uptrend_only'] else 'balanced'} market conditions"
    )
    print("   This may bias results toward BUY-only strategies")

    # Load model
    model = PPO.load(model_path)

    # Run backtest with enhanced tracking
    obs, _ = env.reset()
    done = False
    total_reward = 0
    trades = []
    portfolio_values = []
    btc_holdings = []
    actions_taken = []

    initial_btc = config["training"]["environment"].get("initial_btc", 0.0)
    current_btc = initial_btc

    step = 0
    while not done and step < len(env.df) - 1:
        action, _ = model.predict(obs, deterministic=True)
        action = int(action)  # Ensure action is integer
        action = max(0, min(2, action))  # Clamp action to valid range [0, 2]
        obs, reward, terminated, truncated, info = env.step(action)

        total_reward += reward
        portfolio_values.append(env.portfolio_value)

        # Track BTC holdings
        if hasattr(env, "btc_balance"):
            current_btc = env.btc_balance
        elif hasattr(env, "position"):
            current_btc = env.position
        btc_holdings.append(current_btc)

        actions_taken.append(int(action))

        # Record trades with BTC info
        if hasattr(env, "position") and hasattr(env, "entry_price"):
            prev_position = getattr(env, "_previous_position", 0)
            if abs(env.position - prev_position) > 1e-6:
                trade_type = "BUY" if env.position > prev_position else "SELL"
                btc_change = env.position - prev_position
                trades.append(
                    {
                        "step": step,
                        "type": trade_type,
                        "price": env.df.iloc[step]["close"],
                        "btc_change": btc_change,
                        "portfolio_value": env.portfolio_value,
                        "timestamp": env.df.iloc[step]["timestamp"]
                        if "timestamp" in env.df.columns
                        else step,
                    }
                )

        step += 1
        if step % 1000 == 0:
            print(
                f"Step {step}/{len(env.df)-1}: Portfolio = ¥{env.portfolio_value:,.0f}, BTC = {current_btc:.6f}"
            )

        done = terminated or truncated

    # Calculate final metrics
    final_value = (
        portfolio_values[-1]
        if len(portfolio_values) > 0
        else config["training"]["environment"].get("initial_balance", 10000)
    )
    initial_value = config["training"]["environment"].get("initial_balance", 10000)
    total_return = (final_value - initial_value) / initial_value * 100

    final_btc = btc_holdings[-1] if len(btc_holdings) > 0 else initial_btc
    btc_return = (
        (final_btc - initial_btc) / max(initial_btc, 1e-8) * 100
        if initial_btc > 0
        else 0
    )

    # Action distribution
    action_counts = np.bincount(actions_taken, minlength=3)
    action_distribution = {
        "HOLD": int(action_counts[0]),
        "BUY": int(action_counts[1]),
        "SELL": int(action_counts[2]),
    }

    # Calculate Sharpe ratio
    returns = np.diff(portfolio_values) / portfolio_values[:-1]
    if len(returns) > 0 and np.std(returns) > 0:
        sharpe_ratio = np.mean(returns) / np.std(returns) * np.sqrt(252)
    else:
        sharpe_ratio = 0.0

    # Calculate drawdown
    portfolio_values = np.array(portfolio_values)
    peak = np.maximum.accumulate(portfolio_values)
    drawdown = (peak - portfolio_values) / peak * 100
    max_drawdown = np.max(drawdown)

    # Trade analysis
    buy_trades = [t for t in trades if t["type"] == "BUY"]
    sell_trades = [t for t in trades if t["type"] == "SELL"]

    results = {
        "model_name": model_name,
        "initial_portfolio": initial_value,
        "final_portfolio": final_value,
        "total_return_pct": total_return,
        "initial_btc": initial_btc,
        "final_btc": final_btc,
        "btc_return_pct": btc_return,
        "total_reward": total_reward,
        "total_trades": len(trades),
        "buy_trades": len(buy_trades),
        "sell_trades": len(sell_trades),
        "max_drawdown_pct": max_drawdown,
        "sharpe_ratio": sharpe_ratio,
        "action_distribution": action_distribution,
        "steps_completed": step,
        "portfolio_values": portfolio_values.tolist(),
        "btc_holdings": btc_holdings,
        "trades": trades,
        "data_quality": data_analysis,
    }

    print(f"✅ {model_name} backtest completed")
    print(f"📊 Final Portfolio Value: ${final_value:.2f}")
    print(f"🪙 Final BTC Holdings: {final_btc:.6f} BTC ({btc_return:+.2f}%)")
    print(f"📈 Total Return: {total_return:.2f}%")
    print(f"📊 Total Trades: {len(trades)}")
    print(f"📊 Sharpe Ratio: {sharpe_ratio:.1f}")
    return results


def compare_models():
    """Compare v445.3 and v445.4 models."""
    print("🔍 SAC v445.3 vs v445.4 Comprehensive Backtest Comparison")
    print("=" * 70)

    # Model configurations
    models = [
        {
            "name": "v445.3_strong_selling",
            "model_path": "models/sac_v445.3_strong_selling_optimized_final.zip",
            "config_path": "config/v445/sac_v445.3_strong_selling_optimized.json",
        },
        {
            "name": "v445.4_ultra_aggressive",
            "model_path": "models/sac_v445.4_ultra_aggressive_selling_final.zip",
            "config_path": "config/v445/sac_v445.4_ultra_aggressive_selling.json",
        },
    ]

    results = {}

    # Run backtests
    for model_config in models:
        if os.path.exists(model_config["model_path"]) and os.path.exists(
            model_config["config_path"]
        ):
            results[model_config["name"]] = run_backtest(
                model_config["model_path"],
                model_config["config_path"],
                model_config["name"],
            )
        else:
            print(f"❌ Model or config not found: {model_config['name']}")

    if len(results) < 2:
        print("❌ Need both models for comparison")
        return

    # Comparison analysis
    print("\n" + "=" * 70)
    print("📊 COMPREHENSIVE COMPARISON RESULTS")
    print("=" * 70)

    v3_results = results["v445.3_strong_selling"]
    v4_results = results["v445.4_ultra_aggressive"]

    # Key metrics comparison
    metrics = [
        ("Final Portfolio Value", "final_portfolio", "¥{:,.0f}"),
        ("Total Return %", "total_return_pct", "{:.2f}%"),
        ("Total Reward", "total_reward", "{:.2f}"),
        ("Max Drawdown %", "max_drawdown_pct", "{:.2f}%"),
        ("Sharpe Ratio", "sharpe_ratio", "{:.3f}"),
        ("Total Trades", "total_trades", "{:d}"),
        ("Buy Trades", "buy_trades", "{:d}"),
        ("Sell Trades", "sell_trades", "{:d}"),
    ]

    print("\n📈 Key Performance Metrics:")
    print("-" * 50)
    for metric_name, key, format_str in metrics:
        v3_val = v3_results[key]
        v4_val = v4_results[key]
        diff = v4_val - v3_val
        diff_symbol = "↑" if diff > 0 else "↓" if diff < 0 else "="
        print("<15")

    # Action distribution comparison
    print("\n🎯 Action Distribution:")
    print("-" * 50)
    actions = ["HOLD", "BUY", "SELL"]
    for action in actions:
        v3_count = v3_results["action_distribution"][action]
        v4_count = v4_results["action_distribution"][action]
        v3_pct = v3_count / sum(v3_results["action_distribution"].values()) * 100
        v4_pct = v4_count / sum(v4_results["action_distribution"].values()) * 100
        diff_pct = v4_pct - v3_pct
        diff_symbol = "↑" if diff_pct > 0 else "↓" if diff_pct < 0 else "="
        print("<6")

    # Trade analysis
    print("\n💼 Trade Analysis:")
    print("-" * 50)

    # Calculate trade frequencies
    v3_steps = v3_results["steps_completed"]
    v4_steps = v4_results["steps_completed"]

    v3_trade_freq = (
        v3_results["total_trades"] / v3_steps * 1000
    )  # trades per 1000 steps
    v4_trade_freq = v4_results["total_trades"] / v4_steps * 1000

    print(f"📊 v445.3 Trade Frequency: {v3_trade_freq:.2f} trades/1000 steps")
    print(f"📊 v445.4 Trade Frequency: {v4_trade_freq:.2f} trades/1000 steps")
    # Buy/Sell ratio
    if v3_results["sell_trades"] > 0:
        v3_buy_sell_ratio = v3_results["buy_trades"] / v3_results["sell_trades"]
    else:
        v3_buy_sell_ratio = float("inf")

    if v4_results["sell_trades"] > 0:
        v4_buy_sell_ratio = v4_results["buy_trades"] / v4_results["sell_trades"]
    else:
        v4_buy_sell_ratio = float("inf")

    print(f"📊 v445.3 Buy/Sell Ratio: {v3_buy_sell_ratio:.2f}")
    print(f"📊 v445.4 Buy/Sell Ratio: {v4_buy_sell_ratio:.2f}")
    # Risk analysis
    print("\n⚠️ Risk Analysis:")
    print("-" * 50)
    print(f"📊 v445.3 Max Drawdown: {v3_results['max_drawdown_pct']:.2f}%")
    print(f"📊 v445.4 Max Drawdown: {v4_results['max_drawdown_pct']:.2f}%")
    # Conclusion
    print("\n🎯 CONCLUSION:")
    print("-" * 50)

    if v4_results["total_return_pct"] > v3_results["total_return_pct"]:
        print("✅ v445.4 Ultra Aggressive outperforms v445.3 in total returns")
    elif v4_results["total_return_pct"] < v3_results["total_return_pct"]:
        print("✅ v445.3 Strong Selling outperforms v445.4 in total returns")
    else:
        print("⚖️ Both models show similar performance")

    # Horizontal expansion: Multi-dimensional analysis
    print("\n🔍 HORIZONTAL EXPANSION ANALYSIS")
    print("=" * 50)

    # Time-based analysis
    print("\n📅 TIME-BASED PERFORMANCE ANALYSIS")
    for model_name, result in results.items():
        portfolio_values = np.array(result["portfolio_values"])
        trades = result["trades"]

        # Split into quarters (assuming ~1000 steps per quarter)
        quarter_size = len(portfolio_values) // 4
        quarterly_returns = []

        for i in range(4):
            start_idx = i * quarter_size
            end_idx = min((i + 1) * quarter_size, len(portfolio_values))
            if end_idx > start_idx:
                quarter_values = portfolio_values[start_idx:end_idx]
                if len(quarter_values) > 1:
                    quarter_return = (
                        (quarter_values[-1] - quarter_values[0])
                        / quarter_values[0]
                        * 100
                    )
                    quarterly_returns.append(quarter_return)

        print(f"\n{model_name} Quarterly Performance:")
        for i, ret in enumerate(quarterly_returns):
            print(f"  Q{i+1}: {ret:+.2f}%")

        # Volatility analysis
        if len(portfolio_values) > 1:
            returns = np.diff(portfolio_values) / portfolio_values[:-1]
            volatility = np.std(returns) * np.sqrt(252) * 100  # Annualized volatility
            print(f"  Annualized Volatility: {volatility:.1f}%")

    # Market condition analysis
    print("\n🌊 MARKET CONDITION ANALYSIS")
    for model_name, result in results.items():
        trades = result["trades"]
        portfolio_values = np.array(result["portfolio_values"])

        # Analyze performance during different market phases
        # This is a simplified analysis - in practice you'd need market trend detection
        total_steps = len(portfolio_values)
        if total_steps > 10:
            # Simulate market phases (this should be based on actual trend analysis)
            uptrend_periods = int(total_steps * 0.6)  # Assume 60% uptrend
            downtrend_periods = int(total_steps * 0.3)  # Assume 30% downtrend
            sideways_periods = total_steps - uptrend_periods - downtrend_periods

            print(f"\n{model_name} Market Phase Performance:")
            print(f"  Assumed Uptrend Periods: {uptrend_periods} steps")
            print(f"  Assumed Downtrend Periods: {downtrend_periods} steps")
            print(f"  Assumed Sideways Periods: {sideways_periods} steps")

            # Trade frequency by phase (simplified)
            total_trades = len(trades)
            print(f"  Total Trades: {total_trades}")
            print(f"  Buy Trades: {result['buy_trades']}")
            print(f"  Sell Trades: {result['sell_trades']}")

    # Risk-adjusted return analysis
    print("\n📊 RISK-ADJUSTED RETURN ANALYSIS")
    for model_name, result in results.items():
        total_return = result["total_return_pct"]
        max_drawdown = result["max_drawdown_pct"]
        sharpe_ratio = result["sharpe_ratio"]

        # Calmar ratio (return / max drawdown)
        calmar_ratio = total_return / max_drawdown if max_drawdown > 0 else float("inf")

        # Sortino ratio (using simplified downside deviation)
        portfolio_values = np.array(result["portfolio_values"])
        returns = np.diff(portfolio_values) / portfolio_values[:-1]
        downside_returns = returns[returns < 0]
        downside_deviation = (
            np.std(downside_returns) if len(downside_returns) > 0 else 0
        )
        sortino_ratio = (
            np.mean(returns) / downside_deviation * np.sqrt(252)
            if downside_deviation > 0
            else float("inf")
        )

        print(f"\n{model_name} Risk Metrics:")
        print(f"  Sharpe Ratio: {sharpe_ratio:.2f}")
        print(f"  Calmar Ratio: {calmar_ratio:.2f}")
        print(f"  Sortino Ratio: {sortino_ratio:.2f}")
        print(f"  Max Drawdown: {max_drawdown:.2f}%")

    # BTC accumulation analysis
    print("\n🪙 BTC ACCUMULATION ANALYSIS")
    for model_name, result in results.items():
        initial_btc = result["initial_btc"]
        final_btc = result["final_btc"]
        btc_return = result["btc_return_pct"]

        print(f"\n{model_name} BTC Performance:")
        print(f"  Initial BTC: {initial_btc:.6f}")
        print(f"  Final BTC: {final_btc:.6f}")
        print(f"  BTC Return: {btc_return:+.2f}%")
        print(f"  Net BTC Gained: {final_btc - initial_btc:+.6f}")

        # BTC vs USD performance comparison
        usd_return = result["total_return_pct"]
        print(f"  USD Return: {usd_return:+.2f}%")
        if abs(btc_return) > 0.01:  # Avoid division by very small numbers
            btc_vs_usd_ratio = (
                usd_return / btc_return if btc_return != 0 else float("inf")
            )
            print(f"  BTC/USD Performance Ratio: {btc_vs_usd_ratio:.2f}")

    # Trade timing analysis
    print("\n⏰ TRADE TIMING ANALYSIS")
    for model_name, result in results.items():
        trades = result["trades"]
        if len(trades) > 1:
            trade_intervals = []
            for i in range(1, len(trades)):
                interval = trades[i]["step"] - trades[i - 1]["step"]
                trade_intervals.append(interval)

            if trade_intervals:
                avg_interval = np.mean(trade_intervals)
                min_interval = np.min(trade_intervals)
                max_interval = np.max(trade_intervals)

                print(f"\n{model_name} Trade Timing:")
                print(f"  Average Steps Between Trades: {avg_interval:.1f}")
                print(f"  Min Steps Between Trades: {min_interval}")
                print(f"  Max Steps Between Trades: {max_interval}")
                print(
                    f"  Trading Frequency: {len(trades) / result['steps_completed'] * 100:.2f}% of steps"
                )

    # Comparative advantage analysis
    print("\n⚖️ COMPARATIVE ADVANTAGE ANALYSIS")
    v3_results = results["v445.3_strong_selling"]
    v4_results = results["v445.4_ultra_aggressive"]

    # Multi-criteria comparison
    criteria = {
        "Return": (v4_results["total_return_pct"] - v3_results["total_return_pct"]),
        "Risk (lower better)": (
            v3_results["max_drawdown_pct"] - v4_results["max_drawdown_pct"]
        ),
        "Sell Trades": (v4_results["sell_trades"] - v3_results["sell_trades"]),
        "Sharpe Ratio": (v4_results["sharpe_ratio"] - v3_results["sharpe_ratio"]),
        "BTC Return": (v4_results["btc_return_pct"] - v3_results["btc_return_pct"]),
    }

    print("v445.4 vs v445.3 Advantage:")
    for criterion, advantage in criteria.items():
        if "lower better" in criterion:
            status = "✅ Better" if advantage > 0 else "❌ Worse"
        else:
            status = "✅ Better" if advantage > 0 else "❌ Worse"
        print(f"  {criterion}: {advantage:+.2f} ({status})")

    # Overall recommendation
    positive_criteria = sum(1 for adv in criteria.values() if adv > 0)
    total_criteria = len(criteria)

    if positive_criteria >= total_criteria * 0.6:  # 60% or more criteria better
        recommendation = "v445.4 (Ultra-aggressive selling configuration)"
    else:
        recommendation = "v445.3 (Balanced configuration)"

    print(f"\n🎯 RECOMMENDATION: {recommendation} shows better overall performance")
    print(f"   ({positive_criteria}/{total_criteria} criteria favorable to v445.4)")

    if v4_results["max_drawdown_pct"] > v3_results["max_drawdown_pct"]:
        print("⚠️ v445.4 shows higher risk (greater drawdown)")
    elif v4_results["max_drawdown_pct"] < v3_results["max_drawdown_pct"]:
        print("✅ v445.4 shows lower risk (smaller drawdown)")

    if v4_results["sell_trades"] > v3_results["sell_trades"]:
        print("🎯 v445.4 executes more SELL trades as intended")
    else:
        print("🤔 v445.4 does not show expected increase in SELL trades")

    # Save detailed results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = f"results/v445_comparison_backtest_{timestamp}.json"

    comparison_results = {
        "timestamp": timestamp,
        "models_compared": list(results.keys()),
        "detailed_results": results,
        "summary": {
            "winner_by_return": "v445.4"
            if v4_results["total_return_pct"] > v3_results["total_return_pct"]
            else "v445.3",
            "winner_by_risk": "v445.4"
            if v4_results["max_drawdown_pct"] < v3_results["max_drawdown_pct"]
            else "v445.3",
            "v445.4_sell_advantage": v4_results["sell_trades"]
            - v3_results["sell_trades"],
        },
    }

    with open(output_file, "w") as f:
        json.dump(comparison_results, f, indent=2, default=str)

    print(f"\n💾 Detailed results saved to: {output_file}")


if __name__ == "__main__":
    compare_models()
