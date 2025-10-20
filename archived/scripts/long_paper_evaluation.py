"""
Long Paper Evaluation - Final Acceptance Test.

Comprehensive evaluation of trained model with production-grade metrics:
- Sharpe Ratio (target > 0.5)
- Maximum Drawdown (target < 30%)
- Legal SELL Rate (target >= 15%)
- Regime Stability (no failures across market conditions)
- Trade Statistics (frequency, P&L, win rate)
- Risk Metrics (volatility, Sortino, Calmar)

Evaluation Length: >= 500 steps (representative sample)
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

import argparse
import json
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import numpy as np
from sb3_contrib import MaskablePPO

# 年間取引日数
from ztb.trading.constants import TRADING_DAYS_PER_YEAR  # = 252
from ztb.trading.environment.environment import EnvironmentConfig, HeavyTradingEnv
from ztb.utils.data_utils import load_csv_data_optimized


def calculate_sharpe_ratio(returns: np.ndarray, risk_free_rate: float = 0.0) -> float:
    """Calculate Sharpe ratio."""
    if len(returns) == 0:
        return 0.0

    excess_returns = returns - risk_free_rate
    if np.std(excess_returns) == 0:
        return 0.0

    return float(
        np.mean(excess_returns)
        / np.std(excess_returns)
        * np.sqrt(TRADING_DAYS_PER_YEAR)
    )


def calculate_sortino_ratio(returns: np.ndarray, risk_free_rate: float = 0.0) -> float:
    """Calculate Sortino ratio (downside deviation)."""
    if len(returns) == 0:
        return 0.0

    excess_returns = returns - risk_free_rate
    downside_returns = excess_returns[excess_returns < 0]

    if len(downside_returns) == 0 or np.std(downside_returns) == 0:
        return 0.0

    return float(
        np.mean(excess_returns)
        / np.std(downside_returns)
        * np.sqrt(TRADING_DAYS_PER_YEAR)
    )


def calculate_max_drawdown(equity_curve: np.ndarray) -> Tuple[float, int, int]:
    """
    Calculate maximum drawdown.

    Returns:
        (max_drawdown_pct, start_idx, end_idx)
    """
    if len(equity_curve) == 0:
        return 0.0, 0, 0

    running_max = np.maximum.accumulate(equity_curve)
    drawdown = (equity_curve - running_max) / running_max

    max_dd_idx = np.argmin(drawdown)
    max_dd = float(drawdown[max_dd_idx])

    # Find start of drawdown
    start_idx = 0
    for i in range(max_dd_idx, -1, -1):
        if equity_curve[i] == running_max[i]:
            start_idx = i
            break

    return abs(max_dd) * 100, start_idx, max_dd_idx


def calculate_calmar_ratio(returns: np.ndarray, equity_curve: np.ndarray) -> float:
    """Calculate Calmar ratio (annualized return / max drawdown)."""
    if len(returns) == 0 or len(equity_curve) == 0:
        return 0.0

    annualized_return = np.mean(returns) * TRADING_DAYS_PER_YEAR
    max_dd, _, _ = calculate_max_drawdown(equity_curve)

    if max_dd == 0:
        return 0.0

    return float(annualized_return / (max_dd / 100))


def evaluate_regime_stability(
    action_history: List[int],
    reward_history: List[float],
    window_size: int = 100,
) -> Dict[str, float]:
    """
    Evaluate stability across different market regimes.

    Args:
        action_history: List of actions taken
        reward_history: List of rewards received
        window_size: Window for regime detection

    Returns:
        Dict with regime stability metrics
    """
    if len(action_history) < window_size * 2:
        return {
            "regime_count": 1,
            "avg_regime_sharpe": 0.0,
            "regime_sharpe_std": 0.0,
            "min_regime_sharpe": 0.0,
        }

    # Split into windows
    n_windows = len(action_history) // window_size
    regime_sharpes = []

    for i in range(n_windows):
        start = i * window_size
        end = start + window_size
        window_rewards = reward_history[start:end]

        if len(window_rewards) > 0:
            sharpe = calculate_sharpe_ratio(np.array(window_rewards))
            regime_sharpes.append(sharpe)

    return {
        "regime_count": len(regime_sharpes),
        "avg_regime_sharpe": float(np.mean(regime_sharpes)) if regime_sharpes else 0.0,
        "regime_sharpe_std": float(np.std(regime_sharpes)) if regime_sharpes else 0.0,
        "min_regime_sharpe": float(np.min(regime_sharpes)) if regime_sharpes else 0.0,
    }


def long_paper_evaluation(
    model_path: Path,
    data_path: Path,
    output_path: Path,
    min_steps: int = 500,
    config: Optional[dict] = None,
):
    """
    Run comprehensive Long Paper evaluation.

    Args:
        model_path: Path to trained model
        data_path: Path to evaluation data
        output_path: Path to save results JSON
        min_steps: Minimum evaluation steps
        config: Optional environment config
    """
    print("=" * 60)
    print("Long Paper Evaluation - Final Acceptance Test")
    print("=" * 60)
    print(f"Model: {model_path}")
    print(f"Data: {data_path}")
    print(f"Output: {output_path}")
    print(f"Min steps: {min_steps}")
    print()

    # Load model
    print("Loading model...")
    model = MaskablePPO.load(str(model_path))
    print("  ✅ Model loaded")
    print()

    # Load data
    print("Loading evaluation data...")
    df = load_csv_data_optimized(data_path)
    print(f"  Loaded {len(df)} rows, {len(df.columns)} columns")
    print()

    # Create environment
    print("Creating environment...")
    env_config = EnvironmentConfig.from_dict(config or {})
    env = HeavyTradingEnv(df=df, config=env_config)
    print("  ✅ Environment created")
    print()

    # Run evaluation
    print(f"Running evaluation ({min_steps}+ steps)...")
    obs, _ = env.reset()
    done = False

    # Tracking
    action_history = []
    reward_history = []
    equity_curve = [env.portfolio_value]
    legal_sell_count = 0
    total_legal_steps = 0
    action_counts = {0: 0, 1: 0, 2: 0}
    trade_pnls = []
    last_position = 0.0
    steps = 0

    while not done and steps < len(df):
        # Get action mask
        action_mask = env.action_masks()

        # Predict
        action, _ = model.predict(obs, action_masks=action_mask, deterministic=True)
        action = int(action)

        # Track
        action_history.append(action)
        action_counts[action] += 1

        if action_mask[2]:  # SELL is legal
            total_legal_steps += 1
            if action == 2:
                legal_sell_count += 1

        # Step
        obs, reward, done, truncated, info = env.step(action)
        reward_history.append(reward)
        equity_curve.append(env.portfolio_value)

        # Track trades
        current_position = getattr(env, "position", 0.0)
        if current_position != last_position:
            # Position changed = trade occurred
            if len(reward_history) > 1:
                trade_pnls.append(reward_history[-1])
        last_position = current_position

        done = done or truncated
        steps += 1

    print(f"  ✅ Evaluation complete ({steps} steps)")
    print()

    # Calculate metrics
    print("Calculating metrics...")

    returns = np.array(reward_history)
    equity = np.array(equity_curve)

    sharpe = calculate_sharpe_ratio(returns)
    sortino = calculate_sortino_ratio(returns)
    calmar = calculate_calmar_ratio(returns, equity)
    max_dd, dd_start, dd_end = calculate_max_drawdown(equity)
    legal_sell_rate = (
        legal_sell_count / total_legal_steps if total_legal_steps > 0 else 0.0
    )

    total_return = (equity[-1] - equity[0]) / equity[0] * 100
    avg_reward = np.mean(returns)
    reward_volatility = np.std(returns)

    win_rate = (
        len([p for p in trade_pnls if p > 0]) / len(trade_pnls) if trade_pnls else 0.0
    )
    avg_win = (
        np.mean([p for p in trade_pnls if p > 0])
        if any(p > 0 for p in trade_pnls)
        else 0.0
    )
    avg_loss = (
        np.mean([p for p in trade_pnls if p < 0])
        if any(p < 0 for p in trade_pnls)
        else 0.0
    )
    profit_factor = abs(avg_win / avg_loss) if avg_loss != 0 else 0.0

    regime_metrics = evaluate_regime_stability(action_history, reward_history)

    # Compile results
    results = {
        "timestamp": datetime.now().isoformat(),
        "model_path": str(model_path),
        "data_path": str(data_path),
        "evaluation_steps": steps,
        # Performance Metrics
        "sharpe_ratio": sharpe,
        "sortino_ratio": sortino,
        "calmar_ratio": calmar,
        "total_return_pct": total_return,
        "avg_reward": avg_reward,
        "reward_volatility": reward_volatility,
        # Risk Metrics
        "max_drawdown_pct": max_dd,
        "drawdown_start_step": dd_start,
        "drawdown_end_step": dd_end,
        "drawdown_duration": dd_end - dd_start,
        # Action Metrics
        "legal_sell_rate": legal_sell_rate,
        "action_distribution": {
            "hold": action_counts[0],
            "buy": action_counts[1],
            "sell": action_counts[2],
            "hold_pct": action_counts[0] / steps * 100,
            "buy_pct": action_counts[1] / steps * 100,
            "sell_pct": action_counts[2] / steps * 100,
        },
        # Trade Metrics
        "total_trades": len(trade_pnls),
        "win_rate": win_rate,
        "avg_win": avg_win,
        "avg_loss": avg_loss,
        "profit_factor": profit_factor,
        # Regime Stability
        "regime_stability": regime_metrics,
        # Acceptance Criteria
        "acceptance": {
            "sharpe_target": 0.5,
            "sharpe_pass": sharpe > 0.5,
            "sell_target": 0.15,
            "sell_pass": legal_sell_rate >= 0.15,
            "mdd_target": 30.0,
            "mdd_pass": max_dd < 30.0,
            "regime_stable": regime_metrics["min_regime_sharpe"] > 0.0,
            "overall_pass": (
                sharpe > 0.5
                and legal_sell_rate >= 0.15
                and max_dd < 30.0
                and regime_metrics["min_regime_sharpe"] > 0.0
            ),
        },
    }

    # Save results
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)

    print("  ✅ Metrics calculated")
    print()

    # Print results
    print("=" * 60)
    print("EVALUATION RESULTS")
    print("=" * 60)
    print()

    print("Performance Metrics:")
    print(
        f"  Sharpe Ratio: {sharpe:.3f} (target: >0.5) {'✅' if results['acceptance']['sharpe_pass'] else '❌'}"
    )
    print(f"  Sortino Ratio: {sortino:.3f}")
    print(f"  Calmar Ratio: {calmar:.3f}")
    print(f"  Total Return: {total_return:+.2f}%")
    print()

    print("Risk Metrics:")
    print(
        f"  Max Drawdown: {max_dd:.2f}% (target: <30%) {'✅' if results['acceptance']['mdd_pass'] else '❌'}"
    )
    print(f"  Reward Volatility: {reward_volatility:.4f}")
    print()

    print("Action Metrics:")
    print(
        f"  Legal SELL Rate: {legal_sell_rate:.1%} (target: >=15%) {'✅' if results['acceptance']['sell_pass'] else '❌'}"
    )
    print("  Action Distribution:")
    print(f"    HOLD: {action_counts[0]} ({action_counts[0]/steps*100:.1f}%)")
    print(f"    BUY:  {action_counts[1]} ({action_counts[1]/steps*100:.1f}%)")
    print(f"    SELL: {action_counts[2]} ({action_counts[2]/steps*100:.1f}%)")
    print()

    print("Trade Metrics:")
    print(f"  Total Trades: {len(trade_pnls)}")
    print(f"  Win Rate: {win_rate:.1%}")
    print(f"  Profit Factor: {profit_factor:.2f}")
    print()

    print("Regime Stability:")
    print(f"  Regimes Analyzed: {regime_metrics['regime_count']}")
    print(f"  Avg Regime Sharpe: {regime_metrics['avg_regime_sharpe']:.3f}")
    print(
        f"  Min Regime Sharpe: {regime_metrics['min_regime_sharpe']:.3f} {'✅' if results['acceptance']['regime_stable'] else '❌'}"
    )
    print()

    print("=" * 60)
    print("FINAL ACCEPTANCE")
    print("=" * 60)
    if results["acceptance"]["overall_pass"]:
        print("✅ PASS - Model ready for production deployment")
    else:
        print("❌ FAIL - Model requires further training")
        print("\nFailed criteria:")
        if not results["acceptance"]["sharpe_pass"]:
            print(f"  - Sharpe ratio ({sharpe:.3f}) below target (0.5)")
        if not results["acceptance"]["sell_pass"]:
            print(f"  - SELL rate ({legal_sell_rate:.1%}) below target (15%)")
        if not results["acceptance"]["mdd_pass"]:
            print(f"  - Max drawdown ({max_dd:.2f}%) above target (30%)")
        if not results["acceptance"]["regime_stable"]:
            print("  - Regime instability detected")
    print()

    print(f"Results saved to: {output_path}")

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Long Paper evaluation - final acceptance test"
    )
    parser.add_argument(
        "--model",
        type=Path,
        required=True,
        help="Path to trained model",
    )
    parser.add_argument(
        "--data",
        type=Path,
        default=Path("ml-dataset-final.csv"),
        help="Evaluation data CSV",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/long_paper_results.json"),
        help="Output JSON path",
    )
    parser.add_argument(
        "--min-steps",
        type=int,
        default=500,
        help="Minimum evaluation steps",
    )
    parser.add_argument(
        "--config",
        type=Path,
        help="Optional environment config JSON",
    )

    args = parser.parse_args()

    # Load config if provided
    config = None
    if args.config and args.config.exists():
        with open(args.config) as f:
            config = json.load(f)

    long_paper_evaluation(
        model_path=args.model,
        data_path=args.data,
        output_path=args.output,
        min_steps=args.min_steps,
        config=config,
    )


if __name__ == "__main__":
    main()
