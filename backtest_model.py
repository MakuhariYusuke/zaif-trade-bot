#!/usr/bin/env python3
"""
Backtest Script for Zaif Trade Bot.

Tests a trained PPO model using historical BTC/JPY data.
"""

import argparse
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, cast

import numpy as np
import pandas as pd
from stable_baselines3 import PPO

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from ztb.trading.environment import HeavyTradingEnv


def calculate_metrics(
    trades: List[Dict[str, Any]], initial_capital: float = 10000.0
) -> Dict[str, Any]:
    """Calculate comprehensive trading metrics."""
    if not trades:
        return {
            "total_return": 0.0,
            "annual_return": 0.0,
            "sharpe_ratio": 0.0,
            "max_drawdown": 0.0,
            "win_rate": 0.0,
            "total_trades": 0,
            "avg_trade_return": 0.0,
            "profit_factor": 0.0,
        }

    # Calculate returns
    capital = initial_capital
    capital_history = [capital]
    returns = []

    for trade in trades:
        pnl = trade.get("pnl", 0)
        capital += pnl
        capital_history.append(capital)
        if capital > initial_capital:
            returns.append((capital - initial_capital) / initial_capital)

    # Calculate metrics
    total_return = (capital - initial_capital) / initial_capital
    annual_return = total_return  # Simplified for now

    # Sharpe ratio (simplified)
    if returns:
        sharpe_ratio = np.mean(returns) / (np.std(returns) + 1e-6) * np.sqrt(252)
    else:
        sharpe_ratio = 0.0

    # Max drawdown
    capital_history = np.array(capital_history)
    peak = np.maximum.accumulate(capital_history)
    drawdown = (capital_history - peak) / peak
    max_drawdown = np.min(drawdown)

    # Win rate
    winning_trades = [t for t in trades if t.get("pnl", 0) > 0]
    win_rate = len(winning_trades) / len(trades) if trades else 0.0

    # Average trade return
    avg_trade_return = np.mean([t.get("pnl", 0) for t in trades]) if trades else 0.0

    # Profit factor
    gross_profit = sum(t.get("pnl", 0) for t in trades if t.get("pnl", 0) > 0)
    gross_loss = abs(sum(t.get("pnl", 0) for t in trades if t.get("pnl", 0) < 0))
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else float("inf")

    return {
        "total_return": total_return * 100,
        "annual_return": annual_return * 100,
        "sharpe_ratio": sharpe_ratio,
        "max_drawdown": max_drawdown * 100,
        "win_rate": win_rate * 100,
        "total_trades": len(trades),
        "avg_trade_return": avg_trade_return,
        "profit_factor": profit_factor,
        "final_capital": capital,
    }


def run_backtest(
    model_path: str,
    data_path: str,
    initial_capital: float = 10000.0,
    transaction_cost: float = 0.0005,
    max_steps: Optional[int] = None,
    verbose: bool = True,
) -> Dict[str, Any]:
    """Run backtest simulation."""
    logging.basicConfig(level=logging.INFO if verbose else logging.WARNING)
    logger = logging.getLogger(__name__)

    # Load model
    logger.info(f"Loading model from {model_path}")
    model = PPO.load(model_path)

    # Load data
    logger.info(f"Loading data from {data_path}")
    df = pd.read_csv(data_path)
    logger.info(f"Loaded {len(df)} rows of data")

    # Limit data if specified
    if max_steps and len(df) > max_steps:
        df = df.head(max_steps)
        logger.info(f"Limited to {max_steps} steps")

    # Create environment
    config = {
        "transaction_cost": transaction_cost,
        "enable_correlation_reduction": True,
        "correlation_threshold": 0.95,
        "max_position_size": 0.5,
        "reward_trade_frequency_penalty": 0.01,
        "reward_trade_frequency_halflife": 1.0,
        "reward_trade_cooldown_steps": 0,
        "reward_trade_cooldown_penalty": 0.01,
        "reward_max_consecutive_trades": 20,
        "reward_consecutive_trade_penalty": 0.01,
        "reward_position_penalty_scale": 0.1,
        "reward_position_penalty_exponent": 2.0,
        "reward_inventory_penalty_scale": 0.01,
        "reward_volatility_penalty_scale": 0.01,
    }

    env = HeavyTradingEnv(
        df=df,
        config=config,
        random_start=False,
    )

    # Run backtest
    logger.info("Starting backtest simulation")
    obs, _ = env.reset()
    done = False
    trades = []
    last_position = 0
    entry_price = 0
    entry_time = 0

    step = 0
    actions_taken = []

    while not done:
        action, _ = model.predict(obs, deterministic=True)
        action = cast(int, action.item() if hasattr(action, "item") else action)
        actions_taken.append(action)

        obs, _, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        step += 1

        # Track position changes for trade recording
        current_position = env.position
        if step % 100 == 0:  # Debug output every 100 steps
            logger.info(f"Step {step}: action={action}, position={current_position}")

        if abs(current_position) > 0 and abs(last_position) == 0:  # Opening a position
            entry_price = env.df.iloc[min(env.current_step, len(env.df) - 1)]["close"]
            entry_time = env.current_step
            logger.info(f"Opened position at step {step}: {current_position}")
        elif (
            abs(last_position) > 0 and abs(current_position) == 0
        ):  # Closing a position
            exit_price = env.df.iloc[min(env.current_step, len(env.df) - 1)]["close"]
            pnl = (
                (exit_price - entry_price) * last_position * (1 - transaction_cost * 2)
            )
            trade = {
                "entry_time": entry_time,
                "exit_time": env.current_step,
                "entry_price": entry_price,
                "exit_price": exit_price,
                "pnl": pnl,
                "position": last_position,
            }
            trades.append(trade)
            logger.info(f"Closed position at step {step}: pnl={pnl}")

        last_position = current_position

        # Safety check to prevent infinite loops
        if step > 10000:
            logger.warning("Simulation exceeded 10000 steps, terminating")
            break

    logger.info(f"Simulation completed in {step} steps")
    logger.info(
        f"Actions distribution: {pd.Series(actions_taken).value_counts().to_dict()}"
    )

    # Calculate final metrics
    metrics = calculate_metrics(trades, initial_capital)

    logger.info("Backtest completed")
    logger.info(f"Total Return: {metrics['total_return']:.2f}%")
    logger.info(f"Win Rate: {metrics['win_rate']:.2f}%")
    logger.info(f"Total Trades: {metrics['total_trades']}")
    logger.info(f"Sharpe Ratio: {metrics['sharpe_ratio']:.2f}")

    return {
        "metrics": metrics,
        "trades": trades,
        "config": {
            "model_path": model_path,
            "data_path": data_path,
            "initial_capital": initial_capital,
            "transaction_cost": transaction_cost,
            "max_steps": max_steps,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run backtest for trained model")
    parser.add_argument(
        "--model-path",
        required=True,
        help="Path to trained model (.zip)",
    )
    parser.add_argument(
        "--data-path",
        default="ml-dataset-enhanced.csv",
        help="Path to historical data (default: ml-dataset-enhanced.csv)",
    )
    parser.add_argument(
        "--initial-capital",
        type=float,
        default=10000.0,
        help="Initial capital (default: 10000.0)",
    )
    parser.add_argument(
        "--transaction-cost",
        type=float,
        default=0.0005,
        help="Transaction cost as fraction (default: 0.0005)",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=None,
        help="Maximum steps to simulate (default: all data)",
    )
    parser.add_argument(
        "--output",
        help="Output file for results (JSON format)",
    )

    args = parser.parse_args()

    # Run backtest
    results = run_backtest(
        model_path=args.model_path,
        data_path=args.data_path,
        initial_capital=args.initial_capital,
        transaction_cost=args.transaction_cost,
        max_steps=args.max_steps,
    )

    # Print results
    print("\n=== Backtest Results ===")
    metrics = results["metrics"]
    for key, value in metrics.items():
        if isinstance(value, float):
            print(f"{key}: {value:.4f}")
        else:
            print(f"{key}: {value}")

    # Save results if requested
    if args.output:
        import json

        with open(args.output, "w") as f:
            json.dump(results, f, indent=2, default=str)
        print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()  # type: ignore[no-untyped-call]
