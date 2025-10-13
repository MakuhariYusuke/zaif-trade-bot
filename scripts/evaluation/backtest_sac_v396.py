#!/usr/bin/env python3
"""
Backtest Script for SAC v396 Model.

Tests the trained SAC model (50k steps) using historical BTC/JPY data.
"""

import argparse
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, cast

import numpy as np
from numpy.typing import NDArray
import pandas as pd
from sb3_contrib import RecurrentPPO
from stable_baselines3 import SAC

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from ztb.utils.file_utils import safe_json_dump
from ztb.utils.performance_utils import timed
from ztb.trading.environment.environment import HeavyTradingEnv


@timed
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
            "final_capital": initial_capital,
        }

    # Calculate returns
    capital = initial_capital
    capital_history = [capital]
    returns = []

    for trade in trades:
        pnl = trade.get("pnl", 0)
        capital += pnl
        capital_history.append(capital)
        if len(capital_history) > 1:
            ret = (capital_history[-1] - capital_history[-2]) / capital_history[-2]
            returns.append(ret)

    # Calculate metrics
    total_return = (capital - initial_capital) / initial_capital
    
    # Sharpe ratio (annualized, assuming 252 trading days)
    if returns and len(returns) > 1:
        sharpe_ratio = np.mean(returns) / (np.std(returns) + 1e-8) * np.sqrt(252)
    else:
        sharpe_ratio = 0.0

    # Max drawdown
    capital_history_array: NDArray[np.float64] = np.array(capital_history)
    peak = np.maximum.accumulate(capital_history_array)
    drawdown = (capital_history_array - peak) / (peak + 1e-8)
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
        "sharpe_ratio": sharpe_ratio,
        "max_drawdown": max_drawdown * 100,
        "win_rate": win_rate * 100,
        "total_trades": len(trades),
        "avg_trade_return": avg_trade_return,
        "profit_factor": profit_factor,
        "final_capital": capital,
        "gross_profit": gross_profit,
        "gross_loss": gross_loss,
    }


def run_backtest(
    model_path: str,
    data_path: str,
    initial_capital: float = 10000.0,
    transaction_cost: float = 0.0005,
    max_steps: Optional[int] = None,
    deterministic: bool = True,
    verbose: bool = True,
) -> Dict[str, Any]:
    """Run backtest simulation with SAC model."""
    logging.basicConfig(level=logging.INFO if verbose else logging.WARNING)
    logger = logging.getLogger(__name__)

    # Load model
    logger.info(f"Loading SAC model from {model_path}")
    try:
        model = SAC.load(model_path)
        logger.info("Successfully loaded SAC model")
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        raise

    # Load data
    logger.info(f"Loading data from {data_path}")
    df = pd.read_csv(data_path)
    logger.info(f"Loaded {len(df)} rows of data")

    # Limit data if specified
    if max_steps and len(df) > max_steps:
        df = df.head(max_steps)
        logger.info(f"Limited to {max_steps} steps")

    # Create environment with SAC v396 configuration
    # IMPORTANT: Must match training configuration (continuous actions)
    config = {
        "initial_balance": 200000,
        "transaction_cost": transaction_cost,
        "max_position_size": 0.01,
        "enable_action_masking": False,
        "use_continuous_actions": True,  # Critical for SAC!
        "use_standardized_observations": True,
        "continuous_to_discrete_threshold": 0.33,  # Default threshold
        "reward_settings": {
            "use_simple_reward": True,
            "reward_scale": 100.0,
            "reward_clip_min": -1.0,
            "reward_clip_max": 1.0,
            "enable_inactivity_penalty": True,
            "inactivity_penalty_rate": 0.001,
            "enable_opportunity_cost": True,
            "opportunity_cost_rate": 0.0005,
        },
    }

    env = HeavyTradingEnv(
        df=df,
        config=config,
        random_start=False,
    )

    # Run backtest
    logger.info(f"Starting backtest simulation (deterministic={deterministic})")
    obs, _ = env.reset()
    done = False
    trades = []
    last_position = 0
    entry_price = 0
    entry_time = 0

    step = 0
    actions_taken = []
    discrete_actions_taken = []
    episode_rewards = []
    current_episode_reward = 0.0

    while not done:
        # Get action from SAC model
        action, _ = model.predict(obs, deterministic=deterministic)
        
        # SAC outputs continuous actions
        # Track the continuous action value for analysis
        if isinstance(action, np.ndarray):
            continuous_action_value = action.item()
        else:
            continuous_action_value = action
        
        actions_taken.append(continuous_action_value)

        obs, reward, terminated, truncated, _ = env.step(action)
        current_episode_reward += reward
        done = terminated or truncated
        step += 1

        # Track the discrete action that was actually executed
        # (environment converts continuous to discrete internally)
        from ztb.trading.environment.constants import continuous_to_discrete_action
        discrete_action = continuous_to_discrete_action(continuous_action_value)
        discrete_actions_taken.append(discrete_action)

        # Track position changes for trade recording
        current_position = env.position
        if step % 500 == 0:  # Debug output every 500 steps
            logger.info(f"Step {step}/{len(df)}: continuous_action={continuous_action_value:.3f}, discrete_action={discrete_action}, position={current_position:.4f}, reward={reward:.4f}")

        # Detect position changes
        # Note: Using threshold of 0.001 (instead of 0.01) to detect small positions
        position_threshold = 0.001
        
        # 1. Opening new position from flat
        if abs(current_position) > position_threshold and abs(last_position) <= position_threshold:
            entry_price = env.df.iloc[min(env.current_step, len(env.df) - 1)]["close"]
            entry_time = env.current_step
            logger.debug(f"Opened position at step {step}: {current_position:.4f}")
        
        # 2. Closing position to flat
        elif abs(last_position) > position_threshold and abs(current_position) <= position_threshold:
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
                "duration": env.current_step - entry_time,
            }
            trades.append(trade)
            logger.debug(f"Closed position at step {step}: pnl={pnl:.2f}")
        
        # 3. Position reversal (Long→Short or Short→Long)
        elif (last_position > position_threshold and current_position < -position_threshold) or (last_position < -position_threshold and current_position > position_threshold):
            # Close previous position
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
                "duration": env.current_step - entry_time,
            }
            trades.append(trade)
            logger.debug(f"Reversed position at step {step}: {last_position:.4f}→{current_position:.4f}, pnl={pnl:.2f}")
            
            # Open new reversed position
            entry_price = exit_price
            entry_time = env.current_step

        last_position = current_position

        # Safety check to prevent infinite loops
        if step > len(df) + 100:
            logger.warning(f"Simulation exceeded expected steps ({len(df)}), terminating")
            break

    logger.info(f"Simulation completed in {step} steps")
    
    # Action statistics for continuous SAC actions
    actions_array = np.array(actions_taken)
    logger.info(f"Continuous action statistics: mean={np.mean(actions_array):.4f}, std={np.std(actions_array):.4f}, min={np.min(actions_array):.4f}, max={np.max(actions_array):.4f}")
    
    # Discrete action distribution
    discrete_actions_array = np.array(discrete_actions_taken)
    unique, counts = np.unique(discrete_actions_array, return_counts=True)
    discrete_dist = dict(zip(unique, counts))
    logger.info(f"Discrete action distribution: HOLD={discrete_dist.get(0, 0)}, BUY={discrete_dist.get(1, 0)}, SELL={discrete_dist.get(2, 0)}")
    logger.info(f"Action percentages: HOLD={discrete_dist.get(0, 0)/len(discrete_actions_taken)*100:.1f}%, BUY={discrete_dist.get(1, 0)/len(discrete_actions_taken)*100:.1f}%, SELL={discrete_dist.get(2, 0)/len(discrete_actions_taken)*100:.1f}%")

    # Calculate final metrics
    metrics = calculate_metrics(trades, initial_capital)

    logger.info("\n=== Backtest Results ===")
    logger.info(f"Total Return: {metrics['total_return']:.2f}%")
    logger.info(f"Sharpe Ratio: {metrics['sharpe_ratio']:.2f}")
    logger.info(f"Max Drawdown: {metrics['max_drawdown']:.2f}%")
    logger.info(f"Win Rate: {metrics['win_rate']:.2f}%")
    logger.info(f"Total Trades: {metrics['total_trades']}")
    logger.info(f"Avg Trade Return: {metrics['avg_trade_return']:.2f}")
    logger.info(f"Profit Factor: {metrics['profit_factor']:.2f}")
    logger.info(f"Final Capital: {metrics['final_capital']:.2f}")

    return {
        "metrics": metrics,
        "trades": trades,
        "config": {
            "model_path": model_path,
            "data_path": data_path,
            "initial_capital": initial_capital,
            "transaction_cost": transaction_cost,
            "max_steps": max_steps,
            "deterministic": deterministic,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run backtest for SAC v396 model")
    parser.add_argument(
        "--model-path",
        default="checkpoints/sac_session/sac_v396_50k_final.zip",
        help="Path to trained SAC model (.zip)",
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
        "--deterministic",
        action="store_true",
        default=True,
        help="Use deterministic actions (default: True)",
    )
    parser.add_argument(
        "--output",
        default="backtest_sac_v396_results.json",
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
        deterministic=args.deterministic,
    )

    # Print results
    print("\n" + "="*60)
    print("=== SAC v396 (50k Steps) Backtest Results ===")
    print("="*60)
    metrics = results["metrics"]
    for key, value in metrics.items():
        if isinstance(value, float):
            print(f"{key:25s}: {value:12.4f}")
        else:
            print(f"{key:25s}: {value:12}")
    print("="*60)

    # Save results if requested
    if args.output:
        output_path = Path(args.output)
        safe_json_dump(results, output_path, indent=2, default=str)
        print(f"\n✅ Results saved to {args.output}")


if __name__ == "__main__":
    main()
