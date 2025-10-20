#!/usr/bin/env python3
"""
Backtest Model Script for SAC Trading Models
"""

import json
import os
import sys

import pandas as pd

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)
# 年間取引日数
from ztb.trading.constants import TRADING_DAYS_PER_YEAR  # = 252


def run_backtest(model_path, data_path, output_path=None):
    """Run backtest for a given model and data"""
    try:
        import numpy as np
        from stable_baselines3 import SAC

        from ztb.trading.environment.heavy_env.core import HeavyTradingEnv
        from ztb.trading.environment.utils.config import EnvironmentConfig

        # Load data
        df = pd.read_csv(data_path)
        print(f"Loaded data: {len(df)} rows")

        # Create environment config similar to training
        env_config_dict = {
            "initial_portfolio_value": 1000000.0,
            "transaction_cost": 0.001,
            "max_position_size": 1.0,
            "enable_action_masking": False,
            "use_continuous_actions": True,
            "use_standardized_observations": True,
            "random_start": False,  # For backtesting, start from beginning
            "curriculum_stage": "baseline",
            "continuous_to_discrete_threshold": 0.1,
        }
        env_config = EnvironmentConfig.from_dict(env_config_dict)

        # Create HeavyTradingEnv for backtesting
        env = HeavyTradingEnv(df=df, config=env_config)
        print(f"Created HeavyTradingEnv with {len(env.features)} features")

        # Load model
        model = SAC.load(model_path)
        print(f"Loaded model: {model_path}")

        # Run backtest
        obs, info = env.reset()
        done = False
        total_reward = 0
        trades = []
        step = 0

        while not done and step < len(df) - 1:
            action, _ = model.predict(obs, deterministic=True)
            if step < 5:  # Debug first 5 steps
                print(f"Debug: Step {step}, Obs shape: {obs.shape}, Action: {action}")
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            total_reward += reward
            step += 1

            # Record trade if position changed or pnl occurred
            if (
                info.get("position", 0) != 0
                or (
                    hasattr(env, "position_manager")
                    and env.position_manager.get_position_info()["trades_count"] > 0
                )
                or info.get("pnl", 0) != 0
            ):
                trade_info = {
                    "step": step,
                    "action": action.tolist() if hasattr(action, "tolist") else action,
                    "discrete_action": info.get("discrete_action", 0),
                    "reward": float(reward),
                    "position": float(info.get("position", 0)),
                    "entry_price": float(info.get("entry_price", 0)),
                    "pnl": float(info.get("pnl", 0)),
                    "portfolio_value": float(
                        info.get("portfolio_value", env.initial_portfolio_value)
                    ),
                }
                trades.append(trade_info)

        # Calculate metrics
        if trades:
            pnl_values = [t["pnl"] for t in trades if t["pnl"] != 0]
            total_return = sum(pnl_values) if pnl_values else 0
            win_rate = (
                sum(1 for pnl in pnl_values if pnl > 0) / len(pnl_values) * 100
                if pnl_values
                else 0
            )
            max_drawdown = min(pnl_values) if pnl_values else 0

            # Calculate Sharpe ratio (simplified)
            returns = np.array(pnl_values)
            if len(returns) > 1:
                sharpe_ratio = (
                    np.mean(returns) / np.std(returns) * np.sqrt(TRADING_DAYS_PER_YEAR)
                    if np.std(returns) > 0
                    else 0
                )
            else:
                sharpe_ratio = 0
        else:
            total_return = 0
            win_rate = 0
            max_drawdown = 0
            sharpe_ratio = 0

        results = {
            "metrics": {
                "total_return": float(total_return),
                "sharpe_ratio": float(sharpe_ratio),
                "win_rate": float(win_rate),
                "max_drawdown": float(max_drawdown),
                "total_trades": len(trades),
            },
            "trades": trades,
            "total_steps": step,
            "model_path": model_path,
            "data_path": data_path,
        }

        if output_path:
            with open(output_path, "w") as f:
                json.dump(results, f, indent=2)
            print(f"Results saved to {output_path}")

        return results

    except Exception as e:
        print(f"Error in backtest: {e}")
        import traceback

        traceback.print_exc()
        return None


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Backtest SAC trading model")
    parser.add_argument("--model", required=True, help="Path to model file")
    parser.add_argument("--data", required=True, help="Path to data CSV file")
    parser.add_argument("--output", help="Output JSON file path")

    args = parser.parse_args()

    results = run_backtest(args.model, args.data, args.output)

    if results:
        print("\n=== BACKTEST RESULTS ===")
        metrics = results["metrics"]
        print(f"Total Return: {metrics['total_return']:.2f}%")
        print(f"Sharpe Ratio: {metrics['sharpe_ratio']:.2f}")
        print(f"Win Rate: {metrics['win_rate']:.2f}%")
        print(f"Max Drawdown: {metrics['max_drawdown']:.2f}%")
        print(f"Total Trades: {metrics['total_trades']}")
    else:
        sys.exit(1)
