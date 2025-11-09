#!/usr/bin/env python3
"""
SAC v445.3 Backtest Script - Strong Selling Optimized

Backtest the trained SAC v445.3 model to evaluate SELL action improvements.
Measures actual profit in JPY and BTC.
"""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict

import numpy as np
import pandas as pd

# Add project root to path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from stable_baselines3 import PPO

from ztb.trading.environment.heavy_env.core import HeavyTradingEnv
from ztb.trading.environment.utils.config import EnvironmentConfig
from ztb.utils.logging_utils import get_logger

logger = get_logger(__name__)


class SACV445Backtester:
    """Backtester for SAC v445.3 model."""

    def __init__(self, config_path: str, model_path: str):
        """Initialize backtester with config and model."""
        self.config_path = config_path
        self.model_path = model_path
        self.config = self._load_config()
        self.model = None
        self.env = None
        self.results = {}

    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from JSON file."""
        with open(self.config_path, "r", encoding="utf-8") as f:
            return json.load(f)

    def _initialize_components(self):
        """Initialize model and environment."""
        # Load the trained model
        self.model = PPO.load(self.model_path)
        logger.info(f"Loaded model from {self.model_path}")

        # Load data for backtest
        data_config = self.config.get("training", {}).get("data_config", {})
        csv_path = data_config.get("data_path", "data/btc_jpy_real_dataset.csv")

        if not Path(csv_path).exists():
            raise FileNotFoundError(f"Data file not found: {csv_path}")

        df = pd.read_csv(csv_path)
        logger.info(f"Loaded data from {csv_path}, shape: {df.shape}")

        # Create environment config the same way as training
        env_config_dict = self.config["training"]["environment"].copy()
        env_config_dict["use_continuous_actions"] = True

        # Extract reward_scaling from reward_settings if nested
        if "reward_settings" in env_config_dict and isinstance(
            env_config_dict["reward_settings"], dict
        ):
            if "reward_scaling" in env_config_dict["reward_settings"]:
                env_config_dict["reward_scaling"] = float(
                    env_config_dict["reward_settings"]["reward_scaling"]
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

        # Create environment
        self.env = HeavyTradingEnv(
            df=df, config=env_config, use_continuous_actions=True
        )
        logger.info(
            f"Environment created with observation space: {self.env.observation_space}"
        )
        logger.info(f"Environment created with action space: {self.env.action_space}")

    def run_backtest(self) -> Dict[str, Any]:
        """Run the backtest and return results."""
        logger.info("Starting backtest...")

        # Initialize tracking variables
        obs, _ = self.env.reset()
        done = False
        total_steps = 0
        portfolio_values = []
        actions_taken = []
        rewards_received = []

        # Track portfolio state
        initial_portfolio_value = self.env.portfolio_value
        initial_position = self.env.position  # BTC position (-1 to 1)
        initial_jpy_balance = initial_portfolio_value * (
            1 - abs(initial_position)
        )  # Approximate JPY
        initial_btc_balance = (
            initial_portfolio_value
            * abs(initial_position)
            / self.env.df.iloc[0]["close"]
        )  # BTC amount

        logger.info(
            f"Initial portfolio: JPY={initial_jpy_balance:.2f}, BTC={initial_btc_balance:.8f}, Position={initial_position:.4f}, Total={initial_portfolio_value:.2f}"
        )

        while not done and total_steps < len(self.env.df):
            # Get action from model
            action, _ = self.model.predict(obs, deterministic=True)
            actions_taken.append(action[0])

            # Execute action in environment
            obs, reward, terminated, truncated, info = self.env.step(action)
            rewards_received.append(reward)

            # Track portfolio value
            portfolio_values.append(self.env.portfolio_value)

            done = terminated or truncated
            total_steps += 1

            if total_steps % 1000 == 0:
                logger.info(
                    f"Step {total_steps}: Portfolio value = {self.env.portfolio_value:.2f}"
                )

        # Calculate final results
        final_portfolio_value = self.env.portfolio_value
        final_position = self.env.position
        current_price = self.env.df.iloc[
            min(self.env.current_step, len(self.env.df) - 1)
        ]["close"]
        final_jpy_balance = final_portfolio_value * (1 - abs(final_position))
        final_btc_balance = final_portfolio_value * abs(final_position) / current_price

        # Calculate profits
        jpy_profit = final_jpy_balance - initial_jpy_balance
        btc_profit = final_btc_balance - initial_btc_balance
        total_profit = final_portfolio_value - initial_portfolio_value

        # Calculate returns
        total_return_pct = (
            (final_portfolio_value - initial_portfolio_value) / initial_portfolio_value
        ) * 100

        # Action statistics
        sell_actions = sum(1 for a in actions_taken if a < -0.3)
        buy_actions = sum(1 for a in actions_taken if a > 0.3)
        hold_actions = len(actions_taken) - sell_actions - buy_actions

        # Compile results
        self.results = {
            "backtest_info": {
                "model": "sac_v445.3_strong_selling_optimized",
                "config_file": self.config_path,
                "model_file": self.model_path,
                "timestamp": datetime.now().isoformat(),
                "total_steps": total_steps,
            },
            "initial_state": {
                "portfolio_value": initial_portfolio_value,
                "jpy_balance": initial_jpy_balance,
                "btc_balance": initial_btc_balance,
            },
            "final_state": {
                "portfolio_value": final_portfolio_value,
                "jpy_balance": final_jpy_balance,
                "btc_balance": final_btc_balance,
            },
            "profits": {
                "jpy_profit": jpy_profit,
                "btc_profit": btc_profit,
                "total_profit": total_profit,
                "total_return_pct": total_return_pct,
            },
            "action_statistics": {
                "total_actions": len(actions_taken),
                "sell_actions": sell_actions,
                "buy_actions": buy_actions,
                "hold_actions": hold_actions,
                "sell_percentage": (sell_actions / len(actions_taken)) * 100
                if actions_taken
                else 0,
                "buy_percentage": (buy_actions / len(actions_taken)) * 100
                if actions_taken
                else 0,
                "hold_percentage": (hold_actions / len(actions_taken)) * 100
                if actions_taken
                else 0,
            },
            "performance_metrics": {
                "total_reward": sum(rewards_received),
                "average_reward": sum(rewards_received) / len(rewards_received)
                if rewards_received
                else 0,
                "max_portfolio_value": max(portfolio_values) if portfolio_values else 0,
                "min_portfolio_value": min(portfolio_values)
                if portfolio_values
                else initial_portfolio_value,
            },
        }

        logger.info("Backtest completed successfully")
        return self.results

    def save_results(self, output_path: str):
        """Save backtest results to JSON file."""

        # Convert numpy types to Python types for JSON serialization
        def convert_numpy_types(obj):
            if isinstance(obj, (np.float32, np.float64)):
                return float(obj)
            elif isinstance(obj, (np.int32, np.int64)):
                return int(obj)
            elif isinstance(obj, dict):
                return {k: convert_numpy_types(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_numpy_types(item) for item in obj]
            else:
                return obj

        serializable_results = convert_numpy_types(self.results)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(serializable_results, f, indent=2, ensure_ascii=False)
        logger.info(f"Results saved to {output_path}")

    def print_summary(self):
        """Print a summary of the backtest results."""
        if not self.results:
            logger.error("No results to display. Run backtest first.")
            return

        print("\n" + "=" * 60)
        print("🎯 SAC v445.3 バックテスト結果")
        print("=" * 60)

        # Basic info
        print(f"モデル: {self.results['backtest_info']['model']}")
        print(f"ステップ数: {self.results['backtest_info']['total_steps']}")

        print("\n💰 ポートフォリオ推移:")
        print(f"  初期: ¥{self.results['initial_state']['portfolio_value']:,.2f}")
        print(f"  最終: ¥{self.results['final_state']['portfolio_value']:,.2f}")

        print("\n📈 利益:")
        print(f"  総利益: ¥{self.results['profits']['total_profit']:,.2f}")
        print(f"  総リターン: {self.results['profits']['total_return_pct']:+.2f}%")

        print("\n₿ BTC残高:")
        print(f"  初期: {self.results['initial_state']['btc_balance']:.8f} BTC")
        print(f"  最終: {self.results['final_state']['btc_balance']:.8f} BTC")
        print(f"  利益: {self.results['profits']['btc_profit']:+.8f} BTC")

        print("\n💴 JPY残高:")
        print(f"  初期: ¥{self.results['initial_state']['jpy_balance']:,.2f}")
        print(f"  最終: ¥{self.results['final_state']['jpy_balance']:,.2f}")
        print(f"  利益: ¥{self.results['profits']['jpy_profit']:,.2f}")

        print("\n🎯 アクション統計:")
        stats = self.results["action_statistics"]
        print(f"  SELL: {stats['sell_actions']} ({stats['sell_percentage']:.1f}%)")
        print(f"  BUY:  {stats['buy_actions']} ({stats['buy_percentage']:.1f}%)")
        print(f"  HOLD: {stats['hold_actions']} ({stats['hold_percentage']:.1f}%)")

        print("\n📊 パフォーマンス指標:")
        perf = self.results["performance_metrics"]
        print(f"  総報酬: {perf['total_reward']:,.2f}")
        print(f"  平均報酬: {perf['average_reward']:.4f}")
        print(f"  最高ポートフォリオ: ¥{perf['max_portfolio_value']:,.2f}")
        print(f"  最低ポートフォリオ: ¥{perf['min_portfolio_value']:,.2f}")

        print("=" * 60)


def main():
    """Main function to run the backtest."""
    parser = argparse.ArgumentParser(description="Backtest SAC v445.3 model")
    parser.add_argument(
        "--config",
        type=str,
        default="config/v445/sac_v445.3_strong_selling_optimized.json",
        help="Path to configuration file",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="models/sac_v445.3_strong_selling_optimized_final.zip",
        help="Path to trained model file",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="results/sac_v445.3_backtest_results.json",
        help="Path to save results",
    )

    args = parser.parse_args()

    # Initialize backtester
    backtester = SACV445Backtester(args.config, args.model)

    try:
        # Initialize components
        backtester._initialize_components()

        # Run backtest
        results = backtester.run_backtest()

        # Print summary
        backtester.print_summary()

        # Save results
        backtester.save_results(args.output)

        print(f"\n✅ バックテスト完了！結果を {args.output} に保存しました。")

    except Exception as e:
        logger.error(f"Backtest failed: {e}")
        raise


if __name__ == "__main__":
    main()
