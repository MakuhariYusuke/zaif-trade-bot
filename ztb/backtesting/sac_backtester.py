#!/usr/bin/env python3
"""
SAC Backtest Suite - Comprehensive backtesting tools for SAC models

This script provides unified backtesting capabilities for SAC trading models including:
- Single model backtesting
- Multi-model comparison
- Regime-specific analysis
- Performance evaluation
- Risk analysis
- Benchmark comparison
"""

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

# Add project root to path
project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(project_root))

from ztb.utils.logging_utils import get_logger
from ztb.utils.path_utils import get_project_root

# Get project root using utility
project_root = get_project_root()

logger = get_logger(__name__)


@dataclass
class BacktestResult:
    """Container for backtest results."""

    performance_metrics: Dict[str, float]
    trade_log: List[Dict[str, Any]]
    regime_analysis: Dict[str, Any]
    risk_metrics: Dict[str, float]
    benchmark_comparison: Dict[str, Any]


class SACBacktester:
    """Comprehensive SAC model backtester."""

    def __init__(
        self, model_path: Optional[str] = None, config_path: Optional[str] = None
    ):
        """
        Initialize SAC backtester.

        Args:
            model_path: Path to SAC model file
            config_path: Path to configuration file
        """
        self.model_path = Path(model_path) if model_path else None
        self.config_path = Path(config_path) if config_path else None
        self.model = None
        self.config = None
        self.env = None

        if self.model_path and self.model_path.exists():
            self.load_model()

        if self.config_path and self.config_path.exists():
            self.load_config()

    def load_model(self) -> bool:
        """Load SAC model."""
        try:
            self.model = SAC.load(str(self.model_path))
            logger.info(f"Model loaded from {self.model_path}")
            return True
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            return False

    def load_config(self) -> bool:
        """Load configuration."""
        try:
            with open(self.config_path, "r", encoding="utf-8") as f:
                self.config = json.load(f)
            logger.info(f"Config loaded from {self.config_path}")
            return True
        except Exception as e:
            logger.error(f"Failed to load config: {e}")
            return False

    def create_environment(self, data_path: str):
        """Create trading environment for backtesting."""
        try:
            from ztb.trading.environment.heavy_env.core import HeavyTradingEnv
            from ztb.trading.environment.utils.config import EnvironmentConfig

            # Load data
            df = pd.read_csv(data_path)
            logger.info(f"Loaded backtest data: {len(df)} rows from {data_path}")

            # Create environment config
            env_config = EnvironmentConfig(
                max_position_size=self.config.get("max_position_size", 1.0),
                transaction_cost=self.config.get("transaction_cost", 0.0),
                reward_scaling=self.config.get("reward_scaling", 1.0),
            )

            # Create environment
            self.env = HeavyTradingEnv(config=env_config, data=df, mode="backtest")

            return True

        except Exception as e:
            logger.error(f"Failed to create environment: {e}")
            return False

    def run_backtest(
        self, data_path: str, num_episodes: int = 1, deterministic: bool = True
    ) -> BacktestResult:
        """
        Run backtest on the SAC model.

        Args:
            data_path: Path to test data
            num_episodes: Number of episodes to run
            deterministic: Whether to use deterministic policy

        Returns:
            Backtest results
        """
        if not self.model:
            raise ValueError("Model not loaded")

        if not self.create_environment(data_path):
            raise ValueError("Failed to create environment")

        logger.info(f"Starting backtest with {num_episodes} episodes")

        all_trades = []
        episode_results = []

        for episode in range(num_episodes):
            logger.info(f"Running episode {episode + 1}/{num_episodes}")

            obs = self.env.reset()
            done = False
            episode_trades = []
            step_count = 0

            while not done and step_count < len(self.env.data) - 1:
                # Get action from model
                action, _ = self.model.predict(obs, deterministic=deterministic)

                # Step environment
                next_obs, reward, done, info = self.env.step(action)

                # Record trade if any
                if info.get("trade_executed", False):
                    trade_info = {
                        "step": step_count,
                        "timestamp": info.get("timestamp"),
                        "action": float(action[0]),
                        "price": info.get("price", 0),
                        "position": info.get("position", 0),
                        "pnl": info.get("pnl", 0),
                        "reward": float(reward),
                    }
                    episode_trades.append(trade_info)

                obs = next_obs
                step_count += 1

            all_trades.extend(episode_trades)
            episode_results.append(
                {
                    "episode": episode,
                    "trades": len(episode_trades),
                    "total_steps": step_count,
                }
            )

        # Calculate performance metrics
        performance_metrics = self._calculate_performance_metrics(all_trades)

        # Regime analysis
        regime_analysis = self._analyze_regimes(all_trades)

        # Risk metrics
        risk_metrics = self._calculate_risk_metrics(all_trades)

        # Benchmark comparison
        benchmark_comparison = self._compare_with_benchmark(all_trades)

        result = BacktestResult(
            performance_metrics=performance_metrics,
            trade_log=all_trades,
            regime_analysis=regime_analysis,
            risk_metrics=risk_metrics,
            benchmark_comparison=benchmark_comparison,
        )

        logger.info(f"Backtest completed: {len(all_trades)} trades executed")
        return result

    def _calculate_performance_metrics(
        self, trades: List[Dict[str, Any]]
    ) -> Dict[str, float]:
        """Calculate performance metrics from trades."""
        if not trades:
            return {"total_return": 0.0, "win_rate": 0.0, "total_trades": 0}

        pnl_values = [trade["pnl"] for trade in trades]
        total_return = sum(pnl_values)
        win_rate = sum(1 for pnl in pnl_values if pnl > 0) / len(pnl_values)
        total_trades = len(trades)

        # Calculate Sharpe ratio (simplified)
        if len(pnl_values) > 1:
            returns_std = np.std(pnl_values)
            sharpe_ratio = total_return / returns_std if returns_std > 0 else 0.0
        else:
            sharpe_ratio = 0.0

        return {
            "total_return": total_return,
            "win_rate": win_rate,
            "total_trades": total_trades,
            "sharpe_ratio": sharpe_ratio,
            "avg_trade_pnl": total_return / total_trades if total_trades > 0 else 0.0,
        }

    def _analyze_regimes(self, trades: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze performance across different market regimes."""
        # Placeholder for regime analysis
        return {
            "bull_market_performance": 0.0,
            "bear_market_performance": 0.0,
            "sideways_performance": 0.0,
            "high_volatility_performance": 0.0,
        }

    def _calculate_risk_metrics(self, trades: List[Dict[str, Any]]) -> Dict[str, float]:
        """Calculate risk metrics from trades."""
        if not trades:
            return {"max_drawdown": 0.0, "volatility": 0.0}

        pnl_values = [trade["pnl"] for trade in trades]
        cumulative_pnl = np.cumsum(pnl_values)

        # Max drawdown
        peak = np.maximum.accumulate(cumulative_pnl)
        drawdown = peak - cumulative_pnl
        max_drawdown = np.max(drawdown) if len(drawdown) > 0 else 0.0

        # Volatility
        volatility = np.std(pnl_values) if len(pnl_values) > 1 else 0.0

        return {
            "max_drawdown": max_drawdown,
            "volatility": volatility,
            "var_95": np.percentile(pnl_values, 5) if len(pnl_values) > 0 else 0.0,
        }

    def _compare_with_benchmark(self, trades: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Compare performance with benchmark (buy and hold)."""
        # Placeholder for benchmark comparison
        return {"vs_buy_hold": 0.0, "benchmark_return": 0.0, "outperformance": 0.0}

    def print_report(self, result: BacktestResult):
        """Print formatted backtest report."""
        print("\n" + "=" * 60)
        print("SAC MODEL BACKTEST REPORT")
        print("=" * 60)

        print("\n📊 PERFORMANCE METRICS:")
        print(".2f")
        print(".1%")
        print(f"  Total Trades: {result.performance_metrics['total_trades']}")
        print(".3f")
        print(".3f")

        print("\n⚠️  RISK METRICS:")
        print(".4f")
        print(".4f")
        print(".4f")

        print("\n📈 TRADE LOG SUMMARY:")
        print(f"  Total Trades Executed: {len(result.trade_log)}")

        if result.trade_log:
            actions = [trade["action"] for trade in result.trade_log]
            buy_actions = sum(1 for a in actions if a > 0.3333)
            sell_actions = sum(1 for a in actions if a < -0.3333)
            hold_actions = sum(1 for a in actions if -0.3333 <= a <= 0.3333)

            print(f"  BUY Actions: {buy_actions}")
            print(f"  SELL Actions: {sell_actions}")
            print(f"  HOLD Actions: {hold_actions}")

        print("\n" + "=" * 60)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="SAC Backtest Suite")
    parser.add_argument(
        "--model", type=str, required=True, help="Path to SAC model file"
    )
    parser.add_argument(
        "--data", type=str, required=True, help="Path to test data (CSV)"
    )
    parser.add_argument("--config", type=str, help="Path to configuration file")
    parser.add_argument(
        "--episodes", type=int, default=1, help="Number of episodes to run"
    )
    parser.add_argument(
        "--deterministic", action="store_true", help="Use deterministic policy"
    )
    parser.add_argument("--output", type=str, help="Output file for results (JSON)")

    args = parser.parse_args()

    # Initialize backtester
    backtester = SACBacktester(args.model, args.config)

    # Run backtest
    result = backtester.run_backtest(args.data, args.episodes, args.deterministic)

    # Print report
    backtester.print_report(result)

    # Save results if requested
    if args.output:
        output_data = {
            "performance_metrics": result.performance_metrics,
            "trade_log": result.trade_log,
            "regime_analysis": result.regime_analysis,
            "risk_metrics": result.risk_metrics,
            "benchmark_comparison": result.benchmark_comparison,
        }

        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)

        print(f"\n📄 Results saved to: {args.output}")


if __name__ == "__main__":
    main()
