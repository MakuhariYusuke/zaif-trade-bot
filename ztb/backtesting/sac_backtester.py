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
import sys
from pathlib import Path
import json
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any 
from dataclasses import dataclass

# Add project root to path
project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(project_root))

from ztb.utils.path_utils import get_project_root
from ztb.utils.logging_utils import get_logger
from ztb.utils.safety import safe_open_json, ensure_dict

# Import SAC with fallback for type checking
try:
    from stable_baselines3 import SAC
except Exception:
    SAC = None

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

    def __init__(self, model_path: Optional[str] = None, config_path: Optional[str] = None):
        """
        Initialize SAC backtester.

        Args:
            model_path: Path to SAC model file
            config_path: Path to configuration file
        """
        self.model_path = Path(model_path) if model_path else None
        self.config_path = Path(config_path) if config_path else None
        from typing import Any, Dict, Optional

        self.model: Optional[Any] = None
        self.config: Optional[Dict[str, Any]] = None
        self.env: Optional[Any] = None

        if self.model_path and self.model_path.exists():
            self.load_model()

        if self.config_path and self.config_path.exists():
            self.load_config()

    def load_model(self) -> bool:
        """Load SAC model."""
        try:
            if SAC is None:
                raise RuntimeError("stable_baselines3.SAC not available in runtime")
            self.model = SAC.load(str(self.model_path))
            logger.info(f"Model loaded from {self.model_path}")
            return True
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            return False

    def load_config(self) -> bool:
        """Load configuration."""
        try:
            cfg = safe_open_json(self.config_path)
            if cfg is None:
                raise FileNotFoundError("Failed to open config or parse JSON")
            self.config = cfg
            logger.info(f"Config loaded from {self.config_path}")
            return True
        except Exception as e:
            logger.error(f"Failed to load config: {e}")
            return False

    def create_environment(self, data_path: str) -> bool:
        """Create trading environment for backtesting."""
        try:
            from ztb.trading.environment.heavy_env.core import HeavyTradingEnv
            from ztb.trading.environment.utils.config import EnvironmentConfig

            # Load data
            df = pd.read_csv(data_path)
            logger.info(f"Loaded backtest data: {len(df)} rows from {data_path}")

            # Create environment config
            # Normalize config via shared utility
            cfg = ensure_dict(self.config)
            env_config = EnvironmentConfig(
                max_position_size=cfg.get('max_position_size', 1.0),
                transaction_cost=cfg.get('transaction_cost', 0.0),
                reward_scaling=cfg.get('reward_scaling', 1.0)
            )

            # Create environment (use same signature as other code paths)
            self.env = HeavyTradingEnv(df=df, config=env_config)

            return True

        except Exception as e:
            logger.error(f"Failed to create environment: {e}")
            return False

    def run_backtest(self, data_path: str, num_episodes: int = 1,
                    deterministic: bool = True) -> BacktestResult:
        """
        Run backtest on the SAC model.

        Args:
            data_path: Path to test data
            num_episodes: Number of episodes to run
            deterministic: Whether to use deterministic policy

        Returns:
            Backtest results
        """
        # Narrow model to local variable and guard None to satisfy static analysis
        model = self.model
        if model is None:
            raise ValueError("Model not loaded")

        # create_environment may return False on failure; handle that case.
        if not self.create_environment(data_path):
            raise ValueError("Failed to create environment")

        logger.info(f"Starting backtest with {num_episodes} episodes")

        all_trades = []
        episode_results = []

        for episode in range(num_episodes):
            logger.info(f"Running episode {episode + 1}/{num_episodes}")
            # Narrow env and guard
            env = self.env
            if env is None:
                raise ValueError("Environment not initialized for backtest")

            obs = env.reset()
            done = False
            episode_trades = []
            step_count = 0

            while not done and step_count < len(env.data) - 1:
                # Get action from model
                # Use local model reference
                action, _ = model.predict(obs, deterministic=deterministic)

                # Step environment (use local env)
                step_result = env.step(action)
                # Expect common signature (obs, reward, done, info)
                try:
                    next_obs, reward, done, info = step_result
                except Exception:
                    logger.warning("Unexpected env.step() return signature; aborting episode")
                    break

                # Record trade if any; normalize info to dict for safe access
                if not isinstance(info, dict):
                    try:
                        info = dict(info)
                    except Exception:
                        info = {}

                if info.get('trade_executed', False):
                    # Safely convert action to float
                    try:
                        action_value = float(action[0]) if hasattr(action, '__getitem__') else float(action)
                    except Exception:
                        try:
                            action_value = float(action)
                        except Exception:
                            action_value = 0.0

                    trade_info = {
                        'step': step_count,
                        'timestamp': info.get('timestamp'),
                        'action': action_value,
                        'price': info.get('price', 0),
                        'position': info.get('position', 0),
                        'pnl': info.get('pnl', 0),
                        'reward': float(reward)
                    }
                    episode_trades.append(trade_info)

                obs = next_obs
                step_count += 1

            all_trades.extend(episode_trades)
            episode_results.append({
                'episode': episode,
                'trades': len(episode_trades),
                'total_steps': step_count
            })

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
            benchmark_comparison=benchmark_comparison
        )

        logger.info("Backtest completed: %d trades executed", len(all_trades))
        return result

    def _calculate_performance_metrics(self, trades: List[Dict[str, Any]]) -> Dict[str, float]:
        """Calculate performance metrics from trades."""
        if not trades:
            return {'total_return': 0.0, 'win_rate': 0.0, 'total_trades': 0}

        pnl_values = [trade['pnl'] for trade in trades]
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
            'total_return': float(total_return),
            'win_rate': float(win_rate),
            'total_trades': float(total_trades),
            'sharpe_ratio': float(sharpe_ratio),
            'avg_trade_pnl': float(total_return / total_trades) if total_trades > 0 else 0.0
        }

    def _analyze_regimes(self, trades: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze performance across different market regimes."""
        # Placeholder for regime analysis
        return {
            'bull_market_performance': 0.0,
            'bear_market_performance': 0.0,
            'sideways_performance': 0.0,
            'high_volatility_performance': 0.0
        }

    def _calculate_risk_metrics(self, trades: List[Dict[str, Any]]) -> Dict[str, float]:
        """Calculate risk metrics from trades."""
        if not trades:
            return {'max_drawdown': 0.0, 'volatility': 0.0}

        pnl_values = [trade['pnl'] for trade in trades]
        cumulative_pnl = np.cumsum(pnl_values)

        # Max drawdown
        peak = np.maximum.accumulate(cumulative_pnl)
        drawdown = peak - cumulative_pnl
        max_drawdown = np.max(drawdown) if len(drawdown) > 0 else 0.0

        # Volatility
        volatility = np.std(pnl_values) if len(pnl_values) > 1 else 0.0

        return {
            'max_drawdown': float(max_drawdown),
            'volatility': float(volatility),
            'var_95': float(np.percentile(pnl_values, 5)) if len(pnl_values) > 0 else 0.0
        }

    def _compare_with_benchmark(self, trades: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Compare performance with benchmark (buy and hold)."""
        # Placeholder for benchmark comparison
        return {
            'vs_buy_hold': 0.0,
            'benchmark_return': 0.0,
            'outperformance': 0.0
        }

    def print_report(self, result: BacktestResult) -> None:
        """Log formatted backtest report at INFO level."""
        logger.info("%s", "="*60)
        logger.info("SAC MODEL BACKTEST REPORT")
        logger.info("%s", "="*60)

        logger.info("📊 PERFORMANCE METRICS:")
        logger.info("  Total Trades: %s", result.performance_metrics.get('total_trades'))

        logger.info("⚠️  RISK METRICS:")

        logger.info("📈 TRADE LOG SUMMARY:")
        logger.info("  Total Trades Executed: %d", len(result.trade_log))

        if result.trade_log:
            # Map continuous actions to discrete using environment utility to
            # ensure thresholds are consistent across codebase.
            from ztb.trading.environment.constants import (
                continuous_to_discrete_action,
                ACTION_BUY,
                ACTION_SELL,
                ACTION_HOLD,
            )

            actions = [trade['action'] for trade in result.trade_log]
            mapped = [continuous_to_discrete_action(float(a)) for a in actions]

            buy_actions = sum(1 for m in mapped if m == ACTION_BUY)
            sell_actions = sum(1 for m in mapped if m == ACTION_SELL)
            hold_actions = sum(1 for m in mapped if m == ACTION_HOLD)

            logger.info("  BUY Actions: %d", buy_actions)
            logger.info("  SELL Actions: %d", sell_actions)
            logger.info("  HOLD Actions: %d", hold_actions)

    logger.info("%s", "="*60)


def main() -> None:
    """Main entry point."""
    parser = argparse.ArgumentParser(description='SAC Backtest Suite')
    parser.add_argument('--model', type=str, required=True, help='Path to SAC model file')
    parser.add_argument('--data', type=str, required=True, help='Path to test data (CSV)')
    parser.add_argument('--config', type=str, help='Path to configuration file')
    parser.add_argument('--episodes', type=int, default=1, help='Number of episodes to run')
    parser.add_argument('--deterministic', action='store_true', help='Use deterministic policy')
    parser.add_argument('--output', type=str, help='Output file for results (JSON)')

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
            'performance_metrics': result.performance_metrics,
            'trade_log': result.trade_log,
            'regime_analysis': result.regime_analysis,
            'risk_metrics': result.risk_metrics,
            'benchmark_comparison': result.benchmark_comparison
        }

        with open(args.output, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)

        print(f"\n📄 Results saved to: {args.output}")


if __name__ == '__main__':
    main()