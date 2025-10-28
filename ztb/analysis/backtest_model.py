#!/usr/bin/env python3
"""
Backtest Script for Zaif Trade Bot.

Tests a trained model (PPO/SAC) using historical BTC/JPY data.
"""

import argparse
import logging
import sys
from typing import Any, Dict, List, Optional, cast

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from sb3_contrib import MaskablePPO
from stable_baselines3 import PPO, SAC

from ztb.config.manager import ConfigManager
from ztb.trading.constants import TRADING_DAYS_PER_YEAR  # = 252
from ztb.trading.constants import (  # 年間取引日数
    ACTION_BUY,
    ACTION_HOLD,
    ACTION_SELL,
    SAC_CONTINUOUS_THRESHOLD,
    SAC_CONTINUOUS_THRESHOLD_NEG,
)
from ztb.utils.file_utils import safe_json_dump
from ztb.utils.path_utils import get_file_dir, resolve_path
from ztb.utils.performance_utils import timed

# Add project root to path
project_root = get_file_dir(__file__)
sys.path.insert(0, str(project_root))

from ztb.trading.environment.environment import HeavyTradingEnv


# Feature Engineering Strategy Pattern
class FeatureEngineeringStrategy:
    """Base class for feature engineering strategies."""

    def apply(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply feature engineering to the dataframe."""
        raise NotImplementedError

    def get_description(self) -> str:
        """Get description of the feature engineering method."""
        raise NotImplementedError


class SACv427FeatureEngineeringStrategy(FeatureEngineeringStrategy):
    """Strategy for SAC v427 standard features."""

    def apply(self, df: pd.DataFrame) -> pd.DataFrame:
        from ztb.features.sac_v427_feature_engineering import SACv427FeatureEngineer

        feature_engineer = SACv427FeatureEngineer()
        return feature_engineer.generate_v427_features(df)

    def get_description(self) -> str:
        return "SAC v427 standard features"


class SACv427QualityFilteredFeatureEngineeringStrategy(FeatureEngineeringStrategy):
    """Strategy for SAC v427 quality-filtered features."""

    def apply(self, df: pd.DataFrame) -> pd.DataFrame:
        from ztb.features.sac_v427_feature_engineering import (
            generate_v427_quality_filtered_features,
        )

        return generate_v427_quality_filtered_features(df, feature_set="full")

    def get_description(self) -> str:
        return "SAC v427 quality-filtered features (109 features)"


class SACv437FeatureEngineeringStrategy(FeatureEngineeringStrategy):
    """Strategy for SAC v437 features (placeholder for future implementation)."""

    def apply(self, df: pd.DataFrame) -> pd.DataFrame:
        # Placeholder - would implement v437 specific features
        import logging
        logging.getLogger(__name__).warning("SAC v437 feature engineering not yet implemented, using raw data")
        return df

    def get_description(self) -> str:
        return "SAC v437 features (not implemented)"


class FeatureEngineeringFactory:
    """Factory for creating feature engineering strategies."""

    _strategies = {
        "sac_v427": SACv427FeatureEngineeringStrategy,
        "sac_v427_quality_filtered": SACv427QualityFilteredFeatureEngineeringStrategy,
        "sac_v437": SACv437FeatureEngineeringStrategy,
        "sac_v437_1": SACv437FeatureEngineeringStrategy,  # Same as v437 for now
    }

    @classmethod
    def create_strategy(cls, method: str) -> Optional[FeatureEngineeringStrategy]:
        """Create a feature engineering strategy for the given method."""
        strategy_class = cls._strategies.get(method)
        if strategy_class:
            return strategy_class()
        return None

    @classmethod
    def get_available_methods(cls) -> List[str]:
        """Get list of available feature engineering methods."""
        return list(cls._strategies.keys())


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
        sharpe_ratio = (
            np.mean(returns) / (np.std(returns) + 1e-6) * np.sqrt(TRADING_DAYS_PER_YEAR)
        )
    else:
        sharpe_ratio = 0.0

    # Max drawdown
    capital_history_array: NDArray[np.float64] = np.array(capital_history)
    peak = np.maximum.accumulate(capital_history_array)
    drawdown = (capital_history_array - peak) / peak
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
    curriculum_stage: str = "forced_balance",
    transaction_cost: float = 0.0005,
    max_steps: Optional[int] = None,
    config_path: Optional[str] = None,
    verbose: bool = True,
    feature_engineering: Optional[str] = None,
    n_episodes: int = 1,
    deterministic: bool = False,
) -> Dict[str, Any]:
    """Run backtest simulation."""
    logging.basicConfig(level=logging.INFO if verbose else logging.WARNING)
    logger = logging.getLogger(__name__)

    # Load model
    logger.info(f"Loading model from {model_path}")
    model_type = None
    try:
        # Try loading as MaskablePPO first (for models with action masking)
        model = MaskablePPO.load(model_path)
        model_type = "MaskablePPO"
        logger.info("Loaded model as MaskablePPO")
    except Exception as e:
        logger.info(f"Failed to load as MaskablePPO: {e}, trying regular PPO")
        try:
            model = PPO.load(model_path)
            model_type = "PPO"
            logger.info("Loaded model as PPO")
        except Exception as e2:
            logger.info(f"Failed to load as PPO: {e2}, trying SAC")
            model = SAC.load(model_path)
            model_type = "SAC"
            logger.info("Loaded model as SAC")

    # Load data
    logger.info(f"Loading data from {data_path}")
    df = pd.read_csv(data_path)
    logger.info(f"Loaded {len(df)} rows of data")

    # Apply feature engineering if specified
    if feature_engineering:
        logger.info(f"Applying feature engineering: {feature_engineering}")
        strategy = FeatureEngineeringFactory.create_strategy(feature_engineering)
        if strategy:
            df = strategy.apply(df)
            logger.info(f"Applied {strategy.get_description()}: {len(df.columns)} features")
        else:
            logger.warning(f"Unknown feature engineering method: {feature_engineering}")

    # Limit data if specified
    if max_steps and len(df) > max_steps:
        df = df.head(max_steps)
        logger.info(f"Limited to {max_steps} steps")

    # Create environment
    if config_path:
        # Load configuration using new ConfigManager
        config_manager = ConfigManager.get_instance()
        global_config = config_manager.load_config(config_path)

        # Extract environment config from global config
        env_config = (
            global_config.training.environment.model_dump()
            if global_config.training and global_config.training.environment
            else {}
        )
        env_config["transaction_cost"] = transaction_cost
        # Use curriculum_stage from config if available, otherwise use parameter
        env_config["curriculum_stage"] = env_config.get(
            "curriculum_stage", curriculum_stage
        )
    else:
        env_config = {
            "transaction_cost": transaction_cost,
            "enable_correlation_reduction": True,
            "correlation_threshold": 0.95,
            "max_position_size": 0.5,
            "curriculum_stage": curriculum_stage,  # Match training configuration
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
        config=env_config,
        random_start=False,
    )

    # Run multiple episodes
    all_trades = []
    episode_results = []

    for episode in range(n_episodes):
        logger.info(f"Running episode {episode + 1}/{n_episodes}")

        # Reset environment for each episode
        obs, _ = env.reset()
        done = False
        episode_trades = []
        last_position = 0
        entry_price = 0
        entry_time = 0
        step = 0
        actions_taken = []

        while not done:
            # Get action with proper masking for MaskablePPO
            if model_type == "MaskablePPO":
                action_masks = env.get_action_masks()
                action, _ = model.predict(
                    obs, action_masks=action_masks, deterministic=deterministic
                )
            else:
                action, _ = model.predict(obs, deterministic=deterministic)

            # Convert continuous action to discrete for SAC models
            if model_type == "SAC":
                # SAC uses continuous actions, convert to discrete
                # Use centralized thresholds from constants
                if action > SAC_CONTINUOUS_THRESHOLD:
                    discrete_action = ACTION_BUY
                elif action < SAC_CONTINUOUS_THRESHOLD_NEG:
                    discrete_action = ACTION_SELL
                else:
                    discrete_action = ACTION_HOLD
                action = discrete_action
            else:
                action = cast(int, action.item() if hasattr(action, "item") else action)

            actions_taken.append(action)

            obs, _, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            step += 1

            # Track position changes for trade recording
            current_position = env.position
            if step % 100 == 0:  # Debug output every 100 steps
                logger.info(
                    f"Episode {episode + 1}, Step {step}: action={action}, position={current_position}"
                )

            # Detect position changes (considering allow_reverse=True)
            # 1. Opening new position from flat
            if abs(current_position) > 0 and abs(last_position) == 0:
                entry_price = env.df.iloc[min(env.current_step, len(env.df) - 1)][
                    "close"
                ]
                entry_time = env.current_step
                logger.info(
                    f"Episode {episode + 1}: Opened position at step {step}: {current_position}"
                )

            # 2. Closing position to flat
            elif abs(last_position) > 0 and abs(current_position) == 0:
                exit_price = env.df.iloc[min(env.current_step, len(env.df) - 1)][
                    "close"
                ]
                pnl = (
                    (exit_price - entry_price)
                    * last_position
                    * (1 - transaction_cost * 2)
                )
                trade = {
                    "episode": episode + 1,
                    "entry_time": entry_time,
                    "exit_time": env.current_step,
                    "entry_price": entry_price,
                    "exit_price": exit_price,
                    "pnl": pnl,
                    "position": last_position,
                }
                episode_trades.append(trade)
                logger.info(
                    f"Episode {episode + 1}: Closed position at step {step}: pnl={pnl}"
                )

            # 3. Position reversal (Long→Short or Short→Long)
            elif (last_position > 0 and current_position < 0) or (
                last_position < 0 and current_position > 0
            ):
                # Close previous position
                exit_price = env.df.iloc[min(env.current_step, len(env.df) - 1)][
                    "close"
                ]
                pnl = (
                    (exit_price - entry_price)
                    * last_position
                    * (1 - transaction_cost * 2)
                )
                trade = {
                    "episode": episode + 1,
                    "entry_time": entry_time,
                    "exit_time": env.current_step,
                    "entry_price": entry_price,
                    "exit_price": exit_price,
                    "pnl": pnl,
                    "position": last_position,
                }
                episode_trades.append(trade)
                logger.info(
                    f"Episode {episode + 1}: Reversed position at step {step}: pnl={pnl}"
                )

                # Open new position
                entry_price = exit_price
                entry_time = env.current_step

            last_position = current_position

        # Store episode results
        final_portfolio_value = env.portfolio_value
        total_reward = final_portfolio_value - initial_capital

        episode_result = {
            "episode": episode + 1,
            "total_reward": total_reward,
            "final_portfolio_value": final_portfolio_value,
            "total_trades": len(episode_trades),
            "total_steps": step,
        }
        episode_results.append(episode_result)
        all_trades.extend(episode_trades)

        logger.info(
            f"Episode {episode + 1} completed: Reward={total_reward:.2f}, Trades={len(episode_trades)}, Final Value={final_portfolio_value:.2f}"
        )

    # Calculate aggregate metrics
    metrics = calculate_metrics(all_trades, initial_capital)

    logger.info("Backtest completed")
    logger.info(f"Total Return: {metrics['total_return']:.2f}%")
    logger.info(f"Win Rate: {metrics['win_rate']:.2f}%")
    logger.info(f"Total Trades: {metrics['total_trades']}")
    logger.info(f"Sharpe Ratio: {metrics['sharpe_ratio']:.2f}")

    return {
        "metrics": metrics,
        "trades": all_trades,
        "episode_results": episode_results,
        "config": {
            "model_path": model_path,
            "data_path": data_path,
            "initial_capital": initial_capital,
            "transaction_cost": transaction_cost,
            "max_steps": max_steps,
            "feature_engineering": feature_engineering,
            "n_episodes": n_episodes,
            "deterministic": deterministic,
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
        "--config",
        help="Path to config file (JSON format)",
    )
    parser.add_argument(
        "--feature-engineering",
        choices=FeatureEngineeringFactory.get_available_methods(),
        help="Feature engineering method to apply",
    )
    parser.add_argument(
        "--n-episodes",
        type=int,
        default=1,
        help="Number of episodes to run (default: 1)",
    )
    parser.add_argument(
        "--deterministic",
        action="store_true",
        help="Use deterministic actions (default: False)",
    )
    parser.add_argument(
        "--output",
        help="Path to save results as JSON (optional)",
    )

    args = parser.parse_args()

    # Run backtest
    results = run_backtest(
        model_path=args.model_path,
        data_path=args.data_path,
        initial_capital=args.initial_capital,
        transaction_cost=args.transaction_cost,
        max_steps=args.max_steps,
        config_path=args.config,
        feature_engineering=args.feature_engineering,
        n_episodes=args.n_episodes,
        deterministic=args.deterministic,
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
        safe_json_dump(results, resolve_path(args.output), indent=2, default=str)
        print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()  # type: ignore[no-untyped-call]
