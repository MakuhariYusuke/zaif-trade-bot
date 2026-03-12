#!/usr/bin/env python3
"""
SAC v445.3 Backtest Script - Multi-Period Analysis

Test the trained SAC v445.3 model across different market periods
to evaluate performance in various market conditions.
"""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

# Add project root to path
project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(project_root))

from stable_baselines3 import PPO

from ztb.trading.environment.heavy_env.core import HeavyTradingEnv
from ztb.trading.environment.utils.config import EnvironmentConfig
from ztb.utils.logging_utils import get_logger

logger = get_logger(__name__)


class SACV445MultiPeriodTester:
    """Multi-period tester for SAC v445.3 model."""

    def __init__(
        self, config_path: str, model_path: str, data_path: Optional[str] = None
    ):
        """Initialize tester with config and model."""
        self.config_path = config_path
        self.model_path = model_path
        self.data_path = data_path
        self.config = self._load_config()
        self.model = None
        self.df = None
        self.results = {}


    def _initialize_components(self):
        """Initialize model and data."""
        # Load the trained model
        self.model = PPO.load(self.model_path)
        logger.info(f"Loaded model from {self.model_path}")

        # Load data for testing
        if self.data_path:
            # Load custom data
            if not Path(self.data_path).exists():
                raise FileNotFoundError(f"Data file not found: {self.data_path}")
            custom_df = pd.read_csv(self.data_path)
            custom_df["timestamp"] = pd.to_datetime(custom_df["timestamp"])
            logger.info(
                f"Loaded custom data from {self.data_path}, shape: {custom_df.shape}"
            )

            # Also load training data to ensure consistent feature scaling
            data_config = self.config.get("training", {}).get("data_config", {})
            train_data_path = data_config.get(
                "data_path", "data/btc_jpy_real_dataset.csv"
            )
            if Path(train_data_path).exists():
                train_df = pd.read_csv(train_data_path)
                train_df["timestamp"] = pd.to_datetime(train_df["timestamp"])
                # Convert to UTC if not already
                try:
                    # Ensure both dataframes have consistent timezone handling
                    train_df["timestamp"] = (
                        train_df["timestamp"].dt.tz_localize("UTC")
                        if train_df["timestamp"].dt.tz is None
                        else train_df["timestamp"].dt.tz_convert("UTC")
                    )  # type: ignore
                    custom_df["timestamp"] = (
                        custom_df["timestamp"].dt.tz_convert("UTC")
                        if custom_df["timestamp"].dt.tz is not None
                        else custom_df["timestamp"].dt.tz_localize("UTC")
                    )  # type: ignore
                    # Remove timezone info to avoid comparison issues
                    train_df["timestamp"] = train_df["timestamp"].dt.tz_localize(None)  # type: ignore
                    custom_df["timestamp"] = custom_df["timestamp"].dt.tz_localize(None)  # type: ignore
                except (AttributeError, TypeError):
                    # Fallback: just ensure they're datetime
                    pass
                # Combine training and custom data for consistent feature engineering
                combined_df = pd.concat([train_df, custom_df], ignore_index=True)
                combined_df = (
                    combined_df.drop_duplicates(subset=["timestamp"])
                    .sort_values("timestamp")
                    .reset_index(drop=True)
                )
                self.df = combined_df
                logger.info(
                    f"Combined training and custom data, final shape: {self.df.shape}"
                )
            else:
                self.df = custom_df
        else:
            # Load default training data
            data_config = self.config.get("training", {}).get("data_config", {})
            csv_path = data_config.get("data_path", "data/btc_jpy_real_dataset.csv")
            if not Path(csv_path).exists():
                raise FileNotFoundError(f"Data file not found: {csv_path}")
            self.df = pd.read_csv(csv_path)
            self.df["timestamp"] = pd.to_datetime(self.df["timestamp"])

        logger.info(f"Final data shape: {self.df.shape}")

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
            "advanced_market_regime",
        ]
        for field in fields_to_remove:
            env_config_dict.pop(field, None)

        # Disable adaptive feature selection to maintain consistent feature dimensions
        env_config_dict["adaptive_feature_selection"] = {"enabled": False}
        env_config_dict["target_feature_count"] = 140

        env_config = EnvironmentConfig(**env_config_dict)

        # Create environment with full dataset to ensure consistent feature scaling
        self.full_env = HeavyTradingEnv(
            df=self.df, config=env_config, use_continuous_actions=True
        )
        logger.info(
            f"Environment created with observation space: {self.full_env.observation_space}"
        )
        logger.info(
            f"Environment created with action space: {self.full_env.action_space}"
        )

        # Verify observation space matches model expectations
        expected_obs_dim = self.model.observation_space.shape[0]
        actual_obs_dim = self.full_env.observation_space.shape[0]
        if actual_obs_dim != expected_obs_dim:
            logger.warning(
                f"Observation space mismatch: model expects {expected_obs_dim}, environment provides {actual_obs_dim}"
            )
            logger.warning(
                "This may cause prediction errors. Consider retraining the model with current data."
            )
            # Force feature consistency by recreating environment with training data only
            if self.data_path and actual_obs_dim < expected_obs_dim:
                logger.info(
                    "Recreating environment with training data for feature consistency..."
                )
                # Load training data
                train_df = pd.read_csv("data/btc_jpy_real_dataset.csv")
                train_df["timestamp"] = pd.to_datetime(train_df["timestamp"])
                # Create environment with training data only
                self.full_env = HeavyTradingEnv(
                    df=train_df, config=env_config, use_continuous_actions=True
                )
                if self.full_env.observation_space.shape[0] == expected_obs_dim:
                    logger.info(
                        f"Successfully recreated environment with correct observation space: {self.full_env.observation_space}"
                    )
                else:
                    logger.error(
                        f"Training data environment also has wrong dimensions: {self.full_env.observation_space.shape[0]}"
                    )
                    raise ValueError(
                        f"Cannot create environment with correct observation dimensions. Expected {expected_obs_dim}, got {self.full_env.observation_space.shape[0]}"
                    )

    def _identify_market_periods(
        self, window_size_hours: int = 24, overlap_ratio: float = 0.5
    ) -> List[Dict[str, Any]]:
        """Identify different market periods (uptrend, downtrend, sideways).

        Args:
            window_size_hours: Size of each analysis window in hours
            overlap_ratio: Overlap ratio between consecutive windows (0.0 to 1.0)
        """
        assert (
            self.df is not None
        ), "Data must be loaded before identifying market periods"
        periods = []

        # Calculate moving averages and trends
        self.df["MA20"] = self.df["close"].rolling(window=20).mean()
        self.df["MA50"] = self.df["close"].rolling(window=50).mean()

        # Calculate trend strength
        self.df["trend"] = (self.df["MA20"] - self.df["MA50"]) / self.df["MA50"] * 100

        # Identify periods with configurable window size and overlap
        step_size = int(window_size_hours * (1 - overlap_ratio))
        window_size = window_size_hours

        for i in range(0, len(self.df) - window_size, step_size):
            start_idx = i
            end_idx = min(i + window_size, len(self.df))

            period_data = self.df.iloc[start_idx:end_idx]
            start_price = period_data["close"].iloc[0]
            end_price = period_data["close"].iloc[-1]
            price_change = (end_price - start_price) / start_price * 100

            # Classify trend
            avg_trend = period_data["trend"].mean()
            volatility = period_data["close"].pct_change().std() * 100

            if price_change < -2:  # Downtrend (lowered threshold)
                trend_type = "downtrend"
            elif price_change > 2:  # Uptrend (lowered threshold)
                trend_type = "uptrend"
            else:  # Sideways
                trend_type = "sideways"

            periods.append(
                {
                    "start_idx": start_idx,
                    "end_idx": end_idx,
                    "start_date": period_data["timestamp"].iloc[0],
                    "end_date": period_data["timestamp"].iloc[-1],
                    "start_price": start_price,
                    "end_price": end_price,
                    "price_change_pct": price_change,
                    "trend_type": trend_type,
                    "avg_trend": avg_trend,
                    "volatility": volatility,
                    "duration_hours": len(period_data),
                }
            )

        return periods

    def test_period(
        self, start_idx: int, end_idx: int, period_name: str
    ) -> Dict[str, Any]:
        """Test the model on a specific period using the full environment."""
        assert self.df is not None, "Data must be loaded before testing periods"
        assert self.model is not None, "Model must be loaded before testing periods"
        logger.info(f"Testing period: {period_name}")

        # Ensure indices are within data bounds
        max_steps = len(self.df) - 1
        start_idx = min(start_idx, max_steps - 10)  # Leave some buffer
        end_idx = min(end_idx, max_steps)
        period_length = end_idx - start_idx

        if period_length <= 0:
            logger.warning(f"Invalid period length for {period_name}, skipping")
            return {
                "period_name": period_name,
                "start_date": str(self.df.iloc[start_idx]["timestamp"]),
                "end_date": str(self.df.iloc[end_idx]["timestamp"]),
                "duration_hours": 0,
                "initial_portfolio": 10000,
                "final_portfolio": 10000,
                "total_profit": 0,
                "total_return_pct": 0,
                "total_actions": 0,
                "sell_actions": 0,
                "buy_actions": 0,
                "hold_actions": 0,
                "sell_percentage": 0,
                "buy_percentage": 0,
                "hold_percentage": 0,
                "total_reward": 0,
                "average_reward": 0,
            }

        # Reset environment and advance to start_idx
        obs, _ = self.full_env.reset()

        # Advance to start_idx by taking dummy actions
        for step in range(start_idx):
            # Take a neutral action (hold) to advance through the data
            action = np.array([0.0])  # Neutral action
            obs, _, terminated, truncated, _ = self.full_env.step(action)
            if terminated or truncated:
                logger.warning(
                    f"Environment terminated before reaching start_idx {start_idx}"
                )
                # Reset and try again with a smaller start_idx
                obs, _ = self.full_env.reset()
                start_idx = max(0, start_idx - 50)  # Reduce start_idx
                for step in range(start_idx):
                    action = np.array([0.0])
                    obs, _, terminated, truncated, _ = self.full_env.step(action)
                    if terminated or truncated:
                        start_idx = max(0, start_idx - 50)  # Reduce further
                        break
                break

        # Now test the period from current position to end_idx
        done = False
        total_steps = 0
        actions_taken = []
        rewards_received = []
        max_test_steps = min(period_length, 1000)  # Limit to prevent excessive testing

        initial_portfolio_value = self.full_env.portfolio_value

        while not done and total_steps < max_test_steps:
            try:
                action, _ = self.model.predict(obs, deterministic=True)
                actions_taken.append(action[0])

                obs, reward, terminated, truncated, info = self.full_env.step(action)
                rewards_received.append(reward)

                done = terminated or truncated
                total_steps += 1
            except IndexError:
                # Data limit reached
                break

        # Calculate results
        final_portfolio_value = self.full_env.portfolio_value
        total_profit = final_portfolio_value - initial_portfolio_value
        total_return_pct = (
            (total_profit / initial_portfolio_value) * 100
            if initial_portfolio_value > 0
            else 0
        )

        # Action statistics
        sell_actions = sum(1 for a in actions_taken if a < -0.3)
        buy_actions = sum(1 for a in actions_taken if a > 0.3)
        hold_actions = len(actions_taken) - sell_actions - buy_actions

        return {
            "period_name": period_name,
            "start_date": str(self.df.iloc[start_idx]["timestamp"]),
            "end_date": str(self.df.iloc[min(end_idx, len(self.df) - 1)]["timestamp"]),
            "duration_hours": total_steps,  # Use actual steps taken
            "initial_portfolio": initial_portfolio_value,
            "final_portfolio": final_portfolio_value,
            "total_profit": total_profit,
            "total_return_pct": total_return_pct,
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
            "total_reward": sum(rewards_received),
            "average_reward": sum(rewards_received) / len(rewards_received)
            if rewards_received
            else 0,
        }

    def run_multi_period_analysis(
        self, window_sizes: Optional[List[int]] = None, overlap_ratio: float = 0.5
    ) -> Dict[str, Any]:
        """Run analysis across multiple market periods with different window sizes.

        Args:
            window_sizes: List of window sizes in hours to test
            overlap_ratio: Overlap ratio between consecutive windows
        """
        if window_sizes is None:
            window_sizes = [24, 48, 72, 168]  # 1日, 2日, 3日, 1週間

        logger.info("Starting multi-period analysis...")

        # Initialize components
        self._initialize_components()

        all_results = {}

        for window_size in window_sizes:
            logger.info(f"Testing with {window_size} hour windows...")

            # Identify market periods for this window size
            periods = self._identify_market_periods(window_size, overlap_ratio)
            logger.info(f"Identified {len(periods)} periods for {window_size}h windows")

            # Test each period
            period_results = []
            for i, period in enumerate(periods):
                period_name = f"{period['trend_type']}_{i+1}_{period['start_date'].strftime('%Y%m%d')}_{window_size}h"
                result = self.test_period(
                    period["start_idx"], period["end_idx"], period_name
                )
                result.update(
                    {
                        "trend_type": period["trend_type"],
                        "market_trend_type": period["trend_type"],
                        "price_change_pct": period["price_change_pct"],
                        "market_volatility": period["volatility"],
                        "window_size_hours": window_size,
                    }
                )
                period_results.append(result)
                logger.info(
                    f"Completed {period_name}: {result['total_return_pct']:+.2f}%"
                )

            all_results[f"{window_size}h_windows"] = {
                "period_results": period_results,
                "summary": self._generate_window_summary(period_results),
            }

        # Compile overall results
        self.results = {
            "analysis_info": {
                "model": "sac_v445.3_strong_selling_optimized",
                "config_file": self.config_path,
                "model_file": self.model_path,
                "timestamp": datetime.now().isoformat(),
                "window_sizes_tested": window_sizes,
                "overlap_ratio": overlap_ratio,
            },
            "results_by_window_size": all_results,
            "overall_summary": self._generate_overall_summary(all_results),
        }

        return self.results

    def _generate_overall_summary(
        self, results_by_window_size: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Generate overall summary across all window sizes."""
        overall_stats = {}

        for window_key, window_data in results_by_window_size.items():
            summary = window_data["summary"]
            window_size = window_key.replace("h_windows", "h")

            overall_stats[window_size] = {
                "total_periods": summary["overall"]["total_periods"],
                "avg_return": summary["overall"]["avg_return"],
                "win_rate": summary["overall"]["win_rate"],
                "sharpe_ratio": summary["overall"]["sharpe_ratio"],
                "trend_breakdown": summary["by_trend_type"],
            }

        return overall_stats

    def _generate_window_summary(
        self, period_results: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Generate summary statistics for a specific window size."""
        if not period_results:
            return {
                "overall": {
                    "total_periods": 0,
                    "avg_return": 0.0,
                    "win_rate": 0.0,
                    "sharpe_ratio": 0.0,
                },
                "by_trend_type": {},
            }

        returns = [r["total_return_pct"] for r in period_results]
        trend_types = [r["trend_type"] for r in period_results]

        # Overall statistics
        avg_return = np.mean(returns)
        win_rate = (np.array(returns) > 0).mean() * 100
        sharpe_ratio = avg_return / np.std(returns) if np.std(returns) > 0 else 0

        # By trend type
        trend_breakdown = {}
        for trend_type in set(trend_types):
            trend_returns = [
                r["total_return_pct"]
                for r in period_results
                if r["trend_type"] == trend_type
            ]
            if trend_returns:
                trend_breakdown[trend_type] = {
                    "count": len(trend_returns),
                    "avg_return": np.mean(trend_returns),
                    "win_rate": (np.array(trend_returns) > 0).mean() * 100,
                    "sharpe_ratio": np.mean(trend_returns) / np.std(trend_returns)
                    if np.std(trend_returns) > 0
                    else 0,
                }

        return {
            "overall": {
                "total_periods": len(period_results),
                "avg_return": avg_return,
                "win_rate": win_rate,
                "sharpe_ratio": sharpe_ratio,
            },
            "by_trend_type": trend_breakdown,
        }

    def save_results(self, output_path: str):
        """Save analysis results to JSON file."""

        def convert_numpy_types(obj):
            if isinstance(obj, (np.float32, np.float64)):
                return float(obj)
            elif isinstance(obj, (np.int32, np.int64)):
        logger.info(f"Results saved to {output_path}")

    def print_summary(self):
        """Print a summary of the analysis results."""
        if not self.results:
            logger.error("No results to display. Run analysis first.")
            return

        print("\n" + "=" * 100)
        print("🎯 SAC v445.3 多期間・多時間枠分析結果")
        print("=" * 100)

        overall_summary = self.results.get("overall_summary", {})

        for window_size, stats in overall_summary.items():
            print(f"\n📊 {window_size} ウィンドウ:")
            print(f"  テスト期間数: {stats['total_periods']}")
            print(f"  平均リターン: {stats['avg_return']:+.2f}%")
            print(f"  勝率: {stats['win_rate']:.1f}%")
            print(f"  シャープレシオ: {stats['sharpe_ratio']:.3f}")

            print("  トレンド別パフォーマンス:")
            for trend_type, trend_stats in stats["trend_breakdown"].items():
                print(
                    f"    {trend_type.upper()}: {trend_stats['count']}期間, "
                    f"平均{trend_stats['avg_return']:+.2f}%, 勝率{trend_stats['win_rate']:.1f}%"
                )

        print("\n" + "=" * 100)


def main():
    """Main function to run the multi-period analysis."""
    parser = argparse.ArgumentParser(
        description="Multi-period analysis of SAC v445.3 model"
    )
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
        "--data", type=str, help="Path to data file (optional, overrides config)"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="results/sac_v445.3_multi_period_analysis.json",
        help="Path to save results",
    )
    parser.add_argument(
        "--window-sizes",
        type=int,
        nargs="+",
        default=[24, 48, 72, 168],
        help="Window sizes in hours to test",
    )
    parser.add_argument(
        "--overlap",
        type=float,
        default=0.5,
        help="Overlap ratio between windows (0.0-1.0)",
    )

    args = parser.parse_args()

    # Initialize analyzer
    analyzer = SACV445MultiPeriodTester(args.config, args.model, args.data)

    try:
        # Run analysis with specified window sizes
        results = analyzer.run_multi_period_analysis(
            window_sizes=args.window_sizes, overlap_ratio=args.overlap
        )

        # Print summary
        analyzer.print_summary()

        # Save results
        analyzer.save_results(args.output)

        print(f"\n✅ 多期間分析完了！結果を {args.output} に保存しました。")

    except Exception as e:
        logger.error(f"Analysis failed: {e}")
        raise


if __name__ == "__main__":
    main()
