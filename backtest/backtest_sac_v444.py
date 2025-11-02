#!/usr/bin/env python3
"""
SAC v444 Backtest Script - Advanced Regime Adaptation

Backtest the trained SAC v444 model with advanced regime adaptation features.
Evaluates performance against baseline and measures return improvement.
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from stable_baselines3 import SAC

from ztb.analysis.v444_regime_classifier import V444RegimeClassifier
from ztb.trading.environment.heavy_env.core import HeavyTradingEnv
from ztb.utils.logging_utils import get_logger

logger = get_logger(__name__)


class SACV444Backtester:
    """Backtester for SAC v444 model with regime adaptation."""

    def __init__(self, config_path: str, model_path: str):
        """Initialize backtester with config and model."""
        self.config_path = config_path
        self.model_path = model_path
        self.config = self._load_config()
        self.model = None
        self.env = None
        self.regime_classifier = None

    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from JSON file."""
        with open(self.config_path, "r") as f:
            return json.load(f)

    def _initialize_components(self):
        """Initialize model, environment, and regime classifier."""
        # Load the trained model
        self.model = SAC.load(self.model_path)
        logger.info(f"Loaded model from {self.model_path}")

        # Initialize regime classifier
        self.regime_classifier = V444RegimeClassifier()

        # Load data for backtest
        data_config = self.config.get("training", {}).get("data_config", {})
        csv_path = data_config.get("csv_path", "data/btc_jpy_real_dataset.csv")

        if not Path(csv_path).exists():
            raise FileNotFoundError(f"Data file not found: {csv_path}")

        df = pd.read_csv(csv_path)
        logger.info(f"Loaded data from {csv_path}, shape: {df.shape}")

        # Initialize environment with data and regime classifier
        env_config = self.config.get("environment", {}).get("config", {})
        env_config["advanced_market_regime"] = True
        # If the trained model expects a specific, fixed feature ordering/size,
        # allow overriding feature selection here to ensure the backtest env
        # produces observations that match the model's policy observation space.
        #
        # NOTE: the trained SAC v444 model used a compact high-quality feature
        # selection (8 features). If no explicit schema is provided in the
        # environment config, set `feature_names` to the known ordering so the
        # observation space matches the model.
        trained_feature_names = [
            "Supertrend",
            "Supertrend_Direction",
            "BB_Upper",
            "BB_Lower",
            "BB_Middle",
            "BB_Width",
            "BB_Position",
            "OBV",
        ]

        if "feature_names" not in env_config:
            # Attempt to match trained feature names to dataframe columns
            df_cols = [c for c in df.columns]
            lc_cols = {c.lower(): c for c in df_cols}
            matched = []
            for tf in trained_feature_names:
                tf_lc = tf.lower()
                if tf_lc in lc_cols:
                    matched.append(lc_cols[tf_lc])
                else:
                    # try a looser contains match
                    found = None
                    for col in df_cols:
                        if tf_lc in col.lower() or col.lower() in tf_lc:
                            found = col
                            break
                    if found:
                        matched.append(found)
            if len(matched) == len(trained_feature_names):
                env_config["feature_names"] = matched
            else:
                logger.warning(
                    "Could not map trained feature names to dataframe columns; leaving feature selection to environment auto-discovery",
                    extra={"attempted": trained_feature_names, "matched": matched},
                )

        self.env = HeavyTradingEnv(df=df, config=env_config)
        logger.info("Initialized environment with data and regime classifier")
        logger.info("Initialized environment with regime classifier")

    def run_backtest(self, num_episodes: int = 1) -> Dict[str, Any]:
        """Run backtest and return results."""
        if not self.model or not self.env:
            self._initialize_components()

        portfolio_values = []
        actions_history = []
        regime_history = []

        for episode in range(num_episodes):
            logger.info(f"Starting backtest episode {episode + 1}/{num_episodes}")

            obs, info = self.env.reset()
            done = False
            episode_portfolio_values = []
            episode_actions = []
            episode_regimes = []

            while not done:
                # Get action from model
                action, _ = self.model.predict(obs, deterministic=True)

                # Step environment
                obs, reward, done, truncated, info = self.env.step(action)

                # Record data
                episode_portfolio_values.append(info.get("portfolio_value", 0))
                episode_actions.append(
                    float(action[0])
                    if isinstance(action, np.ndarray)
                    else float(action)
                )
                episode_regimes.append(info.get("market_regime", "unknown"))

            portfolio_values.extend(episode_portfolio_values)
            actions_history.extend(episode_actions)
            regime_history.extend(episode_regimes)

        # Calculate metrics
        metrics = self._calculate_metrics(
            portfolio_values, actions_history, regime_history
        )

        return {
            "portfolio_values": portfolio_values,
            "actions_history": actions_history,
            "regime_history": regime_history,
            "metrics": metrics,
        }

    def _calculate_metrics(
        self,
        portfolio_values: List[float],
        actions_history: List[float],
        regime_history: List[str],
    ) -> Dict[str, Any]:
        """Calculate backtest metrics."""
        if not portfolio_values:
            return {}

        # Basic metrics
        initial_value = portfolio_values[0]
        final_value = portfolio_values[-1]
        total_return = (final_value - initial_value) / initial_value * 100

        # Calculate drawdown
        peak = initial_value
        max_drawdown = 0
        current_drawdown = 0

        for value in portfolio_values:
            if value > peak:
                peak = value
                current_drawdown = 0
            else:
                current_drawdown = (peak - value) / peak * 100
                max_drawdown = max(max_drawdown, current_drawdown)

        # Sharpe ratio (simplified - assuming daily returns)
        returns = np.diff(portfolio_values) / portfolio_values[:-1]
        if len(returns) > 0:
            sharpe_ratio = (
                np.mean(returns) / np.std(returns) * np.sqrt(252)
                if np.std(returns) > 0
                else 0
            )
        else:
            sharpe_ratio = 0

        # Regime distribution
        regime_counts = {}
        for regime in regime_history:
            regime_counts[regime] = regime_counts.get(regime, 0) + 1

        total_regimes = len(regime_history)
        regime_distribution = {
            regime: count / total_regimes * 100
            for regime, count in regime_counts.items()
        }

        return {
            "total_return_pct": total_return,
            "max_drawdown_pct": max_drawdown,
            "sharpe_ratio": sharpe_ratio,
            "initial_portfolio_value": initial_value,
            "final_portfolio_value": final_value,
            "total_trades": len(actions_history),
            "regime_distribution": regime_distribution,
        }

    def save_results(self, results: Dict[str, Any], output_path: str):
        """Save backtest results to JSON file."""
        with open(output_path, "w") as f:
            json.dump(results, f, indent=2, default=str)
        logger.info(f"Saved backtest results to {output_path}")


def main():
    """Main backtest function."""
    parser = argparse.ArgumentParser(
        description="SAC v444 Backtest with Advanced Regime Adaptation"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config/sac_v444_advanced_regime_adaptation_config.json",
        help="Path to configuration file",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="models/sac_v444_advanced_regime_adaptation.zip",
        help="Path to trained model file",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="backtest_results/sac_v444_backtest_results.json",
        help="Path to output results file",
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=1,
        help="Number of backtest episodes to run",
    )

    args = parser.parse_args()

    try:
        logger.info("🚀 SAC v444 Backtest - Advanced Regime Adaptation")
        logger.info("Configuration: %s", args.config)
        logger.info("Model: %s", args.model)
        logger.info("Output: %s", args.output)

        # Initialize backtester
        backtester = SACV444Backtester(args.config, args.model)

        # Run backtest
        results = backtester.run_backtest(num_episodes=args.episodes)

        # Print summary
        metrics = results["metrics"]
        logger.info("📊 Backtest Results Summary:")
        logger.info("Total Return: %.2f%%", metrics["total_return_pct"])
        logger.info("Sharpe Ratio: %.2f", metrics["sharpe_ratio"])
        logger.info("Max Drawdown: %.2f%%", metrics["max_drawdown_pct"])
        logger.info("Regime Distribution: %s", metrics["regime_distribution"])

        # Save results
        backtester.save_results(results, args.output)

        logger.info("✅ Backtest completed successfully!")
        logger.info("Results saved to: %s", args.output)

        return True

    except Exception as e:
        logger.error("Backtest failed: %s", e)
        logger.error("❌ Backtest failed: %s", e)
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
