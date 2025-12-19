#!/usr/bin/env python3
"""
SAC v435 Backtest Script
Run backtests for SAC v435, v435.1, and v435.2 models
"""

import json
import logging
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd
from stable_baselines3 import SAC

from ztb.metrics import sharpe_ratio as calculate_sharpe_ratio
from ztb.metrics.metrics import max_drawdown as calculate_max_drawdown
from ztb.risk.risk_manager import RiskManager
from ztb.trading.risk.compat import ensure_risk_manager_protocol

logger = logging.getLogger(__name__)


class SACv435Backtester:
    """SAC v435 Backtester for multiple model variants"""

    def __init__(self, model_variants: List[str] = None):
        """
        Initialize backtester for multiple SAC v435 variants

        Args:
            model_variants: List of model variants to test (e.g., ['v435', 'v435_1', 'v435_2'])
        """
        if model_variants is None:
            model_variants = ["v435", "v435_1", "v435_2"]

        self.model_variants = model_variants
        self.models = {}
        self.configs = {}
        self.results = {}

        logger.info(f"SAC v435 Backtester initialized for variants: {model_variants}")

    def load_models(self) -> bool:
        """Load all model variants"""
        success = True

        for variant in self.model_variants:
            try:
                # Load config
                config_path = Path(f"config/v435/sac_{variant}_config.json")
                if not config_path.exists():
                    logger.error(f"Config not found: {config_path}")
                    success = False
                    continue

                with open(config_path, "r") as f:
                    config = json.load(f)

                self.configs[variant] = config

                # Load model
                model_dir = config.get("output", {}).get(
                    "model_dir", f"models/{variant}"
                )
                model_path = Path(model_dir) / "sac_v435_final.zip"

                if not model_path.exists():
                    logger.warning(f"Model not found: {model_path}")
                    self.models[variant] = None
                    continue

                logger.info(f"Loading model: {model_path}")
                self.models[variant] = SAC.load(str(model_path))

                logger.info(f"✅ Loaded {variant} model successfully")

            except Exception as e:
                logger.error(f"❌ Failed to load {variant} model: {e}")
                success = False

        return success

    def run_backtests(
        self, test_data_path: str = "data/btc_jpy_featured_dataset.csv"
    ) -> Dict[str, Any]:
        """Run backtests for all loaded models"""

        # Load test data
        try:
            df = pd.read_csv(test_data_path)
            logger.info(f"Loaded test data: {len(df)} rows")
        except Exception as e:
            logger.error(f"Failed to load test data: {e}")
            return {}

        results = {}

        for variant, model in self.models.items():
            if model is None:
                logger.warning(f"Skipping {variant} - model not loaded")
                continue

            try:
                logger.info(f"Running backtest for {variant}...")

                # Run backtest
                result = self._run_single_backtest(model, df, self.configs[variant])
                results[variant] = result

                logger.info(
                    f"✅ {variant} backtest completed: {result.get('total_return', 'N/A'):.4f}"
                )

            except Exception as e:
                logger.error(f"❌ {variant} backtest failed: {e}")
                results[variant] = {"error": str(e)}

        self.results = results
        return results

    def _run_single_backtest(
        self, model: SAC, df: pd.DataFrame, config: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Run backtest for a single model"""

        # Initialize portfolio
        initial_balance = 100000
        balance = initial_balance
        position = 0
        trades = []
        portfolio_values = [initial_balance]

        # Risk management
        if config.get("risk_management", {}).get("dynamic_position_sizing", False):
            ensure_risk_manager_protocol(
                RiskManager(config.get("risk_management", {}))
            )

        # Trading parameters
        transaction_cost = config.get("environment", {}).get("transaction_cost", 0.0015)
        max_position_size = config.get("environment", {}).get("max_position_size", 0.1)

        # Simulate trading
        for i in range(len(df) - 1):
            current_price = df.iloc[i]["close"]
            next_price = df.iloc[i + 1]["close"]

            # Prepare observation (simplified)
            obs = self._prepare_observation(df.iloc[i])

            # Get model action
            action, _ = model.predict(obs, deterministic=True)

            # Convert action to position (-1 to 1)
            target_position = float(action[0])  # SAC outputs continuous action

            # Apply position limits
            target_position = np.clip(
                target_position, -max_position_size, max_position_size
            )

            # Calculate position change
            position_change = target_position - position

            if abs(position_change) > 0.01:  # Minimum trade threshold
                # Calculate trade value
                trade_value = abs(position_change) * balance

                # Apply transaction cost
                cost = trade_value * transaction_cost
                balance -= cost

                # Update position
                position = target_position

                # Record trade
                trades.append(
                    {
                        "step": i,
                        "price": current_price,
                        "position": position,
                        "balance": balance,
                        "cost": cost,
                    }
                )

            # Update balance based on price movement
            if position != 0:
                price_change = (next_price - current_price) / current_price
                balance *= 1 + position * price_change

            portfolio_values.append(balance)

        # Calculate final metrics
        total_return = (balance - initial_balance) / initial_balance
        total_trades = len(trades)
        win_rate = 1.0 if total_return > 0 else 0.0

        # Calculate returns for Sharpe ratio
        returns = (
            np.diff(portfolio_values) / portfolio_values[:-1]
            if len(portfolio_values) > 1
            else [0]
        )
        sharpe_ratio = calculate_sharpe_ratio(returns)

        max_drawdown_result = calculate_max_drawdown(portfolio_values)
        max_drawdown = max_drawdown_result["max_drawdown"]

        result = {
            "total_return": total_return,
            "final_balance": balance,
            "total_trades": total_trades,
            "win_rate": win_rate,
            "sharpe_ratio": sharpe_ratio,
            "max_drawdown": max_drawdown,
            "trades": trades,
        }

        return result

    def _prepare_observation(self, row: pd.Series) -> np.ndarray:
        """Prepare observation from data row matching the model's observation space"""
        # Get available features from the data row
        available_features = [
            "close",
            "volume",
            "rsi_14",
            "macd",
            "macd_signal",
            "macd_hist",
            "bb_upper",
            "bb_middle",
            "bb_lower",
            "bb_width",
            "stoch_k",
            "stoch_d",
            "williams_r",
            "ichimoku_tenkan",
            "ichimoku_kijun",
            "ichimoku_senkou_a",
            "ichimoku_senkou_b",
            "atr_14",
            "cci_14",
            "mfi_14",
            "roc_12",
            "mom_10",
            "price_change",
            "volume_change",
            "returns",
            "log_returns",
            "sma_5",
            "sma_10",
            "sma_20",
            "sma_50",
            "volatility_5d",
            "volatility_10d",
            "volatility_20d",
        ]

        # Extract available features, defaulting to 0 if not present
        obs_list = []
        for feature in available_features:
            value = row.get(feature, 0.0)
            if pd.isna(value):
                value = 0.0
            obs_list.append(float(value))

        # Ensure we have at least 10 features for the model
        while len(obs_list) < 10:
            obs_list.append(0.0)

        # Take first 10 features (or pad if needed)
        obs = np.array(obs_list[:10], dtype=np.float32)

        return obs

    def save_results(self, output_path: str = "backtest_results_v435.json"):
        """Save backtest results to file"""
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)

        with open(output_file, "w") as f:
            json.dump(self.results, f, indent=2, default=str)

        logger.info(f"Results saved to {output_file}")

    def print_summary(self):
        """Print backtest summary"""
        print("\n" + "=" * 60)
        print("SAC v435 Backtest Results Summary")
        print("=" * 60)

        for variant, result in self.results.items():
            if "error" in result:
                print(f"\n{variant}: ERROR - {result['error']}")
                continue

            print(f"\n{variant}:")
            print(".4f")
            print(".4f")
            print(f"  Total Trades: {result.get('total_trades', 0)}")
            print(".4f")
            print(".4f")
            print(".4f")

        print("\n" + "=" * 60)


def main():
    """Run backtests for all SAC v435 variants"""

    # Initialize backtester
    backtester = SACv435Backtester(["v435", "v435_1", "v435_2"])

    # Load models
    if not backtester.load_models():
        logger.error("Failed to load all models")
        return

    # Run backtests
    backtester.run_backtests()

    # Print summary
    backtester.print_summary()

    # Save results
    backtester.save_results("backtest_results_v435.json")

    logger.info("Backtest completed successfully!")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
