#!/usr/bin/env python3
"""
SAC v435 Evaluation Script
Phase 5: Training and Evaluation - Performance evaluation with risk management
"""

import json
import logging
from pathlib import Path
from typing import Any, Dict

import numpy as np
import pandas as pd
from stable_baselines3 import SAC

from ztb.metrics import max_drawdown, sharpe_ratio
from ztb.risk.risk_manager import RiskManager
from ztb.trading.risk.compat import ensure_risk_manager_protocol
from ztb.utils.trading_metrics import win_rate

logger = logging.getLogger(__name__)


class SACv435Evaluator:
    """SAC v435 Model Evaluator with Risk Management Analysis"""

    def __init__(self, config_path: str = "config/v435/sac_v435_config.json"):
        """
        Initialize v435 evaluator

        Args:
            config_path: Configuration file path
        """
        self.config_path = Path(config_path)
        self.config = self._load_config()
        self.model = None
        self.risk_manager = None

        # Initialize risk management for evaluation
        if self.config.get("risk_management", {}).get("dynamic_position_sizing", False):
            self._setup_risk_management()

        logger.info("SAC v435 Evaluator initialized")

    def _load_config(self) -> Dict[str, Any]:
        """Load configuration"""
        with open(self.config_path, "r", encoding="utf-8") as f:
            return json.load(f)

    def _setup_risk_management(self):
        """Setup risk management for evaluation"""
        logger.info("Setting up risk management for v435 evaluation")

        risk_config = self.config.get("risk_management", {})

        risk_manager_config = {
            "position_sizer": {
                "enabled": risk_config.get("dynamic_position_sizing", True),
                "volatility_adjustment": risk_config.get("volatility_adjustment", True),
                "min_position_size": 0.001,
                "max_position_size": 0.2,
                "base_position_size": 0.1,
            },
            "drawdown_controller": {
                "enabled": risk_config.get("drawdown_control", True),
                "max_drawdown_limit": risk_config.get("max_drawdown_limit", 0.1),
                "emergency_stop_threshold": 0.15,
                "recovery_threshold": 0.05,
            },
            "market_adaptor": {
                "enabled": True,
                "adaptation_window": 50,
                "volatility_threshold": 0.02,
                "trend_strength_threshold": 0.01,
                "regime_change_threshold": 0.7,
            },
        }

        self.risk_manager = ensure_risk_manager_protocol(
            RiskManager(risk_manager_config)
        )
        logger.info("Risk management setup complete for evaluation")

    def load_model(self, model_path: str) -> SAC:
        """Load trained model"""
        logger.info(f"Loading model from {model_path}")
        self.model = SAC.load(model_path)
        return self.model

    def evaluate_model(self, test_data_path: str = None) -> Dict[str, Any]:
        """
        Evaluate model performance with risk management analysis

        Args:
            test_data_path: Path to test data (optional, uses config default if not provided)

        Returns:
            Evaluation results dictionary
        """
        if test_data_path is None:
            test_data_path = self.config["data"]["primary_dataset"]

        logger.info(f"Evaluating model on {test_data_path}")

        # Load test data
        test_df = pd.read_csv(test_data_path)
        logger.info(f"Loaded test data: {len(test_df)} rows")

        # Initialize simple backtest
        initial_balance = 100000
        portfolio_values = [initial_balance]
        trades = []

        # Run simplified backtest with risk management
        results = self._run_simplified_backtest(test_df, initial_balance)

        # Calculate basic metrics
        performance_metrics = self._calculate_basic_metrics(results)

        # Add risk-specific metrics
        risk_metrics = self._calculate_risk_metrics(results)

        # Combine all results
        evaluation_results = {
            "performance_metrics": performance_metrics,
            "risk_metrics": risk_metrics,
            "backtest_results": results,
            "config": self.config,
            "test_data_info": {
                "path": test_data_path,
                "rows": len(test_df),
                "date_range": [test_df["timestamp"].min(), test_df["timestamp"].max()]
                if "timestamp" in test_df.columns
                else None,
            },
        }

        logger.info("Evaluation complete")
        return evaluation_results

        return evaluation_results

    def _run_simplified_backtest(
        self, test_df: pd.DataFrame, initial_balance: float
    ) -> Dict[str, Any]:
        """Run simplified backtest with risk management"""
        logger.info("Running simplified backtest with risk management")

        portfolio_values = [initial_balance]
        trades = []
        risk_adjustments = []
        current_position = 0

        transaction_cost = self.config["environment"]["transaction_cost"]

        # Simulate trading for first 100 steps (for testing)
        test_steps = min(100, len(test_df) - 1)

        for i in range(test_steps):
            current_data = test_df.iloc[i]
            next_data = test_df.iloc[i + 1]

            # Get model prediction
            obs = self._prepare_observation(current_data)
            action, _ = self.model.predict(obs, deterministic=True)

            # Apply risk management
            original_position = action[0]
            adjusted_position = original_position

            if self.risk_manager:
                try:
                    risk_result = self.risk_manager.calculate_risk_adjusted_position(
                        base_position=original_position,
                        current_price=current_data.get("close", 100000),
                        portfolio_value=portfolio_values[-1],
                        atr=current_data.get("atr", 1000),
                        df=pd.DataFrame([current_data]),
                    )
                    adjusted_position = risk_result["adjusted_position"]

                    # Record risk adjustment
                    risk_adjustments.append(
                        {
                            "step": i,
                            "original_position": original_position,
                            "adjusted_position": adjusted_position,
                            "risk_level": risk_result.get("risk_level", 0),
                        }
                    )

                except Exception as e:
                    logger.warning(f"Risk adjustment failed at step {i}: {e}")

            # Simple trade execution (buy/sell logic)
            if (
                abs(adjusted_position - current_position) > 0.01
            ):  # Position change threshold
                # Calculate trade value
                trade_value = (
                    abs(adjusted_position - current_position) * portfolio_values[-1]
                )

                # Apply transaction cost
                cost = trade_value * transaction_cost

                # Update portfolio (simplified)
                if adjusted_position > current_position:  # Buy
                    portfolio_values.append(portfolio_values[-1] - cost)
                else:  # Sell
                    portfolio_values.append(portfolio_values[-1] - cost)

                # Record trade
                trades.append(
                    {
                        "step": i,
                        "position_change": adjusted_position - current_position,
                        "cost": cost,
                        "portfolio_value": portfolio_values[-1],
                    }
                )

                current_position = adjusted_position
            else:
                portfolio_values.append(portfolio_values[-1])

        results = {
            "portfolio_values": portfolio_values,
            "trades": trades,
            "risk_adjustments": risk_adjustments,
            "total_trades": len(trades),
            "final_portfolio_value": portfolio_values[-1],
            "total_return": (portfolio_values[-1] / initial_balance) - 1,
        }

        logger.info(
            f"Simplified backtest complete: {len(trades)} trades, Final value: {portfolio_values[-1]:.2f}"
        )
        return results

    def _calculate_basic_metrics(
        self, backtest_results: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Calculate basic performance metrics"""
        portfolio_values = backtest_results["portfolio_values"]
        trades = backtest_results["trades"]

        # Basic metrics
        total_return = backtest_results["total_return"]
        total_trades = len(trades)

        # Win rate (simplified)
        if total_trades > 0:
            returns = [
                trade.get("portfolio_value", 0) - trade.get("previous_value", 0)
                for trade in trades
            ]
            win_rate_value = win_rate(returns)
        else:
            win_rate_value = 0

        return {
            "total_return": total_return,
            "total_trades": total_trades,
            "win_rate": win_rate_value,
        }

    def _prepare_observation(self, data: pd.Series) -> np.ndarray:
        """Prepare observation for model input"""
        # This should match the environment's observation preparation
        # Simplified version for evaluation
        obs_features = []

        # Add key features (this should be configurable based on the model's feature set)
        feature_columns = [
            "close",
            "volume",
            "rsi_14",
            "macd",
            "macd_signal",
            "bb_upper",
            "bb_middle",
            "bb_lower",
            "stoch_k",
            "stoch_d",
        ]

        for col in feature_columns:
            if col in data.index:
                obs_features.append(data[col])
            else:
                obs_features.append(0.0)  # Default value for missing features

        return np.array(obs_features, dtype=np.float32)

    def _calculate_risk_metrics(
        self, backtest_results: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Calculate risk-specific metrics"""
        portfolio_values = backtest_results["portfolio_values"]
        risk_adjustments = backtest_results.get("risk_adjustments", [])

        # Basic risk metrics
        returns = np.diff(portfolio_values) / portfolio_values[:-1]
        cumulative_returns = np.cumprod(1 + returns) - 1

        # Maximum drawdown
        max_drawdown_result = max_drawdown(portfolio_values)
        max_drawdown_value = max_drawdown_result

        # Sharpe ratio (assuming risk-free rate of 0)
        sharpe_ratio_value = sharpe_ratio(returns)

        # Risk adjustment statistics
        if risk_adjustments:
            original_positions = [adj["original_position"] for adj in risk_adjustments]
            adjusted_positions = [adj["adjusted_position"] for adj in risk_adjustments]
            risk_levels = [adj["risk_level"] for adj in risk_adjustments]

            position_reduction_ratio = np.mean(
                [
                    abs(adj - orig) / abs(orig) if orig != 0 else 0
                    for orig, adj in zip(original_positions, adjusted_positions)
                ]
            )
        else:
            position_reduction_ratio = 0

        risk_metrics = {
            "max_drawdown": max_drawdown_value,
            "sharpe_ratio": sharpe_ratio_value,
            "volatility": np.std(returns),
            "total_risk_adjustments": len(risk_adjustments),
            "avg_position_reduction": position_reduction_ratio,
            "risk_adjustment_frequency": len(risk_adjustments) / len(portfolio_values)
            if portfolio_values
            else 0,
        }

        return risk_metrics

    def compare_with_baseline(
        self, current_results: Dict[str, Any], baseline_results: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Compare current results with baseline"""
        logger.info("Comparing with baseline results")

        comparison = {
            "return_difference": current_results["performance_metrics"]["total_return"]
            - baseline_results["performance_metrics"]["total_return"],
            "drawdown_difference": current_results["risk_metrics"]["max_drawdown"]
            - baseline_results["risk_metrics"]["max_drawdown"],
            "sharpe_difference": current_results["risk_metrics"]["sharpe_ratio"]
            - baseline_results["risk_metrics"]["sharpe_ratio"],
            "risk_adjustment_impact": current_results["risk_metrics"][
                "avg_position_reduction"
            ],
        }

        return comparison


def main():
    """Main evaluation function"""
    evaluator = SACv435Evaluator()

    # Load model (assuming it's already trained)
    model_path = "models/v435/sac_v435_final.zip"
    if Path(model_path).exists():
        evaluator.load_model(model_path)

        # Run evaluation
        results = evaluator.evaluate_model()

        # Save results
        results_dir = Path(evaluator.config["output"]["results_dir"])
        results_dir.mkdir(parents=True, exist_ok=True)

        results_file = results_dir / "evaluation_results.json"
        with open(results_file, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False, default=str)

        print(f"Evaluation results saved to {results_file}")

        # Print summary
        perf = results["performance_metrics"]
        risk = results["risk_metrics"]

        print("\n📊 Evaluation Summary:")
        print(f"Total Return: {perf.get('total_return', 0):.2%}")
        print(f"Win Rate: {perf.get('win_rate', 0):.2%}")
        print(f"Max Drawdown: {risk.get('max_drawdown', 0):.2%}")
        print(f"Sharpe Ratio: {risk.get('sharpe_ratio', 0):.2f}")
        print(f"Risk Adjustments: {risk['total_risk_adjustments']}")

    else:
        print(f"Model not found at {model_path}. Please train the model first.")


if __name__ == "__main__":
    main()
