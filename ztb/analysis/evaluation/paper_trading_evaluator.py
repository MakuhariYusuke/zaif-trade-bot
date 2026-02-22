#!/usr/bin/env python3
"""
Paper Trading Evaluator - Integrated paper trading functionality

This module provides comprehensive paper trading evaluation capabilities
integrated from archived paper trading scripts.
"""

from __future__ import annotations

import warnings
from typing import Any, Dict, Optional

from ztb.trading.environment.utils.config import EnvironmentConfig
from ztb.utils.logging_utils import get_logger
from ztb.evaluation.paper_trading import (
    _create_environment,
    _load_data,
    _load_model,
    evaluate_paper_trading,
    run_paper_trading,
)

warnings.warn(
    "PaperTradingEvaluator is deprecated; "
    "use ztb.evaluation.unified_evaluation with EvaluationType.PAPER_TRADING.",
    DeprecationWarning,
    stacklevel=2,
)


class PaperTradingEvaluator:
    """Paper trading evaluator for comprehensive model evaluation."""

    def __init__(self, config: Optional[EnvironmentConfig] = None):
        """Initialize paper trading evaluator.

        Args:
            config: Environment configuration for paper trading
        """
        self.logger = get_logger(__name__)
        self.config = config or EnvironmentConfig()

        # Set default paper trading configuration
        self.config.initial_portfolio_value = 200000.0
        self.config.transaction_cost = 1e-05
        self.config.max_position_size = 1.0
        self.config.use_standardized_observations = True
        self.config.curriculum_stage = "profit_optimized"
        self.config.use_continuous_actions = True

    def load_model(self, model_path: str) -> SAC:
        """Load the trained SAC model.

        Args:
            model_path: Path to the trained model file

        Returns:
            Loaded SAC model
        """
        return _load_model(model_path)

    def load_data(self, data_path: str) -> pd.DataFrame:
        """Load BTC/JPY data for paper trading.

        Args:
            data_path: Path to the data file

        Returns:
            Loaded and processed DataFrame
        """
        return _load_data(data_path)

    def create_environment(self, df: pd.DataFrame) -> HeavyTradingEnv:
        """Create environment for paper trading.

        Args:
            df: DataFrame with trading data

        Returns:
            Configured HeavyTradingEnv instance
        """
        return _create_environment(df, self.config)

    def run_paper_trading(
        self,
        model: SAC,
        env: HeavyTradingEnv,
        num_episodes: int = 10,
        max_steps_per_episode: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Run paper trading simulation.

        Args:
            model: Trained SAC model
            env: Trading environment
            num_episodes: Number of episodes to run
            max_steps_per_episode: Maximum steps per episode

        Returns:
            Dictionary containing paper trading results
        """
        return run_paper_trading(
            model,
            env,
            num_episodes=num_episodes,
            max_steps_per_episode=max_steps_per_episode,
        )

    def evaluate_model(
        self,
        model_path: str,
        data_path: str,
        num_episodes: int = 10,
        output_path: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Complete paper trading evaluation pipeline.

        Args:
            model_path: Path to the trained model
            data_path: Path to the evaluation data
            num_episodes: Number of episodes to run
            output_path: Optional path to save results

        Returns:
            Paper trading evaluation results
        """
        return evaluate_paper_trading(
            model_path=model_path,
            data_path=data_path,
            num_episodes=num_episodes,
            env_config=self.config,
            output_path=output_path,
        )

    def print_summary(self, results: Dict[str, Any]) -> None:
        """Print formatted summary of paper trading results.

        Args:
            results: Paper trading results dictionary
        """
        self.logger.info("\n" + "=" * 60)
        self.logger.info("PAPER TRADING EVALUATION RESULTS")
        self.logger.info("=" * 60)
        self.logger.info(
            "Average Reward: %.2f ± %.2f",
            results["avg_reward"],
            results["std_reward"],
        )
        self.logger.info(
            "Average Portfolio Value: %.2f ± %.2f",
            results["avg_portfolio_value"],
            results["std_portfolio_value"],
        )
        self.logger.info(
            "HOLD: %.1f%%, BUY: %.1f%%, SELL: %.1f%%",
            results["action_distribution_percent"]["HOLD"],
            results["action_distribution_percent"]["BUY"],
            results["action_distribution_percent"]["SELL"],
        )
        self.logger.info("=" * 60)
