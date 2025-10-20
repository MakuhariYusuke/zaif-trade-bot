#!/usr/bin/env python3
"""
Reward Function Parameter Evaluator

This module provides evaluation functions for reward function parameter optimization,
including multi-objective scoring and cross-validation across market conditions.
"""

import json
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from ztb.trading.environment.utils.config import RewardSettings
from ztb.utils.logging_utils import get_logger

logger = get_logger(__name__)


@dataclass
class EvaluationMetrics:
    """Metrics for reward function evaluation."""

    total_return: float = 0.0
    sharpe_ratio: float = 0.0
    win_rate: float = 0.0
    max_drawdown: float = 0.0
    volatility: float = 0.0
    consistency_score: float = 0.0
    profit_factor: float = 0.0
    calmar_ratio: float = 0.0
    sortino_ratio: float = 0.0
    recovery_factor: float = 0.0


@dataclass
class EvaluationResult:
    """Result of reward function parameter evaluation."""

    metrics: EvaluationMetrics
    trade_history: List[Dict[str, Any]] = field(default_factory=list)
    portfolio_history: List[Dict[str, Any]] = field(default_factory=list)
    evaluation_time: float = 0.0
    market_conditions: Dict[str, Any] = field(default_factory=dict)


class RewardFunctionEvaluator:
    """
    Evaluator for reward function parameters.

    This class provides:
    - Parameter evaluation across different market conditions
    - Multi-objective scoring (profit, risk, consistency)
    - Cross-validation and robustness testing
    - Performance metrics calculation
    """

    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize reward function evaluator.

        Args:
            config_path: Path to configuration file
        """
        self.config_path = config_path or "configs/reward_evaluation.json"
        self.logger = get_logger("ztb.optimization.reward_evaluator")

        # Load default evaluation settings
        self.evaluation_settings = self._load_evaluation_settings()

    def _load_evaluation_settings(self) -> Dict[str, Any]:
        """Load evaluation settings from config file."""
        default_settings = {
            "evaluation_episodes": 10,
            "max_steps_per_episode": 1000,
            "market_conditions": ["bull", "bear", "sideways", "volatile"],
            "risk_free_rate": 0.02,
            "benchmark_return": 0.05,
            "consistency_window": 50,
            "min_trades_for_evaluation": 10,
        }

        if self.config_path and Path(self.config_path).exists():
            with open(self.config_path, "r", encoding="utf-8") as f:
                loaded_settings = json.load(f)
                default_settings.update(loaded_settings)

        return default_settings

    def evaluate_parameters(
        self,
        parameters: Dict[str, Any],
        stage: str,
        market_data: Optional[pd.DataFrame] = None,
        n_episodes: Optional[int] = None,
        max_steps: Optional[int] = None,
    ) -> EvaluationResult:
        """
        Evaluate reward function parameters.

        Args:
            parameters: Parameters to evaluate
            stage: Reward function stage
            market_data: Market data for evaluation (optional)
            n_episodes: Number of evaluation episodes
            max_steps: Maximum steps per episode

        Returns:
            Evaluation result with metrics and history
        """

        start_time = time.time()

        n_episodes = n_episodes or self.evaluation_settings["evaluation_episodes"]
        max_steps = max_steps or self.evaluation_settings["max_steps_per_episode"]

        # Initialize reward settings with parameters
        reward_settings = self._create_reward_settings(parameters, stage)

        all_trade_history = []
        all_portfolio_history = []
        all_metrics = []

        # Evaluate across different market conditions
        market_conditions = self.evaluation_settings["market_conditions"]

        for condition in market_conditions:
            self.logger.info(
                f"Evaluating parameters under {condition} market conditions"
            )

            # Run evaluation episodes for this condition
            (
                condition_trade_history,
                condition_portfolio_history,
            ) = self._run_evaluation_episodes(
                reward_settings, condition, n_episodes, max_steps, market_data
            )

            # Calculate metrics for this condition
            condition_metrics = self._calculate_metrics(
                condition_trade_history, condition_portfolio_history, condition
            )

            all_trade_history.extend(condition_trade_history)
            all_portfolio_history.extend(condition_portfolio_history)
            all_metrics.append(condition_metrics)

        # Aggregate metrics across all conditions
        aggregated_metrics = self._aggregate_metrics(all_metrics)

        result = EvaluationResult(
            metrics=aggregated_metrics,
            trade_history=all_trade_history,
            portfolio_history=all_portfolio_history,
            evaluation_time=time.time() - start_time,
            market_conditions={
                "conditions_tested": market_conditions,
                "episodes_per_condition": n_episodes,
                "total_episodes": len(market_conditions) * n_episodes,
            },
        )

        self.logger.info(
            f"Parameter evaluation completed in {result.evaluation_time:.2f} seconds"
        )
        self.logger.info(f"Aggregated metrics: {aggregated_metrics}")

        return result

    def _create_reward_settings(
        self, parameters: Dict[str, Any], stage: str
    ) -> RewardSettings:
        """Create reward settings from parameters."""
        # Create default settings and update with optimized parameters
        settings: RewardSettings = {
            "position_soft_cap": 0.8,
            "position_penalty_scale": 0.1,
            "position_penalty_exp": 2.0,
            "inventory_window": 10,
            "inventory_penalty_scale": 0.01,
            "trade_frequency_penalty": 0.001,
            "trade_frequency_halflife": 50.0,
            "trade_cooldown_steps": 5,
            "trade_cooldown_penalty": 0.01,
            "max_consecutive_trades": 10,
            "consecutive_trade_penalty": 0.1,
            "volatility_window": 20,
            "volatility_penalty_scale": 0.01,
            "sharpe_bonus_scale": 0.1,
            "sortino_bonus_scale": 0.1,
            "calmar_bonus_scale": 0.1,
            "reward_clip_value": 10.0,
            "profit_bonus_multipliers": [1.0, 1.2, 1.5, 2.0],
            "enable_forced_diversity": False,
            "custom_reward_params": {},
        }

        # Map parameters to settings based on stage
        if stage == "balanced_transition":
            settings.update(
                {
                    "balance_penalty_tolerance": parameters.get(
                        "balance_penalty_tolerance", 0.05
                    ),
                    "balance_penalty": parameters.get("balance_penalty", 5.0),
                    "custom_reward_params": {
                        "hold_penalty_rate": parameters.get("hold_penalty_rate", 0.01),
                        "trading_bonus_multiplier": parameters.get(
                            "trading_bonus_multiplier", 2.0
                        ),
                        "trading_bonus": parameters.get("trading_bonus", 0.01),
                        "profit_weight": parameters.get("profit_weight", 1.0),
                        "risk_weight": parameters.get("risk_weight", 1.0),
                        "consistency_weight": parameters.get("consistency_weight", 1.0),
                    },
                }
            )

        elif stage == "trading_focused":
            settings.update(
                {
                    "balance_penalty_tolerance": parameters.get(
                        "balance_penalty_tolerance", 0.05
                    ),
                    "balance_penalty": parameters.get("balance_penalty", 10.0),
                    "custom_reward_params": {
                        "hold_penalty_rate": parameters.get("hold_penalty_rate", 0.05),
                        "trading_bonus_multiplier": parameters.get(
                            "trading_bonus_multiplier", 3.0
                        ),
                        "trading_bonus": parameters.get("trading_bonus", 0.05),
                        "profit_weight": parameters.get("profit_weight", 1.0),
                        "risk_weight": parameters.get("risk_weight", 1.0),
                        "consistency_weight": parameters.get("consistency_weight", 1.0),
                    },
                }
            )

        elif stage == "profit_optimized":
            settings.update(
                {
                    "custom_reward_params": {
                        "profit_weight": parameters.get("profit_weight", 2.0),
                        "risk_weight": parameters.get("risk_weight", 0.1),
                        "consistency_weight": parameters.get("consistency_weight", 0.1),
                        "position_penalty_weight": parameters.get(
                            "position_penalty_weight", 0.01
                        ),
                        "drawdown_penalty_weight": parameters.get(
                            "drawdown_penalty_weight", 0.01
                        ),
                        "stagnation_penalty_weight": parameters.get(
                            "stagnation_penalty_weight", 0.01
                        ),
                        "growth_bonus_weight": parameters.get(
                            "growth_bonus_weight", 0.01
                        ),
                        "win_streak_bonus_weight": parameters.get(
                            "win_streak_bonus_weight", 0.01
                        ),
                    }
                }
            )

        elif stage == "ultra_profit":
            settings.update(
                {
                    "custom_reward_params": {
                        "profit_weight": parameters.get("profit_weight", 5.0),
                        "risk_weight": parameters.get("risk_weight", 0.01),
                        "consistency_weight": parameters.get(
                            "consistency_weight", 0.01
                        ),
                        "ultra_profit_multiplier": parameters.get(
                            "ultra_profit_multiplier", 2.0
                        ),
                        "ultra_risk_multiplier": parameters.get(
                            "ultra_risk_multiplier", 0.5
                        ),
                    }
                }
            )

        return settings

    def _run_evaluation_episodes(
        self,
        reward_settings: RewardSettings,
        market_condition: str,
        n_episodes: int,
        max_steps: int,
        market_data: Optional[pd.DataFrame] = None,
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """
        Run evaluation episodes for a specific market condition.

        Args:
            reward_settings: Reward settings for this evaluation
            market_condition: Market condition for this evaluation
            n_episodes: Number of episodes to run
            max_steps: Maximum steps per episode
            market_data: Market data (optional)

        Returns:
            Tuple of (trade_history, portfolio_history)
        """
        # This is a simplified implementation
        # In practice, this would integrate with the actual trading environment

        trade_history = []
        portfolio_history = []

        for episode in range(n_episodes):
            episode_trades = []
            episode_portfolio = []

            # Simulate episode with random but realistic trading behavior
            # This is a placeholder - actual implementation would use the trading environment

            # Generate synthetic trading data based on market condition
            synthetic_data = self._generate_synthetic_episode_data(
                market_condition, max_steps
            )

            for step in range(max_steps):
                # Simulate a trade decision and reward calculation
                action = self._simulate_trading_decision(synthetic_data[step])

                # Calculate reward using the reward settings
                # Note: This is simplified - actual integration would require proper state management
                reward = self._calculate_synthetic_reward(
                    action, synthetic_data[step], reward_settings
                )

                # Record trade and portfolio state
                trade_record = {
                    "episode": episode,
                    "step": step,
                    "action": action,
                    "reward": reward,
                    "market_condition": market_condition,
                    "timestamp": datetime.now().isoformat(),
                }
                episode_trades.append(trade_record)

                portfolio_record = {
                    "episode": episode,
                    "step": step,
                    "portfolio_value": synthetic_data[step].get(
                        "portfolio_value", 10000
                    ),
                    "cash": synthetic_data[step].get("cash", 5000),
                    "position": synthetic_data[step].get("position", 0),
                    "market_condition": market_condition,
                }
                episode_portfolio.append(portfolio_record)

            trade_history.extend(episode_trades)
            portfolio_history.extend(episode_portfolio)

        return trade_history, portfolio_history

    def _generate_synthetic_episode_data(
        self, market_condition: str, max_steps: int
    ) -> List[Dict[str, Any]]:
        """Generate synthetic episode data for evaluation."""
        # Simplified synthetic data generation
        data = []

        base_price = 100.0
        portfolio_value = 10000.0
        cash = 5000.0
        position = 0

        for step in range(max_steps):
            # Generate price movement based on market condition
            if market_condition == "bull":
                price_change = np.random.normal(0.001, 0.01)  # Slight upward trend
            elif market_condition == "bear":
                price_change = np.random.normal(-0.001, 0.01)  # Slight downward trend
            elif market_condition == "sideways":
                price_change = np.random.normal(0.0, 0.005)  # No trend
            elif market_condition == "volatile":
                price_change = np.random.normal(0.0, 0.02)  # High volatility
            else:
                price_change = np.random.normal(0.0, 0.01)

            base_price *= 1 + price_change

            # Simulate portfolio changes
            portfolio_value *= 1 + price_change * 0.1  # Simplified P&L

            data.append(
                {
                    "step": step,
                    "price": base_price,
                    "portfolio_value": portfolio_value,
                    "cash": cash,
                    "position": position,
                    "market_condition": market_condition,
                }
            )

        return data

    def _simulate_trading_decision(self, market_data: Dict[str, Any]) -> Dict[str, Any]:
        """Simulate a trading decision for evaluation."""
        # Simplified trading decision simulation
        # In practice, this would use a trained model or heuristic

        actions = ["hold", "buy", "sell"]
        action = np.random.choice(actions, p=[0.7, 0.15, 0.15])  # Mostly hold

        return {
            "action": action,
            "amount": np.random.uniform(0.1, 1.0) if action != "hold" else 0.0,
            "price": market_data["price"],
        }

    def _calculate_synthetic_reward(
        self,
        action: Dict[str, Any],
        market_data: Dict[str, Any],
        reward_settings: RewardSettings,
    ) -> float:
        """Calculate synthetic reward for evaluation (simplified implementation)."""
        # This is a simplified reward calculation for testing
        # In practice, this would use the actual RewardCalculator with proper state management

        base_reward = 0.0

        if action["action"] == "buy":
            # Reward for buying (simplified)
            trading_bonus = reward_settings.get("custom_reward_params", {}).get(
                "trading_bonus", 0.01
            )
            base_reward = np.random.normal(trading_bonus, 0.05)
        elif action["action"] == "sell":
            # Reward for selling (simplified)
            trading_bonus_multiplier = reward_settings.get(
                "custom_reward_params", {}
            ).get("trading_bonus_multiplier", 2.0)
            base_reward = np.random.normal(trading_bonus_multiplier * 0.01, 0.08)
        else:  # hold
            # Small penalty for holding
            hold_penalty_rate = reward_settings.get("custom_reward_params", {}).get(
                "hold_penalty_rate", 0.01
            )
            base_reward = np.random.normal(-hold_penalty_rate, 0.02)

        return base_reward

    def _calculate_metrics(
        self,
        trade_history: List[Dict[str, Any]],
        portfolio_history: List[Dict[str, Any]],
        market_condition: str,
    ) -> EvaluationMetrics:
        """Calculate performance metrics from trade and portfolio history."""

        if not trade_history or not portfolio_history:
            return EvaluationMetrics()

        # Extract portfolio values
        portfolio_values = [record["portfolio_value"] for record in portfolio_history]
        returns = np.diff(portfolio_values) / portfolio_values[:-1]

        # Calculate basic metrics
        total_return = (portfolio_values[-1] / portfolio_values[0]) - 1.0

        # Sharpe ratio
        risk_free_rate = self.evaluation_settings["risk_free_rate"]
        if len(returns) > 1:
            excess_returns = returns - risk_free_rate / 252  # Daily risk-free rate
            sharpe_ratio = (
                np.mean(excess_returns) / np.std(excess_returns) * np.sqrt(252)
            )
        else:
            sharpe_ratio = 0.0

        # Win rate
        winning_trades = [t for t in trade_history if t.get("reward", 0) > 0]
        win_rate = len(winning_trades) / len(trade_history) if trade_history else 0.0

        # Maximum drawdown
        peak = portfolio_values[0]
        max_drawdown = 0.0
        for value in portfolio_values:
            if value > peak:
                peak = value
            drawdown = (peak - value) / peak
            max_drawdown = max(max_drawdown, drawdown)

        # Volatility
        volatility = np.std(returns) * np.sqrt(252) if len(returns) > 1 else 0.0

        # Consistency score (based on return stability)
        if len(returns) > self.evaluation_settings["consistency_window"]:
            rolling_returns = (
                pd.Series(returns)
                .rolling(self.evaluation_settings["consistency_window"])
                .mean()
            )
            consistency_score = 1.0 / (1.0 + np.std(rolling_returns.dropna()))
        else:
            consistency_score = 0.5  # Neutral score for short histories

        # Profit factor
        gross_profit = sum(
            t.get("reward", 0) for t in trade_history if t.get("reward", 0) > 0
        )
        gross_loss = abs(
            sum(t.get("reward", 0) for t in trade_history if t.get("reward", 0) < 0)
        )
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float("inf")

        # Calmar ratio
        calmar_ratio = total_return / max_drawdown if max_drawdown > 0 else float("inf")

        # Sortino ratio (downside deviation)
        downside_returns = [r for r in returns if r < 0]
        if downside_returns:
            sortino_ratio = np.mean(returns) / np.std(downside_returns) * np.sqrt(252)
        else:
            sortino_ratio = float("inf")

        # Recovery factor
        recovery_factor = (
            total_return / max_drawdown if max_drawdown > 0 else float("inf")
        )

        return EvaluationMetrics(
            total_return=total_return,
            sharpe_ratio=sharpe_ratio,
            win_rate=win_rate,
            max_drawdown=max_drawdown,
            volatility=volatility,
            consistency_score=consistency_score,
            profit_factor=profit_factor,
            calmar_ratio=calmar_ratio,
            sortino_ratio=sortino_ratio,
            recovery_factor=recovery_factor,
        )

    def _aggregate_metrics(
        self, metrics_list: List[EvaluationMetrics]
    ) -> EvaluationMetrics:
        """Aggregate metrics across different market conditions."""

        if not metrics_list:
            return EvaluationMetrics()

        # Simple average aggregation
        # Could be weighted by market condition importance
        aggregated = EvaluationMetrics()

        for metric in metrics_list:
            aggregated.total_return += metric.total_return
            aggregated.sharpe_ratio += metric.sharpe_ratio
            aggregated.win_rate += metric.win_rate
            aggregated.max_drawdown += metric.max_drawdown
            aggregated.volatility += metric.volatility
            aggregated.consistency_score += metric.consistency_score
            aggregated.profit_factor += metric.profit_factor
            aggregated.calmar_ratio += metric.calmar_ratio
            aggregated.sortino_ratio += metric.sortino_ratio
            aggregated.recovery_factor += metric.recovery_factor

        n_conditions = len(metrics_list)
        aggregated.total_return /= n_conditions
        aggregated.sharpe_ratio /= n_conditions
        aggregated.win_rate /= n_conditions
        aggregated.max_drawdown /= n_conditions
        aggregated.volatility /= n_conditions
        aggregated.consistency_score /= n_conditions
        aggregated.profit_factor /= n_conditions
        aggregated.calmar_ratio /= n_conditions
        aggregated.sortino_ratio /= n_conditions
        aggregated.recovery_factor /= n_conditions

        return aggregated

    def create_evaluation_function(
        self, stage: str
    ) -> Callable[[Dict[str, Any]], Dict[str, float]]:
        """
        Create evaluation function for optimization.

        Args:
            stage: Reward function stage

        Returns:
            Function that evaluates parameters and returns scores
        """

        def evaluation_function(parameters: Dict[str, Any]) -> Dict[str, float]:
            """Evaluate parameters and return scores for optimization."""
            result = self.evaluate_parameters(parameters, stage)

            # Return scores in format expected by optimizer
            scores = {
                "profit": result.metrics.total_return,
                "sharpe": result.metrics.sharpe_ratio,
                "win_rate": result.metrics.win_rate,
                "consistency": result.metrics.consistency_score,
                "max_drawdown": -result.metrics.max_drawdown,  # Negative for minimization
                "volatility": -result.metrics.volatility,  # Negative for minimization
                "profit_factor": result.metrics.profit_factor,
                "calmar_ratio": result.metrics.calmar_ratio,
                "sortino_ratio": result.metrics.sortino_ratio,
                "recovery_factor": result.metrics.recovery_factor,
            }

            return scores

        return evaluation_function
