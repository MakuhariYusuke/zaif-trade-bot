"""
Long-term Metrics for SAC v448.

Evaluation metrics for assessing long-term sustainability and risk-adjusted returns.

Version: 1.0
Created: 2025-11-22
Author: SAC v448 Development Team
"""

import logging
from typing import List, Optional

import numpy as np

from ztb.metrics import max_drawdown, sharpe_ratio
from ztb.trading.constants import ACTION_BUY, ACTION_HOLD, ACTION_SELL


class LongTermMetrics:
    """
    Long-term sustainability and performance metrics.

    Provides metrics beyond simple final reward:
    - Sharpe Ratio: Risk-adjusted returns
    - Max Drawdown: Largest peak-to-trough decline
    - Action Balance Stability: Consistency of action distribution over time
    - Transaction Cost Efficiency: How much cost relative to gross PnL
    - Sustainable Profitability Score: Combined metric favoring balanced strategies

    These metrics help identify strategies that achieve high returns through
    extreme biases (unsustainable) vs. balanced approaches (sustainable).

    Usage:
        metrics = LongTermMetrics()

        # Calculate Sharpe ratio
        sharpe = metrics.sharpe_ratio(episode_returns, risk_free_rate=0.0)

        # Calculate max drawdown
        max_dd = metrics.max_drawdown(portfolio_values)

        # Assess action balance stability
        stability = metrics.action_balance_stability(action_history, window=100)

        # Combined sustainability score
        score = metrics.sustainable_profitability_score(
            final_reward=8.5,
            balance_stability=0.08,
            max_dd=-0.15,
            sharpe=1.2
        )
    """

    def __init__(self, logger: Optional[logging.Logger] = None) -> None:
        """
        Initialize long-term metrics calculator.

        Args:
            logger: Optional logger instance
        """
        self.logger = logger or logging.getLogger(__name__)

    @staticmethod
    def sharpe_ratio(
        returns: np.ndarray,
        risk_free_rate: float = 0.0,
        annualization_factor: float = 1.0,
    ) -> float:
        """
        Calculate Sharpe ratio (risk-adjusted return).

        Sharpe Ratio = (Mean Return - Risk Free Rate) / Std Dev of Returns

        Higher is better. Typical values:
            > 2.0: Excellent
            > 1.0: Good
            > 0.5: Acceptable
            < 0.0: Poor (negative excess returns)

        Args:
            returns: Array of period returns (e.g., per-episode PnL)
            risk_free_rate: Risk-free rate (default: 0.0)
            annualization_factor: Factor to annualize (default: 1.0 for no annualization)

        Returns:
            Sharpe ratio. Returns 0.0 if std is zero or insufficient data.
        """
        return sharpe_ratio(
            returns, rf=risk_free_rate, period_per_year=int(annualization_factor)
        )

    @staticmethod
    def max_drawdown(portfolio_values: np.ndarray) -> float:
        """
        Calculate maximum drawdown (largest peak-to-trough decline).

        Max Drawdown = max((Peak - Trough) / Peak) over all peaks

        Lower (more negative) is worse:
            > -10%: Excellent
            > -20%: Good
            > -30%: Acceptable
            < -50%: Poor (high risk)

        Args:
            portfolio_values: Array of portfolio values over time

        Returns:
            Maximum drawdown as negative decimal (e.g., -0.25 for 25% drawdown).
            Returns 0.0 if no drawdown or insufficient data.
        """
        return max_drawdown(portfolio_values)

    @staticmethod
    def action_balance_stability(action_history: List[int], window: int = 100) -> float:
        """
        Measure stability of action distribution over time.

        Calculates variance of action distributions across sliding windows.
        Lower is better (more consistent behavior).

        Typical values:
            < 0.05: Excellent (very stable)
            < 0.10: Good (stable)
            < 0.20: Acceptable
            > 0.30: Poor (erratic behavior)

        Args:
            action_history: List of actions (0=HOLD, 1=BUY, 2=SELL)
            window: Window size for calculating distributions

        Returns:
            Stability score (0.0 = perfectly stable, higher = more variable).
            Returns 0.0 if insufficient data.
        """
        if len(action_history) < window * 2:
            return 0.0

        n_windows = len(action_history) // window
        distributions = []

        for i in range(n_windows):
            window_actions = action_history[i * window : (i + 1) * window]

            # Calculate distribution [HOLD%, BUY%, SELL%]
            dist = [
                window_actions.count(ACTION_HOLD) / window,  # HOLD
                window_actions.count(ACTION_BUY) / window,  # BUY
                window_actions.count(ACTION_SELL) / window,  # SELL
            ]
            distributions.append(dist)

        # Calculate variance across windows for each action type
        distributions = np.array(distributions)
        variances = distributions.var(axis=0)

        # Mean variance across action types
        mean_variance = variances.mean()

        return mean_variance if np.isfinite(mean_variance) else 0.0

    @staticmethod
    def transaction_cost_efficiency(
        gross_pnl: float, transaction_costs: float
    ) -> float:
        """
        Calculate transaction cost efficiency.

        Efficiency = Transaction Costs / |Gross PnL|

        Lower is better:
            < 10%: Excellent (efficient trading)
            < 20%: Good
            < 30%: Acceptable
            > 50%: Poor (costs eating profits)

        Args:
            gross_pnl: Gross profit/loss before costs
            transaction_costs: Total transaction costs

        Returns:
            Cost efficiency ratio. Returns 1.0 if gross_pnl is zero.
        """
        if abs(gross_pnl) < 1e-6:
            return 1.0  # Worst case: all cost, no profit

        efficiency = abs(transaction_costs / gross_pnl)

        return efficiency if np.isfinite(efficiency) else 1.0

    def sustainable_profitability_score(
        self,
        final_reward: float,
        balance_stability: float,
        max_dd: float,
        sharpe: float,
        weights: Optional[dict] = None,
    ) -> float:
        """
        Combined score favoring sustainable strategies.

        Combines multiple metrics with configurable weights:
        - Final reward (default 40%): Absolute performance
        - Balance stability (default 20%, inverted): Consistency
        - Max drawdown (default 20%, inverted): Risk control
        - Sharpe ratio (default 20%): Risk-adjusted return

        Higher is better. Typical values:
            > 0.7: Excellent (sustainable + profitable)
            > 0.5: Good
            > 0.3: Acceptable
            < 0.3: Poor (either unprofitable or unsustainable)

        Args:
            final_reward: Episode final reward
            balance_stability: From action_balance_stability()
            max_dd: From max_drawdown() (negative value)
            sharpe: From sharpe_ratio()
            weights: Optional custom weights dict with keys:
                    'reward', 'stability', 'drawdown', 'sharpe'

        Returns:
            Sustainability score in [0, 1+]
        """
        # Default weights
        default_weights = {
            "reward": 0.40,
            "stability": 0.20,
            "drawdown": 0.20,
            "sharpe": 0.20,
        }
        w = weights if weights is not None else default_weights

        # Normalize components to [0, 1] range

        # 1. Reward score (normalize to 0-1, assuming reward ∈ [-10, 10])
        reward_score = np.clip((final_reward + 10) / 20, 0, 1)

        # 2. Stability score (invert: lower stability variance is better)
        # Assuming balance_stability ∈ [0, 0.5]
        stability_score = np.clip(1.0 - balance_stability * 10, 0, 1)

        # 3. Drawdown score (invert: less negative is better)
        # Assuming max_dd ∈ [-1, 0]
        dd_score = np.clip(1.0 + max_dd, 0, 1)

        # 4. Sharpe score (normalize: assuming sharpe ∈ [-2, 4])
        sharpe_score = np.clip((sharpe + 2) / 6, 0, 1)

        # Combined weighted score
        combined = (
            w["reward"] * reward_score
            + w["stability"] * stability_score
            + w["drawdown"] * dd_score
            + w["sharpe"] * sharpe_score
        )

        self.logger.debug(
            f"Sustainability score: {combined:.3f} "
            f"(reward={reward_score:.3f}, stability={stability_score:.3f}, "
            f"dd={dd_score:.3f}, sharpe={sharpe_score:.3f})"
        )

        return combined

    def evaluate_episode(self, episode_data: dict) -> dict:
        """
        Comprehensive episode evaluation.

        Args:
            episode_data: Dictionary containing:
                - final_reward: float
                - portfolio_values: np.ndarray
                - action_history: List[int]
                - gross_pnl: float (optional)
                - transaction_costs: float (optional)

        Returns:
            Dictionary with all calculated metrics
        """
        results = {
            "final_reward": episode_data["final_reward"],
        }

        # Sharpe ratio (if multiple returns available)
        if "episode_returns" in episode_data:
            results["sharpe_ratio"] = self.sharpe_ratio(episode_data["episode_returns"])

        # Max drawdown
        if "portfolio_values" in episode_data:
            results["max_drawdown"] = self.max_drawdown(
                episode_data["portfolio_values"]
            )

        # Balance stability
        if "action_history" in episode_data:
            results["balance_stability"] = self.action_balance_stability(
                episode_data["action_history"]
            )

        # Transaction cost efficiency
        if "gross_pnl" in episode_data and "transaction_costs" in episode_data:
            results["cost_efficiency"] = self.transaction_cost_efficiency(
                episode_data["gross_pnl"], episode_data["transaction_costs"]
            )

        # Sustainable profitability score (if all metrics available)
        if all(
            k in results for k in ["sharpe_ratio", "max_drawdown", "balance_stability"]
        ):
            results["sustainability_score"] = self.sustainable_profitability_score(
                final_reward=results["final_reward"],
                balance_stability=results["balance_stability"],
                max_dd=results["max_drawdown"],
                sharpe=results["sharpe_ratio"],
            )

        return results
