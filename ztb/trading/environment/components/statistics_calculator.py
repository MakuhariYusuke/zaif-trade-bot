"""
Statistics Calculator - Handles statistics calculation logic.

This module separates statistics-related logic from the main environment class,
including reward statistics, trading metrics, and performance analysis.
"""

from collections import deque
from typing import Any

import numpy as np
from numpy.typing import NDArray

from ztb.trading.constants import ACTION_BUY, ACTION_HOLD, ACTION_SELL
from ztb.trading.environment.types import EPSILON, StatisticsDict
from ztb.utils.logging_utils import get_logger

logger = get_logger(__name__)

class StatisticsCalculator:
    """
    Calculates various statistics for trading environment.

    This class handles:
    - Reward statistics (mean, std, sharpe ratio)
    - Trading metrics (win rate, total trades)
    - Performance analysis
    """

    # 409# C1: Match HeavyTradingEnv.DEFAULT_MAX_HISTORY_LENGTH to prevent
    # unbounded memory growth during long training runs.
    DEFAULT_MAX_HISTORY = 512

    def __init__(self, max_history: int | None = None):
        """Initialize StatisticsCalculator.

        Args:
            max_history: Maximum history length for deques.
                         Defaults to DEFAULT_MAX_HISTORY (512).
        """
        maxlen = max_history if max_history is not None else self.DEFAULT_MAX_HISTORY
        self.reward_history: deque[float] = deque(maxlen=maxlen)
        self.position_history: deque[float] = deque(maxlen=maxlen)
        self.portfolio_value_history: deque[float] = deque(maxlen=maxlen)
        self.action_history: deque[int] = deque(maxlen=maxlen)

    def reset(self) -> None:
        """Reset all statistics."""
        self.reward_history.clear()
        self.position_history.clear()
        self.portfolio_value_history.clear()
        self.action_history.clear()

    def add_reward(self, reward: float) -> None:
        """
        Add reward to history.

        Args:
            reward: Reward value to add

        Raises:
            TypeError: If reward is not numeric
            ValueError: If reward is invalid
        """
        try:
            if not isinstance(reward, (int, float)):
                raise TypeError(f"Reward must be numeric, got {type(reward)}")

            if not np.isfinite(reward):
                raise ValueError(f"Reward must be finite, got {reward}")

            self.reward_history.append(float(reward))

        except Exception as e:
            logger.error(f"Failed to add reward {reward}: {e}")
            raise

    def add_position(self, position: float) -> None:
        """Add position to history."""
        self.position_history.append(position)

    def add_portfolio_value(self, value: float) -> None:
        """Add portfolio value to history."""
        self.portfolio_value_history.append(value)

    def add_action(self, action: int) -> None:
        """Add action to history."""
        self.action_history.append(action)

    def get_statistics(self) -> StatisticsDict:
        """
        Calculate comprehensive statistics.

        Returns:
            Dictionary with various statistics

        Raises:
            RuntimeError: If calculation fails
        """
        try:
            if not self.reward_history:
                return {}

            rewards: NDArray[np.float64] = np.array(self.reward_history)

            stats = {
                "total_reward": float(np.sum(rewards)),
                "mean_reward": float(np.mean(rewards)),
                "std_reward": float(np.std(rewards)),
                "sharpe_ratio": float(np.mean(rewards) / (np.std(rewards) + EPSILON)),
                "max_reward": float(np.max(rewards)),
                "min_reward": float(np.min(rewards)),
                "reward_count": len(rewards),
            }

            # Trading statistics
            if self.action_history:
                buy_actions = sum(
                    1 for a in self.action_history if a == ACTION_BUY
                )  # ACTION_BUY=1
                sell_actions = sum(
                    1
                    for a in self.action_history
                    if a == ACTION_SELL or a == 2  # Legacy support (2=SELL)
                )  # ACTION_SELL=-1
                hold_actions = sum(
                    1 for a in self.action_history if a == ACTION_HOLD
                )  # ACTION_HOLD=0

                stats.update({
                    "total_actions": len(self.action_history),
                    "buy_actions": buy_actions,
                    "sell_actions": sell_actions,
                    "hold_actions": hold_actions,
                    "action_distribution": {
                        "buy": buy_actions / len(self.action_history),
                        "sell": sell_actions / len(self.action_history),
                        "hold": hold_actions / len(self.action_history),
                    }
                })

            # Position statistics
            if self.position_history:
                positions = np.array(self.position_history)
                stats.update({
                    "mean_position": float(np.mean(np.abs(positions))),
                    "max_position": float(np.max(np.abs(positions))),
                    "position_changes": int(np.sum(np.diff(positions) != 0)),
                })

            # Portfolio statistics
            if self.portfolio_value_history:
                portfolio_values = np.array(self.portfolio_value_history)
                if len(portfolio_values) > 1:
                    returns = np.diff(portfolio_values) / portfolio_values[:-1]
                    stats.update({
                        "portfolio_return": float((portfolio_values[-1] - portfolio_values[0]) / portfolio_values[0]),
                        "portfolio_volatility": float(np.std(returns)) if len(returns) > 0 else 0.0,
                        "max_portfolio_value": float(np.max(portfolio_values)),
                        "min_portfolio_value": float(np.min(portfolio_values)),
                    })

            # Win rate calculation
            positive_rewards = np.count_nonzero(rewards > 0)
            stats["win_rate"] = float(positive_rewards / len(rewards)) if len(rewards) > 0 else 0.0

            return stats

        except Exception as e:
            logger.error(f"Failed to calculate statistics: {e}")
            raise RuntimeError(f"Statistics calculation failed: {e}") from e

    def get_trades_per_1k_steps(self, current_step: int) -> float:
        """
        Calculate trades per 1000 steps.

        Args:
            current_step: Current step number

        Returns:
            Trades per 1000 steps
        """
        if current_step == 0:
            return 0.0

        # Count actual trades (non-hold actions)
        trade_actions = sum(1 for a in self.action_history if a != 0)  # ACTION_HOLD = 0
        return trade_actions / (current_step / 1000)

    def get_recent_performance(self, window: int = 100) -> dict[str, Any]:
        """
        Get recent performance statistics.

        Args:
            window: Number of recent steps to analyze

        Returns:
            Recent performance metrics
        """
        if len(self.reward_history) < window:
            return self.get_statistics()

        recent_rewards = list(self.reward_history)[-window:]
        recent_rewards_array = np.array(recent_rewards)

        return {
            "recent_mean_reward": float(np.mean(recent_rewards_array)),
            "recent_std_reward": float(np.std(recent_rewards_array)),
            "recent_win_rate": float(np.count_nonzero(recent_rewards_array > 0) / len(recent_rewards_array)),
            "recent_total_reward": float(np.sum(recent_rewards_array)),
            "window_size": window,
        }

    def get_action_frequencies(self) -> dict[int, float]:
        """
        Get action frequency distribution.

        Returns:
            Dictionary mapping action to frequency
        """
        if not self.action_history:
            return {}

        total_actions = len(self.action_history)
        frequencies = {}

        for action in set(self.action_history):
            count = sum(1 for a in self.action_history if a == action)
            frequencies[action] = count / total_actions

        return frequencies
