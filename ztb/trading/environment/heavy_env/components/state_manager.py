"""State management component for HeavyTradingEnv."""

from typing import TYPE_CHECKING, Any, Dict, List, Optional

import numpy as np
from numpy.typing import NDArray

from ztb.utils.logging_utils import get_logger

if TYPE_CHECKING:
    from ztb.trading.environment.heavy_env.core import HeavyTradingEnv


class StateManager:
    """Manages trading environment state (position, prices, PnL, etc.)."""

    def __init__(self, env: "HeavyTradingEnv"):
        """Initialize state manager with reference to environment."""
        self.env = env
        self.logger = get_logger(__name__)

    def reset_state(self) -> None:
        """Reset all state variables for a new episode."""
        # Position and trading state
        self.env.position = 0.0
        self.env.entry_price = 0.0
        self.env.total_pnl = 0.0
        self.env.trades_count = 0
        self.env.realized_pnl = 0.0

        # History buffers
        self.env.reward_history.clear()
        self.env.position_history.clear()
        self.env.position_abs_history.clear()
        self.env.portfolio_value_history.clear()
        self.env.pnl_history.clear()
        self.env.trade_interval_history.clear()
        self.env.action_history.clear()

        # Episode tracking
        self.env._current_episode_actions.clear()
        self.env._action_counts = self.env.ACTION_COUNTS_INITIAL.copy()

        # Streaming state
        self.env._stream_last_timestamp = None
        self.env._stream_rows_appended = 0
        self.env._last_trade_step = None
        self.env._consecutive_trade_steps = 0

        # Portfolio state
        self.env.portfolio_value = self.env.initial_portfolio_value
        self.env._previous_portfolio_value = None

        self.logger.debug("Environment state reset")

    def update_position_state(self, action: int, current_step: int, trade_pnl: float) -> None:
        """Update position-related state after action execution.

        Args:
            action: Action taken
            current_step: Current step
            trade_pnl: PnL from the trade
        """
        # Update action tracking
        self.env._current_episode_actions.append(action)
        self.env.action_history.append(action)

        # Update trade intervals
        if self.env._last_trade_step is not None:
            interval = current_step - self.env._last_trade_step
            self.env.trade_interval_history.append(interval)

        # Update consecutive trades
        if action != 0:  # Not HOLD
            if self.env._last_trade_step == current_step - 1:
                self.env._consecutive_trade_steps += 1
            else:
                self.env._consecutive_trade_steps = 1
            self.env._last_trade_step = current_step
        else:
            self.env._consecutive_trade_steps = 0

        # Update action counts
        if 0 <= action < len(self.env._action_counts):
            self.env._action_counts[action] += 1

    def update_portfolio_state(self, trade_pnl: float, unrealized_pnl: float) -> None:
        """Update portfolio-related state.

        Args:
            trade_pnl: Realized PnL from trade
            unrealized_pnl: Unrealized PnL from position
        """
        # Update portfolio value
        self.env.portfolio_value = (
            self.env.initial_portfolio_value + self.env.realized_pnl + unrealized_pnl
        )

        # Update history
        self.env.pnl_history.append(trade_pnl)
        self.env.position_abs_history.append(abs(self.env.position))
        self.env.portfolio_value_history.append(self.env.portfolio_value)
        self.env.position_history.append(self.env.position)

    def get_state_summary(self) -> Dict[str, Any]:
        """Get summary of current state for debugging/monitoring.

        Returns:
            Dictionary with state information
        """
        return {
            "position": self.env.position,
            "entry_price": self.env.entry_price,
            "total_pnl": self.env.total_pnl,
            "realized_pnl": self.env.realized_pnl,
            "trades_count": self.env.trades_count,
            "portfolio_value": self.env.portfolio_value,
            "current_step": self.env.current_step,
            "consecutive_trades": self.env._consecutive_trade_steps,
            "action_counts": self.env._action_counts.copy(),
        }

    def validate_state_consistency(self) -> bool:
        """Validate that state variables are consistent.

        Returns:
            True if state is consistent, False otherwise
        """
        try:
            # Check portfolio value consistency
            expected_portfolio = self.env.initial_portfolio_value + self.env.realized_pnl
            if self.env.position != 0 and self.env.entry_price > 0:
                # Add unrealized PnL if position exists
                current_price = self.env._resolve_price()
                if self.env.position > 0:
                    unrealized = (current_price - self.env.entry_price) * abs(self.env.position)
                else:
                    unrealized = (self.env.entry_price - current_price) * abs(self.env.position)
                expected_portfolio += unrealized

            portfolio_diff = abs(self.env.portfolio_value - expected_portfolio)
            if portfolio_diff > 1e-6:  # Allow small floating point differences
                self.logger.warning(
                    f"Portfolio value inconsistency: {self.env.portfolio_value} vs {expected_portfolio} (diff: {portfolio_diff})"
                )
                return False

            # Check position bounds
            if not (-2 <= self.env.position <= 2):  # Allow some margin for calculations
                self.logger.warning(f"Position out of bounds: {self.env.position}")
                return False

            return True

        except Exception as e:
            self.logger.error(f"State validation error: {e}")
            return False