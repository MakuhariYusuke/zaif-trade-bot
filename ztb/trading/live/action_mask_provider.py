"""
Action Mask Provider for Live Trading

Provides action masking functionality for MaskablePPO models in live trading
without requiring a full Gymnasium environment instance.

This module addresses Bug #27 by allowing proper action masking in production.
"""

from dataclasses import dataclass
from typing import Any, Dict

import numpy as np
from numpy.typing import NDArray

# Action indices for mask array
ACTION_HOLD = 0
ACTION_BUY = 1
ACTION_SELL = -1


@dataclass
class ActionMaskConfig:
    """Configuration for action masking in live trading."""

    min_holding_period: int = 5
    """Minimum time steps to hold a position before allowing closure."""

    enable_forced_close: bool = True
    """Enable forced position close on take-profit/stop-loss."""

    max_position_age: int = 1000
    """Maximum age of position before forcing closure."""


class ActionMaskProvider:
    """
    Lightweight action mask provider for MaskablePPO in live trading.

    This class provides the same action masking logic as the Gymnasium environment
    but without requiring a full environment instance.

    Bug #27 Fix: Enables proper MaskablePPO usage in live trading with
    action masking safety features (min_holding_period, forced closes, etc.)
    """

    def __init__(self, config: ActionMaskConfig):
        """
        Initialize the action mask provider.

        Args:
            config: Configuration for action masking behavior
        """
        self.config = config

        # State tracking (synchronized with PositionManager)
        self.current_position: float = 0.0
        self.position_entry_step: int = 0
        self.current_step: int = 0

        # Forced close state
        self.forced_close_reason: str | None = None

    def update_state(
        self,
        current_position: float,
        position_entry_step: int,
        current_step: int,
        forced_close_reason: str | None = None,
    ) -> None:
        """
        Update internal state for mask calculation.

        Args:
            current_position: Current position size (positive=long, negative=short, 0=flat)
            position_entry_step: Step when current position was opened
            current_step: Current time step
            forced_close_reason: Reason for forced close (take-profit/stop-loss) or None
        """
        self.current_position = current_position
        self.position_entry_step = position_entry_step
        self.current_step = current_step
        self.forced_close_reason = forced_close_reason

    def get_action_mask(self) -> NDArray[np.bool_]:
        """
        Get valid action mask based on current state.

        Returns:
            NDArray[np.bool_]: Boolean array [hold_valid, buy_valid, sell_valid]
                              Indices correspond to ACTION_HOLD=0, ACTION_BUY=1, ACTION_SELL=2

        Mask Logic:
        - Forced close: Only allow closing action
        - Min holding period: Block closing if position too young
        - Max position age: Force close if position too old
        - Default: Allow all actions compatible with current position
        """
        # Default: all actions allowed [HOLD, BUY, SELL]
        mask = np.array([True, True, True], dtype=bool)

        # Check forced close condition
        if self._should_force_close():
            return self._get_forced_close_mask()

        # Check min holding period
        if not self._min_holding_period_satisfied():
            mask = self._block_closing_actions(mask)

        # Block invalid position transitions
        mask = self._apply_position_constraints(mask)

        return mask

    def get_action_masks(self) -> NDArray[np.bool_]:
        """Alias for get_action_mask() for compatibility with predict_with_masks."""
        return self.get_action_mask()

    def _should_force_close(self) -> bool:
        """Check if position should be forcibly closed."""
        if not self.config.enable_forced_close:
            return False

        # Forced close due to take-profit/stop-loss
        if self.forced_close_reason is not None:
            return True

        # Forced close due to max position age
        if self.current_position != 0:
            position_age = self.current_step - self.position_entry_step
            if position_age >= self.config.max_position_age:
                return True

        return False

    def _get_forced_close_mask(self) -> NDArray[np.bool_]:
        """Get mask that only allows position closing.

        Returns:
            NDArray[np.bool_]: [hold_valid, buy_valid, sell_valid]
        """
        if self.current_position > 0:
            # Long position: only SELL allowed
            # [HOLD=False, BUY=False, SELL=True]
            return np.array([False, False, True], dtype=bool)
        elif self.current_position < 0:
            # Short position: only BUY allowed
            # [HOLD=False, BUY=True, SELL=False]
            return np.array([False, True, False], dtype=bool)
        else:
            # Flat: shouldn't reach here, but allow all actions
            return np.array([True, True, True], dtype=bool)

    def _min_holding_period_satisfied(self) -> bool:
        """Check if minimum holding period is satisfied."""
        if self.current_position == 0:
            return True  # No position, no restriction

        position_age = self.current_step - self.position_entry_step
        return position_age >= self.config.min_holding_period

    def _block_closing_actions(self, mask: NDArray[np.bool_]) -> NDArray[np.bool_]:
        """Block actions that would close the current position.

        Args:
            mask: Current mask [HOLD, BUY, SELL]

        Returns:
            Updated mask with closing actions blocked
        """
        if self.current_position > 0:
            # Long position: block SELL (index 2)
            mask[ACTION_SELL] = False
        elif self.current_position < 0:
            # Short position: block BUY (index 1)
            mask[ACTION_BUY] = False

        return mask

    def _apply_position_constraints(self, mask: NDArray[np.bool_]) -> NDArray[np.bool_]:
        """
        Apply constraints to prevent invalid position transitions.

        Args:
            mask: Current mask [HOLD, BUY, SELL]

        Rules:
        - Can't BUY when already long (position > 0)
        - Can't SELL when already short (position < 0)
        - Always allow HOLD

        Returns:
            Updated mask with position constraints applied
        """
        if self.current_position > 0:
            # Long position: block BUY (index 1), allow SELL (close) and HOLD
            mask[ACTION_BUY] = False
        elif self.current_position < 0:
            # Short position: block SELL (index 2), allow BUY (close) and HOLD
            mask[ACTION_SELL] = False

        # HOLD (index 0) is always allowed (unless forced close)

        return mask

    def get_mask_info(self) -> Dict[str, Any]:
        """
        Get detailed information about current masking state.

        Returns:
            Dict with mask state, position info, and reasoning
        """
        mask = self.get_action_mask()

        position_age = 0
        if self.current_position != 0:
            position_age = self.current_step - self.position_entry_step

        return {
            "mask": mask.tolist(),
            "mask_human": {
                "HOLD": mask[ACTION_HOLD],
                "BUY": mask[ACTION_BUY],
                "SELL": mask[ACTION_SELL],
            },
            "position": self.current_position,
            "position_age": position_age,
            "min_holding_satisfied": self._min_holding_period_satisfied(),
            "forced_close": self._should_force_close(),
            "forced_close_reason": self.forced_close_reason,
            "current_step": self.current_step,
        }


def create_mask_provider_from_env_config(
    env_config: Dict[str, Any],
) -> ActionMaskProvider:
    """
    Create ActionMaskProvider from environment configuration.

    Args:
        env_config: Dictionary containing environment configuration

    Returns:
        Configured ActionMaskProvider instance
    """
    mask_config = ActionMaskConfig(
        min_holding_period=env_config.get("min_holding_period", 5),
        enable_forced_close=env_config.get("enable_forced_close", True),
        max_position_age=env_config.get("max_position_age", 1000),
    )

    return ActionMaskProvider(mask_config)
