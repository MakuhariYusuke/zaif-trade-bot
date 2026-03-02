"""
Action Executor - Handles action conversion and execution logic.

This module separates action-related logic from the main environment class,
including conversion between continuous and discrete actions, and action validation.
"""

from typing import Any, Optional

import numpy as np

from ztb.trading.constants import ACTION_BUY, ACTION_HOLD, ACTION_SELL
from ztb.trading.environment.constants import continuous_to_discrete_action
from ztb.utils.logging_utils import get_logger

logger = get_logger(__name__)

class ActionExecutor:
    """
    Handles action conversion and execution for trading environment.

    This class handles:
    - Continuous to discrete action conversion
    - Action validation and mapping
    - Action threshold configuration
    """

    def __init__(
        self,
        action_threshold: float,
        negative_action_threshold: float,
    ):
        """
        Initialize ActionExecutor.

        Args:
            action_threshold: Threshold for BUY action
            negative_action_threshold: Threshold for SELL action
        """
        self.action_threshold = action_threshold
        self.negative_action_threshold = negative_action_threshold

    def convert_and_validate_action(
        self,
        action: int | np.ndarray,
        dynamic_threshold: float | None = None,
        dynamic_negative_threshold: float | None = None,
    ) -> tuple[int, float | None]:
        """
        Convert action to discrete format and validate.

        Args:
            action: Raw action (continuous or discrete)
            dynamic_threshold: Optional dynamic threshold for BUY (overrides self.action_threshold)
            dynamic_negative_threshold: Optional dynamic threshold for SELL (overrides self.negative_action_threshold)

        Returns:
            tuple of (discrete_action, continuous_value)

        Raises:
            ValueError: If action is invalid
            TypeError: If action type is unsupported
        """
        # Use dynamic thresholds if provided, otherwise use defaults
        threshold = (
            dynamic_threshold
            if dynamic_threshold is not None
            else self.action_threshold
        )
        negative_threshold = (
            dynamic_negative_threshold
            if dynamic_negative_threshold is not None
            else self.negative_action_threshold
        )

        try:
            if isinstance(action, np.ndarray):
                if action.size != 1:
                    raise ValueError(
                        f"Continuous action must be 1D array with single value, got shape {action.shape}"
                    )

                continuous_value = float(action[0])
                if not np.isfinite(continuous_value):
                    raise ValueError(
                        f"Continuous action value must be finite, got {continuous_value}"
                    )

                discrete_action = continuous_to_discrete_action(
                    continuous_value,
                    threshold=threshold,
                    negative_threshold=negative_threshold,
                )
                return discrete_action, continuous_value
            elif isinstance(action, (int, np.integer)):
                action_int = int(action)
                if action_int in (ACTION_HOLD, ACTION_BUY, ACTION_SELL):
                    return action_int, None
                elif action_int in (-1, 0, 1, 2):
                    action_mapping = {
                        -1: ACTION_SELL,
                        0: ACTION_HOLD,
                        1: ACTION_BUY,
                        2: ACTION_SELL,  # Alias for compatibility
                    }
                    return action_mapping[action_int], None
                else:
                    raise ValueError(
                        f"Invalid discrete action: {action_int}. Must be -1, 0, 1, or ACTION_* constants"
                    )
            else:
                raise TypeError(
                    f"Unsupported action type: {type(action)}. Must be int or np.ndarray"
                )
        except Exception as e:
            logger.error(f"Failed to convert and validate action {action}: {e}")
            raise

    def get_action_info(self, action: int | np.ndarray) -> dict[str, Any]:
        """
        Get action information for logging/debugging.

        Args:
            action: Raw action

        Returns:
            Dictionary with action details
        """
        discrete_action, continuous_value = self.convert_and_validate_action(action)

        info = {
            "raw_action": action,
            "discrete_action": discrete_action,
            "action_type": "continuous"
            if isinstance(action, np.ndarray)
            else "discrete",
        }

        if continuous_value is not None:
            info["continuous_value"] = continuous_value
            info["threshold_used"] = self.action_threshold
            info["negative_threshold_used"] = self.negative_action_threshold

        return info
