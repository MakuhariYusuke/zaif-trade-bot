"""
Action Executor - Handles action conversion and execution logic.

This module separates action-related logic from the main environment class,
including conversion between continuous and discrete actions, and action validation.
"""

from typing import Any, Optional, Union

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
        action: Union[int, np.ndarray],
    ) -> tuple[int, Optional[float]]:
        """
        Convert action to discrete format and validate.

        Args:
            action: Raw action (continuous or discrete)

        Returns:
            Tuple of (discrete_action, continuous_value)

        Raises:
            ValueError: If action is invalid
            TypeError: If action type is unsupported
        """
        try:
            if isinstance(action, np.ndarray):
                if action.size != 1:
                    raise ValueError(f"Continuous action must be 1D array with single value, got shape {action.shape}")

                continuous_value = float(action[0])
                if not np.isfinite(continuous_value):
                    raise ValueError(f"Continuous action value must be finite, got {continuous_value}")

                discrete_action = continuous_to_discrete_action(
                    continuous_value,
                    threshold=self.action_threshold,
                    negative_threshold=self.negative_action_threshold,
                )
                return discrete_action, continuous_value
            elif isinstance(action, (int, np.integer)):
                action_int = int(action)
                if action_int in (ACTION_HOLD, ACTION_BUY, ACTION_SELL):
                    return action_int, None
                elif action_int in (0, 1, 2):
                    action_mapping = {
                        0: ACTION_HOLD,
                        1: ACTION_BUY,
                        2: ACTION_SELL,
                    }
                    return action_mapping[action_int], None
                else:
                    raise ValueError(f"Invalid discrete action: {action_int}. Must be 0, 1, 2, or ACTION_* constants")
            else:
                raise TypeError(f"Unsupported action type: {type(action)}. Must be int or np.ndarray")
        except Exception as e:
            logger.error(f"Failed to convert and validate action {action}: {e}")
            raise

    def get_action_info(self, action: Union[int, np.ndarray]) -> dict[str, Any]:
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
            "action_type": "continuous" if isinstance(action, np.ndarray) else "discrete",
        }

        if continuous_value is not None:
            info["continuous_value"] = continuous_value
            info["threshold_used"] = self.action_threshold
            info["negative_threshold_used"] = self.negative_action_threshold

        return info