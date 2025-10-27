"""Validation component for HeavyTradingEnv."""

from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple

import numpy as np
from numpy.typing import NDArray

from ztb.utils.logging_utils import get_logger
from ztb.utils.errors import ValidationError

if TYPE_CHECKING:
    from ztb.trading.environment.heavy_env.core import HeavyTradingEnv


class ValidationManager:
    """Handles validation logic for trading environment."""

    def __init__(self, env: "HeavyTradingEnv"):
        """Initialize validation manager with reference to environment."""
        self.env = env
        self.logger = get_logger(__name__)

    def validate_action(self, action: int) -> int:
        """Validate and convert continuous action to discrete if needed.

        Args:
            action: Action to validate (int or continuous)

        Returns:
            Validated discrete action

        Raises:
            ValidationError: If action is invalid
        """
        # Handle continuous actions
        if isinstance(action, (float, np.ndarray)):
            continuous_action = float(action) if isinstance(action, np.ndarray) else action
            if continuous_action > self.env.action_threshold:
                return 1  # BUY
            elif continuous_action < self.env.negative_action_threshold:
                return 2  # SELL
            else:
                return 0  # HOLD

        # Validate discrete action
        if not isinstance(action, int):
            raise ValidationError(f"Action must be int or float, got {type(action)}")

        if action not in [0, 1, 2]:
            raise ValidationError(f"Invalid discrete action: {action}, must be 0, 1, or 2")

        return action

    def validate_observation_request(self, step: int) -> None:
        """Validate observation request parameters.

        Args:
            step: Step to get observation for

        Raises:
            ValidationError: If request is invalid
        """
        if step < 0:
            raise ValidationError(f"Step cannot be negative: {step}")

        if step >= self.env.n_steps:
            raise ValidationError(f"Step {step} exceeds episode length {self.env.n_steps}")

    def validate_position_size(self, position: float) -> None:
        """Validate position size is within bounds.

        Args:
            position: Position to validate

        Raises:
            ValidationError: If position is invalid
        """
        max_pos = self.env.config.max_position_size
        if abs(position) > max_pos * 1.1:  # Allow 10% margin for calculations
            raise ValidationError(f"Position {position} exceeds maximum {max_pos}")

    def validate_price_data(self, price: float) -> None:
        """Validate price data is reasonable.

        Args:
            price: Price to validate

        Raises:
            ValidationError: If price is invalid
        """
        if price <= 0:
            raise ValidationError(f"Price must be positive: {price}")

        # Check for extreme price changes (more than 50% in reasonable range)
        if hasattr(self, '_last_validated_price') and self._last_validated_price:
            change_ratio = abs(price - self._last_validated_price) / self._last_validated_price
            if change_ratio > 0.5:  # 50% change
                self.logger.warning(f"Large price change detected: {change_ratio:.1%} from {self._last_validated_price} to {price}")

        self._last_validated_price = price

    def validate_reward_calculation(self, reward: float) -> float:
        """Validate and clip reward if necessary.

        Args:
            reward: Raw reward value

        Returns:
            Validated and clipped reward
        """
        if not np.isfinite(reward):
            self.logger.warning(f"Non-finite reward detected: {reward}, setting to 0")
            return 0.0

        # Apply reward clipping if configured
        reward_clip = self.env.reward_settings.get("reward_clip_value")
        if reward_clip and abs(reward) > reward_clip:
            clipped_reward = np.clip(reward, -reward_clip, reward_clip)
            self.logger.debug(f"Reward clipped from {reward} to {clipped_reward}")
            return clipped_reward

        return reward

    def validate_environment_state(self) -> List[str]:
        """Validate overall environment state and return any issues.

        Returns:
            List of validation issues (empty if all valid)
        """
        issues = []

        # Check position bounds
        try:
            self.validate_position_size(self.env.position)
        except ValidationError as e:
            issues.append(f"Position validation: {e}")

        # Check portfolio value
        if self.env.portfolio_value < 0:
            issues.append(f"Negative portfolio value: {self.env.portfolio_value}")

        # Check data availability
        if self.env.current_step >= self.env.n_steps:
            issues.append(f"Step {self.env.current_step} exceeds episode length {self.env.n_steps}")

        # Check required arrays
        if self.env._feature_matrix is None:
            issues.append("Feature matrix not initialized")

        if self.env._price_array is None and self.env._close_array is None:
            issues.append("Price data not available")

        return issues

    def get_validation_summary(self) -> Dict[str, Any]:
        """Get summary of validation state.

        Returns:
            Dictionary with validation information
        """
        issues = self.validate_environment_state()

        return {
            "is_valid": len(issues) == 0,
            "issues": issues,
            "position_valid": len([i for i in issues if "Position" in i]) == 0,
            "portfolio_valid": "Negative portfolio" not in str(issues),
            "data_valid": "not initialized" not in str(issues),
        }