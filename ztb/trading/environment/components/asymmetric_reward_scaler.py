"""
Asymmetric Reward Scaler Component.

This component applies asymmetric reward scaling based on position direction.
Follows Single Responsibility Principle by focusing only on asymmetric scaling.
"""

from ztb.utils.logging_utils import get_logger

from .interfaces import IAsymmetricRewardScaler


class AsymmetricRewardScaler(IAsymmetricRewardScaler):
    """
    Applies asymmetric reward scaling based on position direction.

    This class encapsulates asymmetric reward scaling logic including:
    - Position direction detection
    - Long/short position multipliers
    - Profit/loss based scaling
    """

    def __init__(
        self,
        long_position_reward_multiplier: float = 1.3,
        short_position_reward_multiplier: float = 0.7,
        long_position_penalty_multiplier: float = 0.9,
        short_position_penalty_multiplier: float = 0.95,
    ):
        """
        Initialize AsymmetricRewardScaler.

        Args:
            long_position_reward_multiplier: Multiplier for long position rewards
            short_position_reward_multiplier: Multiplier for short position rewards
            long_position_penalty_multiplier: Multiplier for long position penalties
            short_position_penalty_multiplier: Multiplier for short position penalties
        """
        self.long_position_reward_multiplier = long_position_reward_multiplier
        self.short_position_reward_multiplier = short_position_reward_multiplier
        self.long_position_penalty_multiplier = long_position_penalty_multiplier
        self.short_position_penalty_multiplier = short_position_penalty_multiplier

        self.logger = get_logger("ztb.trading.environment.asymmetric_reward_scaler")

    def _get_position_direction(self, position: float) -> str:
        """
        Determine position direction for asymmetric reward scaling.

        Args:
            position: Current position size

        Returns:
            'long', 'short', or 'neutral'
        """
        if position > 0.01:  # Long position threshold
            return "long"
        elif position < -0.01:  # Short position threshold
            return "short"
        else:
            return "neutral"

    def scale_reward(self, reward: float, position: float, pnl: float) -> float:
        """
        Apply asymmetric reward scaling based on position direction.

        This addresses position imbalance issues by:
        - Boosting rewards for long positions to encourage more balanced trading
        - Reducing rewards for short positions to prevent over-reliance on short trades
        - Applying different penalty multipliers for losses in different directions

        Args:
            reward: Base reward value
            position: Current position
            pnl: Profit/Loss from action

        Returns:
            Scaled reward value
        """
        position_direction = self._get_position_direction(position)

        if position_direction == "neutral":
            return reward  # No scaling for neutral positions

        # Apply asymmetric scaling based on profit/loss and position direction
        if pnl > 0:  # Profitable trade
            if position_direction == "long":
                reward *= self.long_position_reward_multiplier
                self.logger.debug(
                    f"Applied long position reward boost: {self.long_position_reward_multiplier}x"
                )
            elif position_direction == "short":
                reward *= self.short_position_reward_multiplier
                self.logger.debug(
                    f"Applied short position reward reduction: {self.short_position_reward_multiplier}x"
                )
        else:  # Loss trade
            if position_direction == "long":
                reward *= self.long_position_penalty_multiplier
                self.logger.debug(
                    f"Applied long position penalty reduction: {self.long_position_penalty_multiplier}x"
                )
            elif position_direction == "short":
                reward *= self.short_position_penalty_multiplier
                self.logger.debug(
                    f"Applied short position penalty boost: {self.short_position_penalty_multiplier}x"
                )

        return reward