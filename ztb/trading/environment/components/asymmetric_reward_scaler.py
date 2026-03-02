"""
Asymmetric Reward Scaler Component.

This component applies asymmetric reward scaling based on position direction.
Follows Single Responsibility Principle by focusing only on asymmetric scaling.
"""

from ztb.trading.environment.utils.config import EnvironmentConfig
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

    def __init__(self, env_config: EnvironmentConfig):
        """
        Initialize AsymmetricRewardScaler.

        Args:
            env_config: Environment configuration object
        """
        self._config = env_config
        self.logger = get_logger("ztb.trading.environment.asymmetric_reward_scaler")
        self._penalty_count = 0
        self._load_settings()

    def _load_settings(self):
        """Load settings from the environment configuration."""
        reward_settings = getattr(self._config, "reward_settings", None)

        if reward_settings is None:
            # Default settings when reward_settings is None
            self.long_pos_reward_multiplier = 1.0
            self.short_pos_reward_multiplier = 1.0
            self.long_pos_penalty_multiplier = 1.0
            self.short_pos_penalty_multiplier = 1.0
        else:
            settings = None
            if isinstance(reward_settings, dict):
                settings = reward_settings.get("asymmetric_reward_scaling")
            else:
                settings = getattr(reward_settings, "asymmetric_reward_scaling", None)

            if not isinstance(settings, dict):
                settings = {}

            self.long_pos_reward_multiplier = settings.get(
                "long_position_reward_multiplier", 1.0
            )
            self.short_pos_reward_multiplier = settings.get(
                "short_position_reward_multiplier", 1.0
            )
            self.long_pos_penalty_multiplier = settings.get(
                "long_position_penalty_multiplier", 1.0
            )
            self.short_pos_penalty_multiplier = settings.get(
                "short_position_penalty_multiplier", 1.0
            )

        # Thresholds are not part of asymmetric_reward_scaling dict, let's assume they are hardcoded for now
        # or need to be added to the config structure. For now, keep them as they were.
        # This can be a point of future improvement.
        self.long_pos_threshold = 0.01
        self.short_pos_threshold = -0.01

    def _get_position_direction(self, position: float) -> str:
        """
        Determine position direction for asymmetric reward scaling.

        Args:
            position: Current position size

        Returns:
            'long', 'short', or 'neutral'
        """
        if position > self.long_pos_threshold:  # Long position threshold
            return "long"
        elif position < self.short_pos_threshold:  # Short position threshold
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
                reward *= self.long_pos_reward_multiplier
                # self.logger.debug(
                #     f"Applied long position reward boost: {self.long_pos_reward_multiplier}x"
                # )
            elif position_direction == "short":
                reward *= self.short_pos_reward_multiplier
                # self.logger.debug(
                #     f"Applied short position reward reduction: {self.short_pos_reward_multiplier}x"
                # )
        else:  # Loss trade
            self._penalty_count += 1
            if position_direction == "long":
                reward *= self.long_pos_penalty_multiplier
                # self.logger.debug(
                #     f"Applied long position penalty reduction: {self.long_pos_penalty_multiplier}x"
                # ) if self._penalty_count % 20 == 0 else None
            elif position_direction == "short":
                reward *= self.short_pos_penalty_multiplier
                # self.logger.debug(
                #     f"Applied short position penalty boost: {self.short_pos_penalty_multiplier}x"
                # ) if self._penalty_count % 20 == 0 else None

        return reward
