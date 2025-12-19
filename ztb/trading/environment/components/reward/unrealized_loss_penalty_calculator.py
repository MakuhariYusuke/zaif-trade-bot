"""
Calculates a penalty for holding a position with unrealized losses over time.
"""

from ztb.trading.environment.utils.config import RewardSettings


class UnrealizedLossPenaltyCalculator:
    """
    Calculates a penalty that increases exponentially as a losing position is held.

    This encourages the agent to cut losses early rather than holding on in the
    hope of a price reversal.
    """

    def __init__(self, reward_settings: RewardSettings):
        """
        Initializes the UnrealizedLossPenaltyCalculator.

        Args:
            reward_settings: The reward settings configuration object.
        """
        self.enabled = reward_settings.unrealized_loss_penalty_enabled
        self.base = reward_settings.unrealized_loss_penalty_base
        self.max_steps = reward_settings.unrealized_loss_penalty_max_steps
        self._unrealized_loss_steps = 0

    def calculate(self, pnl: float, position: float) -> float:
        """
        Calculates the penalty based on the current PnL and position.

        Args:
            pnl: The current profit and loss.
            position: The current position size.

        Returns:
            The calculated penalty amount (always <= 0).
        """
        if not self.enabled:
            return 0.0

        # If we are in a losing position, increment the counter.
        # Otherwise, reset it.
        if pnl < 0 and position != 0:
            self._unrealized_loss_steps += 1
        else:
            self._unrealized_loss_steps = 0

        if self._unrealized_loss_steps == 0:
            return 0.0

        # The penalty grows exponentially with the number of steps in loss.
        # We cap the steps to prevent the penalty from becoming astronomically large.
        effective_steps = min(self._unrealized_loss_steps, self.max_steps)

        # Formula: -(base^steps - 1) * scale
        # The '-1' ensures the penalty is 0 at step 0 and grows from there.
        # The penalty is negative, representing a punishment.
        penalty = -((self.base**effective_steps) - 1)

        return penalty

    def reset(self):
        """Resets the internal state of the calculator."""
        self._unrealized_loss_steps = 0

    def get_unrealized_loss_steps(self) -> int:
        """Returns the current number of consecutive steps with unrealized loss."""
        return self._unrealized_loss_steps
