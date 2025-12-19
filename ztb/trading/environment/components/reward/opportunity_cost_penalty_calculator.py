"""
Opportunity Cost Penalty Calculator
"""
from ztb.trading.environment.utils.config import RewardSettings
from ztb.trading.constants import ACTION_BUY, ACTION_SELL, ACTION_HOLD

class OpportunityCostPenaltyCalculator:
    """
    Calculates a penalty for not trading when the position is flat.
    """
    def __init__(self, reward_settings: RewardSettings):
        self.enabled = getattr(reward_settings, "opportunity_cost_penalty_enabled", False)
        self.base_penalty = getattr(reward_settings, "opportunity_cost_penalty_base", 0.01)
        self.increase_rate = getattr(reward_settings, "opportunity_cost_penalty_increase_rate", 0.005)
        self.max_steps = getattr(reward_settings, "opportunity_cost_max_steps", 50)
        self.consecutive_flat_holds = 0

    def reset(self):
        """Resets the consecutive hold counter."""
        self.consecutive_flat_holds = 0

    def calculate(self, action: int, position: float) -> float:
        """
        Calculates the opportunity cost penalty.

        Args:
            action: The action taken by the agent.
            position: The current position size.

        Returns:
            The calculated penalty, which is always zero or negative.
        """
        if not self.enabled:
            return 0.0

        is_flat = abs(position) < 1e-6

        if is_flat and action == ACTION_HOLD:
            self.consecutive_flat_holds += 1
        elif action in [ACTION_BUY, ACTION_SELL]:
            self.consecutive_flat_holds = 0

        if self.consecutive_flat_holds > 0:
            penalty = self.base_penalty + (self.increase_rate * (self.consecutive_flat_holds - 1))
            return -min(penalty, self.base_penalty + self.increase_rate * (self.max_steps -1))

        return 0.0

    def get_consecutive_flat_holds(self) -> int:
        """Returns the current count of consecutive flat holds."""
        return self.consecutive_flat_holds
