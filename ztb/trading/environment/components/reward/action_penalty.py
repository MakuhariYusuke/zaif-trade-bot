"""
Action Penalty Calculator Component.

This component calculates penalties for different actions with fairness.
"""

from ztb.trading.constants import ACTION_BUY, ACTION_HOLD, ACTION_SELL
from ztb.trading.environment.constants import EPSILON


class ActionPenaltyCalculator:
    """
    Calculates action penalties with fairness between BUY and SELL.

    This component ensures BUY and SELL actions have identical penalties
    to maintain balance in the reward system.
    """

    def calculate(
        self,
        action: int,
        position: float,
        effective_max_position: float,
        current_price: float,
        atr: float,
        base_action_penalty: float = 0.015,
        buy_action_bonus: float = 0.0,
        sell_action_bonus: float = 0.0,
        hold_action_bonus: float = 0.0,
    ) -> float:
        """
        Calculate fair action penalty with bonus adjustments.

        Args:
            action: Action taken (HOLD=0, BUY=1, SELL=-1)
            position: Current position
            effective_max_position: Maximum position size
            current_price: Current asset price
            atr: Average true range
            base_action_penalty: Base penalty for trading actions
            buy_action_bonus: Bonus for BUY action (applied as negative penalty, resulting in reward)
            sell_action_bonus: Bonus for SELL action (applied as negative penalty, resulting in reward)
            hold_action_bonus: Bonus for HOLD action (applied as negative penalty, resulting in reward)

        Returns:
            Action penalty (always positive, bonuses are handled separately)
        """
        if action == ACTION_HOLD:
            position_size_factor = abs(position) / max(effective_max_position, EPSILON)
            volatility_factor = min(atr / (current_price * 0.01), 1.0)
            penalty = (
                0.01
                + 0.04
                * position_size_factor
                * volatility_factor  # hold_penalty_base  # hold_penalty_position_factor
            )
            penalty *= 1.0  # hold_penalty_multiplier
            # Return base penalty only - bonuses are handled separately in RewardCalculator
            return max(0.0, penalty)  # Ensure non-negative
        elif action == ACTION_BUY:
            # Return base penalty only - bonuses are handled separately in RewardCalculator
            return max(0.0, base_action_penalty)
        elif action == ACTION_SELL:
            # Return base penalty only - bonuses are handled separately in RewardCalculator
            return max(0.0, base_action_penalty)

        return 0.0
