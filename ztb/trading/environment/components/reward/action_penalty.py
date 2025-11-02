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
        Calculate fair action penalty.

        Args:
            action: Action taken (HOLD=0, BUY=1, SELL=-1)
            position: Current position
            effective_max_position: Maximum position size
            current_price: Current asset price
            atr: Average true range
            base_action_penalty: Base penalty for trading actions
            buy_action_bonus: Bonus for BUY action
            sell_action_bonus: Bonus for SELL action
            hold_action_bonus: Bonus for HOLD action

        Returns:
            Action penalty (positive = penalty, negative = bonus)
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
            return penalty + hold_action_bonus
        elif action == ACTION_BUY:
            return base_action_penalty + buy_action_bonus
        elif action == ACTION_SELL:
            return base_action_penalty + sell_action_bonus

        return 0.0
