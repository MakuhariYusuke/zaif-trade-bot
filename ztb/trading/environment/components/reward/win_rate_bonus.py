"""
Win Rate Bonus Calculator Component.

This component calculates bonuses based on win rate performance.
"""
from ztb.trading.constants import ACTION_BUY, ACTION_SELL

class WinRateBonusCalculator:
    """
    Calculates bonus based on win rate and action type.

    This component provides incentives for profitable actions
    and penalties for unprofitable ones.
    """

    def calculate(self, discrete_action: int, pnl: float) -> float:
        """
        Calculate win rate bonus.

        Args:
            discrete_action: Action taken (0=HOLD, 1=BUY, 2=SELL)
            pnl: Profit and loss

        Returns:
            Win rate bonus value
        """
        # Base win rate bonus
        win_rate_bonus_base = 0.1

        if pnl > 0:
            # Bonus for winning trades
            if discrete_action == ACTION_BUY:
                return win_rate_bonus_base * 1.2
            elif discrete_action == ACTION_SELL:
                return win_rate_bonus_base * 1.2
            else:  # HOLD
                return win_rate_bonus_base * 0.5
        else:
            # Penalty for losing trades
            if discrete_action == ACTION_BUY:
                return -win_rate_bonus_base * 0.8
            elif discrete_action == ACTION_SELL:
                return -win_rate_bonus_base * 0.8
            else:  # HOLD
                return -win_rate_bonus_base * 0.3
