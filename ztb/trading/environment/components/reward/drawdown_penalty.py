"""
Drawdown Penalty Calculator Component.

This component calculates penalties for excessive drawdowns.
"""

from typing import List


class DrawdownPenaltyCalculator:
    """
    Calculates penalty for drawdown exceeding safe limits.

    This component prevents excessive losses by penalizing
    prolonged drawdown periods.
    """

    def calculate(
        self, reward_history: List[float], drawdown_window: int = 20
    ) -> float:
        """
        Calculate drawdown penalty.

        Args:
            reward_history: History of rewards
            drawdown_window: Window size for drawdown calculation

        Returns:
            Drawdown penalty value
        """
        if len(reward_history) < drawdown_window:
            return 0.0

        # Simple drawdown calculation (cumulative negative rewards)
        recent_rewards = reward_history[-drawdown_window:]
        cumulative_drawdown = sum(min(0, r) for r in recent_rewards)

        # Penalty if drawdown exceeds 50% of window size
        drawdown_threshold = -drawdown_window * 0.5
        if cumulative_drawdown < drawdown_threshold:
            penalty_scale = 0.1
            return penalty_scale * abs(cumulative_drawdown - drawdown_threshold)

        return 0.0
