"""
Growth Bonus Calculator Component.

This component calculates bonuses for portfolio growth.
"""

from typing import List


class GrowthBonusCalculator:
    """
    Calculates bonus for portfolio growth above threshold.

    This component incentivizes consistent portfolio growth
    by providing bonuses for meaningful increases in value.
    """

    def calculate(
        self,
        portfolio_value_history: List[float],
        growth_window: int = 30,
        growth_threshold: float = 0.005,
    ) -> float:
        """
        Calculate growth bonus.

        Args:
            portfolio_value_history: History of portfolio values
            growth_window: Window size for growth calculation
            growth_threshold: Minimum growth rate to qualify for bonus

        Returns:
            Growth bonus value (positive = bonus)
        """
        if len(portfolio_value_history) < growth_window:
            return 0.0

        recent_values = portfolio_value_history[-growth_window:]
        initial_value = recent_values[0]
        final_value = recent_values[-1]

        if initial_value > 0:
            growth_rate = (final_value - initial_value) / initial_value

            if growth_rate > growth_threshold:
                # Bonus scales with growth rate, capped at reasonable level
                bonus = min(growth_rate * 0.5, 0.1)  # Cap at 10% bonus
                return bonus

        return 0.0
