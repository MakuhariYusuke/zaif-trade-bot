"""
Stagnation Penalty Calculator Component.

This component calculates penalties for portfolio stagnation.
"""

class StagnationPenaltyCalculator:
    """
    Calculates penalty for portfolio stagnation (lack of progress).

    This component prevents the agent from holding positions without
    making meaningful progress in portfolio value.
    """

    def calculate(
        self,
        portfolio_value_history: list[float],
        stagnation_window: int = 20,
        stagnation_threshold: float = 0.005,
    ) -> float:
        """
        Calculate stagnation penalty.

        Args:
            portfolio_value_history: History of portfolio values
            stagnation_window: Window size for stagnation calculation
            stagnation_threshold: Minimum change required to avoid penalty

        Returns:
            Stagnation penalty value (positive = penalty)
        """
        if len(portfolio_value_history) < stagnation_window:
            return 0.0

        # Check if portfolio has made meaningful progress
        recent_values = portfolio_value_history[-stagnation_window:]
        initial_value = recent_values[0]
        final_value = recent_values[-1]

        if initial_value > 0:
            change_ratio = abs(final_value - initial_value) / initial_value

            if change_ratio < stagnation_threshold:
                # Penalty increases with stagnation duration
                stagnation_ratio = (
                    stagnation_threshold - change_ratio
                ) / stagnation_threshold
                return min(stagnation_ratio * 0.1, 0.05)  # Cap at 5% penalty

        return 0.0
