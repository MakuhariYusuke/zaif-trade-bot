"""
Position Penalty Calculator Component.

This component calculates penalties for excessive position usage.
"""

import math


class PositionPenaltyCalculator:
    """
    Calculates penalty for excessive position usage.

    This component ensures positions don't exceed safe limits by applying
    exponential penalties for overuse.
    """

    def __init__(
        self, soft_cap: float = 0.8, scale: float = 0.5, exponent: float = 2.0
    ):
        """
        Initialize PositionPenaltyCalculator.

        Args:
            soft_cap: Utilization threshold before penalty applies
            scale: Penalty scaling factor
            exponent: Penalty exponent for overuse
        """
        self.soft_cap = soft_cap
        self.scale = scale
        self.exponent = exponent

    def calculate(self, position: float, effective_max_position: float) -> float:
        """
        Calculate position penalty.

        Args:
            position: Current position (normalized -1 to 1)
            effective_max_position: Maximum allowed position size

        Returns:
            Penalty value (0 if within limits, positive if exceeded)
        """
        # Position is normalized [-1, 1], so utilisation is simply abs(position)
        # effective_max_position is not needed here as position is already scaled
        position_utilisation = abs(position)

        # If position is very small (close to 0), no penalty
        if position_utilisation < 0.01:  # Near-zero position threshold
            return 0.0

        if position_utilisation > self.soft_cap:
            overuse = position_utilisation - self.soft_cap
            # Prevent math range error by clamping exponent * overuse
            exp_arg = min(self.exponent * overuse, 5.0)
            return self.scale * (math.exp(exp_arg) - 1.0)

        return 0.0
