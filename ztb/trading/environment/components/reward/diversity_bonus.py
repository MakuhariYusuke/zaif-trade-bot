"""
Diversity Bonus Calculator Component.

This component calculates bonuses for action diversity.
"""

from typing import List


class DiversityBonusCalculator:
    """
    Calculates bonus for maintaining action diversity.

    This component encourages varied action selection to prevent
    repetitive behavior patterns.
    """

    def calculate(self, recent_actions: List[int]) -> float:
        """
        Calculate diversity bonus.

        Args:
            recent_actions: List of recent actions

        Returns:
            Diversity bonus value
        """
        if len(recent_actions) < 3:
            return 0.1  # Small bonus for early exploration

        unique_recent = len(set(recent_actions[-10:]))  # Last 10 actions
        diversity_score = unique_recent / 3.0  # Normalize by action types

        # Bonus for maintaining diversity
        base_bonus = 0.05
        diversity_multiplier = diversity_score**2  # Quadratic scaling

        return base_bonus * diversity_multiplier
