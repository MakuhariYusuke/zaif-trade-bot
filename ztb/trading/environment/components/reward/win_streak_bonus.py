"""
Win Streak Bonus Calculator Component.

This component calculates bonuses for consecutive winning trades.
"""

class WinStreakBonusCalculator:
    """
    Calculates bonus for consecutive winning trades.

    This component incentivizes consistent winning performance
    by providing escalating bonuses for win streaks.
    """

    def calculate(
        self,
        reward_history: list[float],
        streak_window: int = 5,
        min_streak: int = 3,
        bonus_per_win: float = 0.01,
    ) -> float:
        """
        Calculate win streak bonus.

        Args:
            reward_history: History of rewards
            streak_window: Window size for streak calculation
            min_streak: Minimum consecutive wins to qualify for bonus
            bonus_per_win: Bonus amount per winning trade in streak

        Returns:
            Win streak bonus value (positive = bonus)
        """
        if len(reward_history) < streak_window:
            return 0.0

        recent_rewards = reward_history[-streak_window:]
        win_count = sum(1 for r in recent_rewards if r > 0)

        if win_count >= min_streak:
            # Bonus increases with number of wins in streak
            return win_count * bonus_per_win

        return 0.0
