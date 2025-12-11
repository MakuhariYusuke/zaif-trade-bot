from typing import List


class RewardUtils:
    """Utility functions for reward calculations to reduce duplication."""

    @staticmethod
    def calculate_pnl_reward(pnl: float, scaling: float = 1.0) -> float:
        """Calculates basic PnL reward."""
        return pnl * scaling

    @staticmethod
    def calculate_balance_penalty(
        action_counts: List[int],
        target_ratios: List[float],
        tolerance: float,
        penalty_coeff: float,
    ) -> float:
        """Calculates penalty for deviation from target action ratios."""
        total_actions = sum(action_counts)
        if total_actions < 10:
            return 0.0

        balance_penalty = 0.0
        action_ratios = [count / total_actions for count in action_counts]

        for i, ratio in enumerate(action_ratios):
            # Ensure we don't go out of bounds if target_ratios is shorter
            if i >= len(target_ratios):
                break

            deviation = abs(ratio - target_ratios[i])
            if deviation > tolerance:
                excess_deviation = deviation - tolerance
                balance_penalty += penalty_coeff * excess_deviation

        return balance_penalty

    @staticmethod
    def calculate_trading_bonus(
        action: int, bonus_amount: float, action_buy: int = 1, action_sell: int = 2
    ) -> float:
        """Calculates bonus for trading actions (BUY/SELL)."""
        if action in [action_buy, action_sell]:
            return bonus_amount
        return 0.0

    @staticmethod
    def calculate_position_penalty(
        position: float,
        effective_max_position: float,
        threshold: float = 0.5,
        penalty_coeff: float = 0.2,
    ) -> float:
        """Calculates penalty for excessive position size."""
        if effective_max_position <= 0:
            return 0.0

        position_utilization = abs(position) / effective_max_position
        if position_utilization > threshold:
            return (position_utilization - threshold) * penalty_coeff
        return 0.0

    @staticmethod
    def calculate_position_size_bonus(
        position: float,
        effective_max_position: float,
        bonus_rate: float = 0.05,
        min_util: float = 0.1,
        max_util: float = 0.8,
    ) -> float:
        """Calculates bonus for maintaining a healthy position size."""
        if effective_max_position <= 0:
            return 0.0

        position_utilization = abs(position) / max(effective_max_position, 0.01)
        if min_util <= position_utilization <= max_util:
            return bonus_rate * position_utilization
        return 0.0

    @staticmethod
    def calculate_activity_bonus(
        recent_actions: List[int],
        bonus_rate: float = 0.02,
        window_size: int = 5,
        min_trades: int = 2,
        action_hold: int = 0,
    ) -> float:
        """Calculates bonus for recent trading activity."""
        if not recent_actions:
            return 0.0

        recent_window = recent_actions[-window_size:]
        recent_trades = sum(1 for a in recent_window if a != action_hold)

        if recent_trades >= min_trades:
            return bonus_rate * (recent_trades / float(window_size))
        return 0.0
