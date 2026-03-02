

class RewardUtils:
    """Utility functions for reward calculations to reduce duplication.

    Purpose:
    - Centralize balance-related calculations (deviation, buy/sell diff) to ensure
      all modules use a consistent definition and behavior (including tolerances).
    - Provide small helpers for common reward shaping computations (activity bonus,
      position penalties, trading bonuses).

    Usage guidance:
    - Prefer `calculate_balance_penalty` for penalties driven by target ratios and
      tolerances used by the environment and calculators.
    - Use `calculate_buy_sell_diff` where a simple absolute imbalance metric is
      sufficient (e.g., monitoring, quick analysis scripts).

    Note:
    - Keep this module dependency-free (no heavy imports) so it can be used in
      unit tests and analysis scripts safely.
    """

    @staticmethod
    def calculate_pnl_reward(pnl: float, scaling: float = 1.0) -> float:
        """Calculates basic PnL reward."""
        return pnl * scaling

    @staticmethod
    def calculate_balance_penalty(
        action_counts: list[int],
        target_ratios: list[float],
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
    def calculate_balance_deviation_from_ratios(
        ratios: list[float], target_ratios: list[float]
    ) -> float:
        """Calculate sum of absolute deviations between action ratios and targets.

        Ratios are expected in 0..1. target_ratios may be shorter; comparison stops
        at the shortest length.
        """
        if not ratios or not target_ratios:
            return 0.0
        s = 0.0
        for i, r in enumerate(ratios):
            if i >= len(target_ratios):
                break
            s += abs(r - target_ratios[i])
        return s

    @staticmethod
    def calculate_balance_deviation_from_percentages(
        percentages: list[float], target_pct: float
    ) -> float:
        """Calculate sum of absolute deviations between percentages and a target percentage.

        Percentages are expected in the same scale as target_pct (e.g. 0..100).
        """
        if not percentages:
            return 0.0
        return sum(abs(p - target_pct) for p in percentages)

    @staticmethod
    def calculate_buy_sell_diff(buy: float, sell: float) -> float:
        """Return absolute difference between buy and sell ratios.

        Centralizes the simple BUY/SELL imbalance metric used in several places.
        """
        try:
            return abs(buy - sell)
        except Exception:
            return 0.0

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
        recent_actions: list[int],
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
