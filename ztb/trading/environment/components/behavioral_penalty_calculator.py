# mypy: disable-error-code=literal-required
"""
Behavioral Penalty Calculator - Component for calculating behavior-related penalties.
"""
from collections import deque
from typing import List
from ztb.trading.constants import ACTION_BUY, ACTION_HOLD, ACTION_SELL
from ztb.trading.environment.utils.config import EnvironmentConfig
from ztb.utils.logging_utils import get_logger


class BehavioralPenaltyCalculator:
    """
    Calculates penalties related to agent's behavior, such as action consistency and balance.
    """

    def __init__(self, config: EnvironmentConfig):
        """
        Initializes the BehavioralPenaltyCalculator.

        Args:
            config: Environment configuration object.
        """
        self.config = config
        self.logger = get_logger(self.__class__.__name__)
        self._load_settings()
        
        self.recent_actions: deque[int] = deque(maxlen=self.lookback)
        self._action_counts: List[int] = [0, 0, 0]  # [HOLD, BUY, SELL]

    def _load_settings(self):
        """Load settings from the environment configuration."""
        reward_settings = self.config.reward_settings

        if reward_settings and isinstance(reward_settings, dict):
            self.consistency_penalty_enabled = bool(reward_settings.get("consistency_penalty_enabled", True))
            self.penalty_value = float(reward_settings.get("consistency_penalty", -0.05))  # type: ignore
            self.lookback = int(reward_settings.get("consistency_lookback", 2))  # type: ignore
            
            # Settings for balance penalty
            self.balance_penalty_enabled = bool(reward_settings.get("balance_penalty_enabled", True))
            self.balance_penalty_value = float(reward_settings.get("balance_penalty", 1.0))  # type: ignore
            self.balance_penalty_tolerance = float(reward_settings.get("balance_penalty_tolerance", 0.05))  # type: ignore
            self.balance_penalty_min_actions = int(reward_settings.get("balance_penalty_min_actions", 10))  # type: ignore
            
            balance_penalty_targets = reward_settings.get("balance_penalty_targets", {})
            if not isinstance(balance_penalty_targets, dict):
                balance_penalty_targets = {}
            self.hold_target = float(balance_penalty_targets.get("hold_target", 0.4))
            self.buy_target = float(balance_penalty_targets.get("buy_target", 0.3))
            self.sell_target = float(balance_penalty_targets.get("sell_target", 0.3))
        else:
            # Default values if reward_settings is None or not a dict
            self.consistency_penalty_enabled = True
            self.penalty_value = -0.05
            self.lookback = 2
            self.balance_penalty_enabled = True
            self.balance_penalty_value = 1.0
            self.balance_penalty_tolerance = 0.05
            self.balance_penalty_min_actions = 10
            self.hold_target = 0.4
            self.buy_target = 0.3
            self.sell_target = 0.3

        if self.lookback <= 0:
            self.lookback = 1 # Ensure maxlen is positive
        
    def record_action(self, action: int):
        """
        Records the given action to the recent actions list and updates action counts.

        Args:
            action: The action to record.
        """
        self.recent_actions.append(action)
        if action == ACTION_HOLD:
            self._action_counts[0] += 1
        elif action == ACTION_BUY:
            self._action_counts[1] += 1
        elif action == ACTION_SELL:
            self._action_counts[2] += 1

    def calculate_consistency_penalty(self) -> float:
        """
        Calculates a penalty for inconsistent actions (e.g., BUY then SELL).

        Returns:
            The calculated penalty, or 0.0 if no penalty is applied.
        """
        if not self.consistency_penalty_enabled or len(self.recent_actions) < self.lookback:
            return 0.0

        # Check for a "whipsaw" pattern (e.g., BUY -> SELL or SELL -> BUY)
        last_action = self.recent_actions[-1]
        prev_action = self.recent_actions[-2]

        if (last_action == ACTION_BUY and prev_action == ACTION_SELL) or \
           (last_action == ACTION_SELL and prev_action == ACTION_BUY):
            self.logger.debug(f"Applying consistency penalty: {self.penalty_value}") if len(self.recent_actions) % 10 == 0 else None
            return self.penalty_value

        return 0.0

    def calculate_balance_penalty(self, action: int, action_bonus: float = 0.0) -> float:
        """
        Calculates a penalty to encourage balanced BUY and SELL actions,
        considering the hypothetical impact of the current action and any bonus.
        A higher bonus should make the agent more resilient to penalties.
        """
        if not self.balance_penalty_enabled:
            return 0.0

        hypothetical_counts = self._action_counts[:]
        action_index = -1
        if action == ACTION_BUY:
            action_index = 1
            hypothetical_counts[action_index] += 1
        elif action == ACTION_SELL:
            action_index = 2
            hypothetical_counts[action_index] += 1
        else:
            return 0.0

        total_actions = sum(hypothetical_counts)
        if total_actions < self.balance_penalty_min_actions:
            return 0.0

        buy_ratio = hypothetical_counts[1] / total_actions
        sell_ratio = hypothetical_counts[2] / total_actions

        penalty = 0.0
        deviation = 0.0
        
        if action == ACTION_BUY:
            deviation = buy_ratio - self.buy_target
        elif action == ACTION_SELL:
            deviation = sell_ratio - self.sell_target

        # Only penalize if the action increases the imbalance
        if deviation > self.balance_penalty_tolerance:
            # The bonus reduces the "effective" deviation that is penalized.
            # A larger bonus leads to a larger reduction in penalized deviation.
            bonus_effect = action_bonus / (self.balance_penalty_value + 1e-6)
            penalized_deviation = max(0, (deviation - self.balance_penalty_tolerance) - bonus_effect)
            penalty = penalized_deviation * self.balance_penalty_value

        return -penalty

    def reset(self):
        """Resets the internal state of the calculator."""
        self.recent_actions.clear()
        self._action_counts = [0, 0, 0]
