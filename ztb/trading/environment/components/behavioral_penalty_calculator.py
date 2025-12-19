# mypy: disable-error-code=literal-required
"""
Behavioral Penalty Calculator - Component for calculating behavior-related penalties.
"""
from collections import deque
from typing import Any, Dict, List, Optional

from ztb.trading.constants import ACTION_BUY, ACTION_HOLD, ACTION_SELL
from ztb.trading.environment.utils.config import EnvironmentConfig
from ztb.utils.logging_utils import get_logger


class BehavioralPenaltyCalculator:
    """
    Calculates penalties related to agent's behavior, such as action consistency and balance.

    SAC v448 Layer 2 enhancements:
    - TrendDetector integration for trend-aware balance target adjustments
    - Emergency intervention for extreme bias (>30% BUY-SELL deviation)
    """

    def __init__(self, config: EnvironmentConfig, trend_detector: Optional[Any] = None):
        """
        Initializes the BehavioralPenaltyCalculator.

        Args:
            config: Environment configuration object.
            trend_detector: Optional TrendDetector instance for trend-aware adjustments.
        """
        self.config = config
        self.trend_detector = trend_detector
        self.logger = get_logger(self.__class__.__name__)
        self._load_settings()

        # Keep enough history to support all lookback-based calculations.
        # Forced balance and balance-penalty logic operate on the same action window,
        # so ensure we can hold at least that many entries before trimming.
        # Only consider the lookback-based windows for deque sizing.
        # This keeps the sliding-window semantics focused on actual lookbacks
        # (consistency, skewness, entropy) and avoids oversized buffers caused
        # by unrelated min_action thresholds (which can be large defaults).
        history_windows = [
            self.lookback,
            getattr(self, "skewness_lookback", 0),
            getattr(self, "action_entropy_lookback", 0),
        ]
        max_lookback = max(1, max(history_windows))

        # Reserve space for lookback-based checks. Use max_lookback as the deque length
        # so that recent_actions contains exactly the number of samples expected by tests.
        self.recent_actions: deque[int] = deque(maxlen=max_lookback)
        self._action_counts: List[int] = [0, 0, 0]  # [HOLD, BUY, SELL]

    def _load_settings(self):
        """Load settings from the environment configuration."""
        # Allow passing either an EnvironmentConfig object with .reward_settings
        # or passing a dict directly as reward_settings (convenience for tests)
        if isinstance(self.config, dict):
            reward_settings = self.config
        else:
            reward_settings = getattr(self.config, "reward_settings", None)
            # Support EnvironmentConfig.formatted configs that store optimization under 'behavior_optimization'
            if reward_settings is None and hasattr(
                self.config, "behavior_optimization"
            ):
                reward_settings = getattr(self.config, "behavior_optimization", None)

        # Flexible accessor for reward settings: support dicts or RewardSettings dataclass.
        def _rs_get(key, default=None):
            if reward_settings is None:
                return default
            if isinstance(reward_settings, dict):
                # Support legacy/nested structure under 'behavior' key first (override flat keys)
                behavior = (
                    reward_settings.get("behavior")
                    if isinstance(reward_settings, dict)
                    else None
                )
                # Support alternate key name 'behavior_optimization' used in some configs
                if isinstance(behavior, dict) is False and isinstance(
                    reward_settings, dict
                ):
                    behavior = reward_settings.get("behavior_optimization")
                if isinstance(behavior, dict):
                    # If the nested behavior dict contains a generic key, prefer it.
                    if key in behavior:
                        # Return simple scalar values; prefer the nested dict handling
                        # for compound configuration objects (e.g. 'consistency_penalty').
                        val = behavior.get(key, default)
                        if not isinstance(val, dict):
                            return val
                    # consistency penalty
                    if key in [
                        "consistency_penalty_enabled",
                        "consistency_penalty",
                        "consistency_lookback",
                    ]:
                        cp = behavior.get("consistency_penalty")
                        if isinstance(cp, dict):
                            if key == "consistency_penalty_enabled":
                                return cp.get("enabled", default)
                            if key == "consistency_penalty":
                                # nested key names could be 'value' or 'penalty'
                                return cp.get("value", cp.get("penalty", default))
                            if key == "consistency_lookback":
                                return cp.get("lookback", default)

                    # emergency intervention settings (allow flat or nested under 'emergency_intervention')
                    if (
                        key.startswith("emergency_intervention")
                        or key == "balance_penalty_min_actions"
                    ):
                        # First, look for nested dict behavior['emergency_intervention']
                        ei = behavior.get("emergency_intervention")
                        if isinstance(ei, dict):
                            if key == "emergency_intervention_enabled":
                                return ei.get("enabled", default)
                            if key == "emergency_intervention_threshold":
                                return ei.get("threshold", default)
                            if key == "emergency_intervention_penalty":
                                return ei.get("penalty", ei.get("value", default))
                            if key == "balance_penalty_min_actions":
                                return ei.get("balance_penalty_min_actions", default)
                        # Otherwise, allow flat behavior['emergency_intervention_enabled'] etc
                        if key in behavior:
                            return behavior.get(key, default)

                    # Map nested 'balance_penalty_targets' keys, which might be in behavior
                    if key in ["buy_target", "sell_target", "hold_target"]:
                        targs = behavior.get("balance_penalty_targets")
                        if isinstance(targs, dict):
                            return targs.get(key, default)

                # Fallback to flat key if nested not present
                val = reward_settings.get(key, default)
                if val is not None:
                    return val
                # reward_settings = { 'behavior': { 'consistency_penalty': {'enabled': False, 'value': 0.1, 'lookback': 3}}}
                behavior = (
                    reward_settings.get("behavior")
                    if isinstance(reward_settings, dict)
                    else None
                )
                if isinstance(behavior, dict) is False and isinstance(
                    reward_settings, dict
                ):
                    behavior = reward_settings.get("behavior_optimization")
                if isinstance(behavior, dict):
                    # consistency penalty
                    if key in [
                        "consistency_penalty_enabled",
                        "consistency_penalty",
                        "consistency_lookback",
                    ]:
                        cp = behavior.get("consistency_penalty")
                        if isinstance(cp, dict):
                            if key == "consistency_penalty_enabled":
                                return cp.get("enabled", default)
                            if key == "consistency_penalty":
                                # nested key names could be 'value' or 'penalty'
                                return cp.get("value", cp.get("penalty", default))
                            if key == "consistency_lookback":
                                return cp.get("lookback", default)

                    # emergency intervention settings
                    if (
                        key.startswith("emergency_intervention")
                        or key == "balance_penalty_min_actions"
                    ):
                        ei = behavior.get("emergency_intervention")
                        if isinstance(ei, dict):
                            if key == "emergency_intervention_enabled":
                                return ei.get("enabled", default)
                            if key == "emergency_intervention_threshold":
                                return ei.get("threshold", default)
                            if key == "emergency_intervention_penalty":
                                return ei.get("penalty", ei.get("value", default))
                            if key == "balance_penalty_min_actions":
                                return ei.get("balance_penalty_min_actions", default)
                        if key in behavior:
                            return behavior.get(key, default)

                    # Map nested 'balance_penalty_targets' keys, which might be in behavior
                    if key in ["buy_target", "sell_target", "hold_target"]:
                        targs = behavior.get("balance_penalty_targets")
                        if isinstance(targs, dict):
                            return targs.get(key, default)
                return default
            # dataclass or object: first try attribute, then custom_reward_params
            # Prefer explicit overrides in custom_reward_params (if provided)
            if hasattr(reward_settings, "custom_reward_params") and isinstance(
                reward_settings.custom_reward_params, dict
            ):
                val = reward_settings.custom_reward_params.get(key, None)
                if val is not None:
                    return val
            if hasattr(reward_settings, key):
                return getattr(reward_settings, key)
            return default

        if reward_settings:
            self.consistency_penalty_enabled = bool(
                _rs_get("consistency_penalty_enabled", True)
            )
            # Ensure penalty is negative (penalty magnitude is stored as positive in configs)
            self.penalty_value = -abs(float(_rs_get("consistency_penalty", 0.05)))  # type: ignore
            self.lookback = int(_rs_get("consistency_lookback", 50))  # type: ignore
            # Clamp lookback=0 to minimum 1 for window semantics expected by tests
            if self.lookback == 0:
                self.lookback = 1
            # Minimum non-HOLD action count required to evaluate consistency penalties
            self.consistency_min_actions = int(_rs_get("consistency_min_actions", 2))

            # Settings for balance penalty
            self.balance_penalty_enabled = bool(
                _rs_get("balance_penalty_enabled", True)
            )
            self.balance_penalty_value = float(_rs_get("balance_penalty", 1.0))  # type: ignore
            self.balance_penalty_tolerance = float(
                _rs_get("balance_penalty_tolerance", 0.05)
            )  # type: ignore
            self.balance_penalty_min_actions = int(
                _rs_get("balance_penalty_min_actions", 1)
            )  # type: ignore

            # Forced balance stage shares the same action history window; default to 100 so
            # existing configs without the key are unaffected while explicit configs enlarge it.
            self.forced_balance_min_actions = int(
                _rs_get("forced_balance_min_actions", 100)
            )

            # Emergency intervention settings (SAC v448 Layer 2)
            # MODIFIED: Disabled by default to prevent early convergence to HOLD
            val = _rs_get("emergency_intervention_enabled", False)
            self.emergency_intervention_enabled = bool(val)
            self.emergency_intervention_threshold = float(
                _rs_get("emergency_intervention_threshold", 0.30)
            )  # type: ignore
            self.emergency_intervention_penalty = float(
                _rs_get("emergency_intervention_penalty", -500.0)
            )  # type: ignore

            # Trend-aware balance target adjustment settings
            self.trend_adjustment_enabled = bool(
                _rs_get("trend_adjustment_enabled", True)
            )
            self.trend_adjustment_strength = float(
                _rs_get("trend_adjustment_strength", 0.1)
            )  # type: ignore

            balance_penalty_targets = _rs_get("balance_penalty_targets", {})
            if not isinstance(balance_penalty_targets, dict):
                balance_penalty_targets = {}
            self.hold_target = float(balance_penalty_targets.get("hold_target", 0.4))
            self.buy_target = float(balance_penalty_targets.get("buy_target", 0.3))
            self.sell_target = float(balance_penalty_targets.get("sell_target", 0.3))
            # Additional shaping: reward actions that move distribution closer to targets
            self.balance_shaping_enabled = bool(
                _rs_get("balance_shaping_enabled", True)
            )
            self.balance_shaping_value = float(_rs_get("balance_shaping_value", 0.5))
            # Entropy shaping encourages diversity in actions (prevents collapse to pure SELL/BUY)
            self.action_entropy_shaping_enabled = bool(
                _rs_get("action_entropy_shaping_enabled", True)
            )
            self.action_entropy_shaping_value = float(
                _rs_get("action_entropy_shaping_value", 0.01)
            )
            self.action_entropy_lookback = int(
                _rs_get("action_entropy_lookback", max(10, self.lookback))
            )
            # Settings for skewness penalty (penalize extreme SELL/BUY skew)
            self.skewness_penalty_enabled = bool(
                _rs_get("skewness_penalty_enabled", False)
            )
            self.skewness_penalty_value = float(_rs_get("skewness_penalty_value", 0.0))
            self.skewness_penalty_tolerance = float(
                _rs_get("skewness_penalty_tolerance", 0.05)
            )
            self.skewness_lookback = int(
                _rs_get("skewness_lookback", max(10, self.lookback))
            )
        else:
            # Default values if reward_settings is None or not a dict
            self.consistency_penalty_enabled = True
            self.penalty_value = -0.05
            self.lookback = 50
            self.balance_penalty_enabled = True
            self.balance_penalty_value = 1.0
            self.balance_penalty_tolerance = 0.05
            self.balance_penalty_min_actions = 10
            self.hold_target = 0.4
            self.buy_target = 0.3
            self.sell_target = 0.3
            self.forced_balance_min_actions = 0
            self.emergency_intervention_enabled = False

        if self.lookback < 0:
            self.lookback = 0  # 0 disables consistency penalty

    def record_action(self, action: int):
        """
        Records the given action to the recent actions list and updates action counts.

        Args:
            action: The action to record.
        """
        popped = None
        if (
            self.recent_actions.maxlen is not None
            and len(self.recent_actions) == self.recent_actions.maxlen
        ):
            # the left-most item will be popped by append
            popped = self.recent_actions[0]
        self.recent_actions.append(action)

        # update cached counts for compatibility with older code
        if popped is not None:
            if popped == ACTION_HOLD:
                self._action_counts[0] = max(0, self._action_counts[0] - 1)
            elif popped == ACTION_BUY:
                self._action_counts[1] = max(0, self._action_counts[1] - 1)
            elif popped == ACTION_SELL:
                self._action_counts[2] = max(0, self._action_counts[2] - 1)

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
        # Consistency penalty disabled if lookback <= 0
        if not self.consistency_penalty_enabled or self.lookback <= 0:
            return 0.0

        # Check for a "whipsaw" pattern (e.g., BUY -> SELL or SELL -> BUY)
        # If the last recorded action is HOLD (no change), do not penalize
        if len(self.recent_actions) == 0:
            return 0.0
        # most recent actual action (including HOLD) - if it's HOLD skip
        if self.recent_actions[-1] == ACTION_HOLD:
            return 0.0

        # Consider the last lookback entries plus the current action.
        # We reserve an extra slot in the deque to store the current action
        # so that a lookback of 1 will include [previous, current].
        if self.lookback > 0:
            window = list(self.recent_actions)[-(self.lookback + 1) :]
        else:
            # lookback == 0 disables consistency penalty earlier, but be safe
            window = list(self.recent_actions)

        # Extract non-HOLD actions within the window
        non_hold = [a for a in window if a != ACTION_HOLD]

        # Require a minimum number of non-HOLD actions to apply consistency penalty
        if len(non_hold) < getattr(self, "consistency_min_actions", 2):
            return 0.0

        # Compare the last two non-HOLD actions (if present) to detect a reversal
        last_action = non_hold[-1] if len(non_hold) >= 1 else None
        prev_action = non_hold[-2] if len(non_hold) >= 2 else None
        if last_action is None or prev_action is None:
            return 0.0

        # Apply penalty only on a direct reversal (BUY <-> SELL)
        if (last_action == ACTION_BUY and prev_action == ACTION_SELL) or (
            last_action == ACTION_SELL and prev_action == ACTION_BUY
        ):
            # Log periodically to avoid log spam
            if len(self.recent_actions) % 10 == 0:
                self.logger.debug(
                    f"Applying consistency penalty: {self.penalty_value} (reversal detected)"
                )
            return self.penalty_value

        return 0.0

    def calculate_balance_penalty(
        self, action: int, action_bonus: float = 0.0
    ) -> float:
        """
        Calculates a penalty to encourage balanced BUY and SELL actions,
        considering the hypothetical impact of the current action and any bonus.
        A higher bonus should make the agent more resilient to penalties.
        """
        if not self.balance_penalty_enabled:
            return 0.0

        hypothetical_counts = self._get_recent_counts()
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
        # Use adjusted targets when trend adjustment is enabled
        adjusted_targets = self._adjust_targets_by_trend()
        adj_buy_target = adjusted_targets.get("buy_target", self.buy_target)
        adj_sell_target = adjusted_targets.get("sell_target", self.sell_target)
        if action == ACTION_BUY:
            deviation = buy_ratio - adj_buy_target
        elif action == ACTION_SELL:
            deviation = sell_ratio - adj_sell_target

        # Only penalize if the action increases the imbalance
        if deviation > self.balance_penalty_tolerance:
            # The bonus reduces the "effective" deviation that is penalized.
            # A larger bonus leads to a larger reduction in penalized deviation.
            bonus_effect = action_bonus / (self.balance_penalty_value + 1e-6)
            penalized_deviation = max(
                0, (deviation - self.balance_penalty_tolerance) - bonus_effect
            )
            penalty = penalized_deviation * self.balance_penalty_value

        return -penalty

    def calculate_balance_shaping(self, action: int) -> float:
        """
        Calculates a small shaping reward for actions that reduce overall deviation from
        the target distribution. Returns a positive reward if the hypothetical action
        reduces deviation, otherwise 0 or a small negative value.
        """
        if not getattr(self, "balance_shaping_enabled", False):
            return 0.0

        hypothetical_counts = self._get_recent_counts()
        if action == ACTION_BUY:
            hypothetical_counts[1] += 1
        elif action == ACTION_SELL:
            hypothetical_counts[2] += 1
        else:
            return 0.0

        total_actions = sum(hypothetical_counts)
        if total_actions == 0:
            return 0.0

        # Use adjusted targets when trend adjustment is enabled
        adjusted_targets = self._adjust_targets_by_trend()
        adj_buy_target = adjusted_targets.get("buy_target", self.buy_target)
        adj_sell_target = adjusted_targets.get("sell_target", self.sell_target)

        # compute current deviation
        recent_counts = self._get_recent_counts()
        current_tot = sum(recent_counts) or 1
        current_buy = recent_counts[1] / current_tot
        current_sell = recent_counts[2] / current_tot
        current_deviation = abs(current_buy - adj_buy_target) + abs(
            current_sell - adj_sell_target
        )

        # compute new deviation
        new_buy = hypothetical_counts[1] / total_actions
        new_sell = hypothetical_counts[2] / total_actions
        new_deviation = abs(new_buy - adj_buy_target) + abs(new_sell - adj_sell_target)

        # shaping reward proportional to improvement (reduction) in deviation
        improvement = current_deviation - new_deviation
        shaping = max(0.0, improvement) * self.balance_shaping_value
        return shaping

    def calculate_skewness_penalty(self) -> float:
        """
        Calculates a penalty based on the imbalance (skew) between BUY and SELL.

        Returns:
            Negative penalty to apply to current reward (0.0 if none).
        """
        if not self.skewness_penalty_enabled:
            return 0.0

        # Use sliding-window action counts (BUY index 1, SELL index 2)
        counts = self._get_recent_counts(self.skewness_lookback)
        total = sum(counts)
        if total < self.skewness_lookback:
            return 0.0

        buy_ratio = counts[1] / total if total > 0 else 0.0
        sell_ratio = counts[2] / total if total > 0 else 0.0

        # compute skew: positive=SELL-heavy, negative=BUY-heavy
        skew = sell_ratio - buy_ratio

        # Only penalize when skew exceeds tolerance in either direction
        if skew > self.skewness_penalty_tolerance:
            penalty = (
                skew - self.skewness_penalty_tolerance
            ) * self.skewness_penalty_value
            self.logger.debug(
                f"Applying skewness penalty: {penalty:.5f} (skew {skew:.4f})"
            )
            return -penalty
        if -skew > self.skewness_penalty_tolerance:
            penalty = (
                (-skew) - self.skewness_penalty_tolerance
            ) * self.skewness_penalty_value
            self.logger.debug(
                f"Applying skewness penalty (BUY-heavy): {penalty:.5f} (skew {skew:.4f})"
            )
            return -penalty

        return 0.0

    def calculate_action_entropy_shaping(self) -> float:
        """
        Return a small positive shaping term if the recent action entropy increases,
        encouraging more diverse actions and avoiding collapse to a single action.
        It uses the last `action_entropy_lookback` actions to compute entropy.
        """
        if not getattr(self, "action_entropy_shaping_enabled", False):
            return 0.0

        # if not enough actions, skip
        hist_len = len(self.recent_actions)
        if hist_len < getattr(self, "action_entropy_lookback", 10):
            return 0.0

        counts = [0, 0, 0]
        for a in list(self.recent_actions)[-self.action_entropy_lookback :]:
            if a == ACTION_HOLD:
                counts[0] += 1
            elif a == ACTION_BUY:
                counts[1] += 1
            elif a == ACTION_SELL:
                counts[2] += 1

        total = sum(counts)
        if total == 0:
            return 0.0

        probs = [c / total for c in counts]
        import math

        entropy = -sum(p * math.log(p) for p in probs if p > 0)

        # target entropy: max is log(3) for 3 actions; we favor higher entropy
        target = getattr(self, "action_entropy_target", math.log(3))
        if entropy < target:
            # reward proportional to the shortfall
            shortfall = target - entropy
            return self.action_entropy_shaping_value * shortfall
        return 0.0

    def calculate_emergency_intervention(self) -> float:
        """
        SAC v448 Layer 2: Emergency intervention penalty for extreme action bias.

        Applies a strong penalty (-500 by default) when the BUY-SELL difference
        exceeds 30%, preventing bias collapse to >90% BUY or >90% SELL.

        Returns:
            Emergency penalty if bias is extreme, 0.0 otherwise.
        """
        if not getattr(self, "emergency_intervention_enabled", False):
            return 0.0

        counts = self._get_recent_counts()
        total = sum(counts)

        if total < self.balance_penalty_min_actions:
            return 0.0

        buy_ratio = counts[1] / total
        sell_ratio = counts[2] / total
        buy_sell_diff = abs(buy_ratio - sell_ratio)

        threshold = getattr(self, "emergency_intervention_threshold", 0.30)
        if buy_sell_diff > threshold:
            penalty = getattr(self, "emergency_intervention_penalty", -500.0)
            self.logger.warning(
                f"Emergency intervention triggered: BUY-SELL diff={buy_sell_diff:.2%} "
                f"(BUY={buy_ratio:.2%}, SELL={sell_ratio:.2%}), penalty={penalty}"
            )
            return penalty

        return 0.0

    def _adjust_targets_by_trend(self) -> Dict[str, float]:
        """
        SAC v448 Layer 2: Adjust balance targets based on market trend.

        Uptrend: Increase buy_target, decrease sell_target
        Downtrend: Decrease buy_target, increase sell_target
        Neutral: Use baseline targets

        Returns:
            Dictionary with adjusted hold_target, buy_target, sell_target.
        """
        if (
            not getattr(self, "trend_adjustment_enabled", True)
            or self.trend_detector is None
        ):
            return {
                "hold_target": self.hold_target,
                "buy_target": self.buy_target,
                "sell_target": self.sell_target,
            }

        trend_signal = self.trend_detector.get_trend_signal()
        strength = getattr(self, "trend_adjustment_strength", 0.1)

        # Adjust targets: positive trend favors BUY, negative favors SELL
        buy_adjustment = trend_signal * strength
        sell_adjustment = -trend_signal * strength

        adjusted_buy = max(0.1, min(0.5, self.buy_target + buy_adjustment))
        adjusted_sell = max(0.1, min(0.5, self.sell_target + sell_adjustment))

        # Normalize to maintain total = 1.0
        total = adjusted_buy + adjusted_sell
        if total > 0.8:  # Leave at least 20% for HOLD
            scale = 0.8 / total
            adjusted_buy *= scale
            adjusted_sell *= scale

        adjusted_hold = 1.0 - adjusted_buy - adjusted_sell

        return {
            "hold_target": adjusted_hold,
            "buy_target": adjusted_buy,
            "sell_target": adjusted_sell,
        }

    def get_target_ratios(self) -> Dict[str, float]:
        """Public accessor for the current target ratios (hold, buy, sell).

        This is used by other components (e.g., RewardCalculator) to query
        what behavioral targets should currently be used when enforcing balance
        penalties. It delegates to _adjust_targets_by_trend to honor trend
        adjustments when enabled.
        """
        return self._adjust_targets_by_trend()

    def reset(self):
        """Resets the internal state of the calculator."""
        self.recent_actions.clear()
        self._action_counts = [0, 0, 0]
        try:
            if self.trend_detector is not None and hasattr(
                self.trend_detector, "reset"
            ):
                self.trend_detector.reset()
        except Exception:
            self.logger.exception("Failed to reset trend_detector")

    def _get_recent_counts(self, lookback: int | None = None) -> List[int]:
        """Return counts for [HOLD, BUY, SELL] from recent_actions for a specified lookback.

        If lookback is None use entire deque (full history held in the deque). This keeps counting
        consistent with sliding-window semantics.
        """
        if lookback is None:
            arr = list(self.recent_actions)
        else:
            arr = list(self.recent_actions)[-lookback:]
        counts = [0, 0, 0]
        for a in arr:
            if a == ACTION_HOLD:
                counts[0] += 1
            elif a == ACTION_BUY:
                counts[1] += 1
            elif a == ACTION_SELL:
                counts[2] += 1
        return counts
