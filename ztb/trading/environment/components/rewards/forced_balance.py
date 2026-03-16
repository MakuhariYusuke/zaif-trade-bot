import dataclasses
import logging
from typing import Any

from ztb.trading.constants import ACTION_BUY, ACTION_HOLD, ACTION_SELL

from .base import RewardComponent, RewardContext

class ForcedBalanceReward(RewardComponent):
    """
    Reward component that encourages corrective actions toward configured target ratios.
    Ported from RewardCalculator._calculate_forced_balance_reward.
    """

    def __init__(self) -> None:
        self.logger = logging.getLogger(__name__)
        self._forced_balance_log_counter = 0
        self._forced_balance_log_interval = 100
        self._forced_balance_last_state = "balanced"
        self._forced_balance_last_summary_step = 0
        self._forced_balance_summary_interval = 1000
        self.ACTION_INDEX_NAMES = ["HOLD", "BUY", "SELL"]
        self.last_reward_details: dict[str, float] = {}

    def get_name(self) -> str:
        return "forced_balance"

    def _get_nested_setting(self, context: RewardContext, key: str) -> Any:
        """Helper to retrieve nested settings from reward_settings using dot notation."""
        if not context.reward_settings:
            return None

        keys = key.split(".")
        current: Any = context.reward_settings

        for k in keys:
            if dataclasses.is_dataclass(current):
                if hasattr(current, k):
                    current = getattr(current, k)
                else:
                    return None
            elif isinstance(current, dict):
                if k in current:
                    current = current[k]
                else:
                    return None
            else:
                return None
        return current

    def _get_setting(self, context: RewardContext, key: str, default: Any, cast=None) -> Any:
        """Component-specialized lookup that still supports the base `cast` arg."""
        # Prefer nested lookup for dot-separated keys; otherwise use base helper
        val = self._get_nested_setting(context, key)
        if val is not None:
            try:
                return cast(val) if cast is not None else val
            except (ValueError, TypeError):
                return default

        # Check custom_reward_params for keys that match our component
        if context.reward_settings and hasattr(context.reward_settings, "custom_reward_params"):
            custom_params = context.reward_settings.custom_reward_params
            if isinstance(custom_params, dict):
                # Try direct key match first
                if key in custom_params:
                    val = custom_params[key]
                    try:
                        return cast(val) if cast is not None else val
                    except (ValueError, TypeError):
                        return default
                # Try component-prefixed key (e.g., forced_balance_min_actions)
                # But avoid double-prefixing if key already starts with component name
                if not key.startswith("forced_balance_"):
                    component_key = f"forced_balance_{key}"
                    if component_key in custom_params:
                        val = custom_params[component_key]
                        try:
                            return cast(val) if cast is not None else val
                        except (ValueError, TypeError):
                            return default

        return super()._get_setting(context, key, default, cast=cast)

    @staticmethod
    def _map_forced_balance_penalty_static(
        deviation: float,
        severity: float,
        penalty_scale: float,
        thresh_small: float,
        thresh_medium: float,
        thresh_large: float,
        penalty_small: float,
        penalty_medium: float,
        penalty_large: float,
        penalty_very_large: float,
    ) -> float:
        """Convert deviation above target into a scaled penalty value."""
        severity_multiplier = 1.0 + 0.5 * min(1.0, severity)
        if deviation < thresh_small:
            base_penalty = penalty_small
        elif deviation < thresh_medium:
            base_penalty = penalty_medium
        elif deviation < thresh_large:
            base_penalty = penalty_large
        else:
            base_penalty = penalty_very_large
        return base_penalty * penalty_scale * severity_multiplier

    @staticmethod
    def _map_forced_balance_bonus_static(
        deviation: float,
        severity: float,
        bonus_scale: float,
        thresh_small: float,
        thresh_medium: float,
        bonus_small: float,
        bonus_medium: float,
        bonus_large: float,
    ) -> float:
        """Convert deviation below target into a bonus encouraging corrective actions."""
        severity_multiplier = 1.0 + 0.5 * min(1.0, severity)
        if deviation < thresh_small:
            base_bonus = bonus_small
        elif deviation < thresh_medium:
            base_bonus = bonus_medium
        else:
            base_bonus = bonus_large
        return base_bonus * bonus_scale * severity_multiplier

    def _map_forced_balance_penalty(
        self, context: RewardContext, deviation: float, severity: float
    ) -> float:
        return self._map_forced_balance_penalty_static(
            deviation=deviation,
            severity=severity,
            penalty_scale=self._get_setting_float(
                context, "forced_balance.penalty.scale", 1.0
            ),
            thresh_small=self._get_setting_float(
                context, "forced_balance.penalty.threshold_small", 0.05
            ),
            thresh_medium=self._get_setting_float(
                context, "forced_balance.penalty.threshold_medium", 0.1
            ),
            thresh_large=self._get_setting_float(
                context, "forced_balance.penalty.threshold_large", 0.2
            ),
            penalty_small=self._get_setting_float(
                context, "forced_balance.penalty.value_small_deviation", 1.0
            ),
            penalty_medium=self._get_setting_float(
                context, "forced_balance.penalty.value_medium_deviation", 2.5
            ),
            penalty_large=self._get_setting_float(
                context, "forced_balance.penalty.value_large_deviation", 5.0
            ),
            penalty_very_large=self._get_setting_float(
                context, "forced_balance.penalty.value_very_large_deviation", 10.0
            ),
        )

    def _map_forced_balance_bonus(
        self, context: RewardContext, deviation: float, severity: float
    ) -> float:
        return self._map_forced_balance_bonus_static(
            deviation=deviation,
            severity=severity,
            bonus_scale=self._get_setting_float(
                context, "forced_balance.bonus.scale", 1.0
            ),
            thresh_small=self._get_setting_float(
                context, "forced_balance.bonus.threshold_small", 0.05
            ),
            thresh_medium=self._get_setting_float(
                context, "forced_balance.bonus.threshold_medium", 0.1
            ),
            bonus_small=self._get_setting_float(
                context, "forced_balance.bonus.value_small_deviation", 6.0
            ),
            bonus_medium=self._get_setting_float(
                context, "forced_balance.bonus.value_medium_deviation", 12.0
            ),
            bonus_large=self._get_setting_float(
                context, "forced_balance.bonus.value_large_deviation", 20.0
            ),
        )

    def calculate(self, context: RewardContext) -> float:
        self.last_reward_details = {}

        # Use action counts from context
        action_counts = context.action_counts
        if not action_counts:
            # Fallback if not provided
            return 0.0

        total_actions = sum(action_counts)

        # Log control
        should_log_detailed = (
            self._forced_balance_log_counter % self._forced_balance_log_interval == 0
        )
        self._forced_balance_log_counter += 1

        min_actions = self._get_setting_int(context, "forced_balance_min_actions", 0)
        self.logger.debug(f"after _get_setting_int, min_actions={min_actions}")
        exploration_reward = self._get_setting_float(
            context, "forced_balance_exploration_reward", 2.0
        )

        self.logger.debug(
            f"min_actions={min_actions}, exploration_reward={exploration_reward}, total_actions={total_actions}"
        )
        self.logger.debug(f"context.reward_settings={context.reward_settings}")
        if hasattr(context.reward_settings, 'custom_reward_params'):
            self.logger.debug(f"custom_reward_params={context.reward_settings.custom_reward_params}")

        if total_actions < min_actions:
            if should_log_detailed:
                self.logger.warning(
                    f"Forced balance: early phase (total_actions={total_actions} < {min_actions}), using exploration reward"
                )
            self.last_reward_details["base_reward"] = exploration_reward
            return exploration_reward

        # 408# B1 防御: min_actions=0 かつ total_actions=0 のゼロ除算回避
        if total_actions == 0:
            return 0.0

        # Calculate ratios and deviations
        action_ratios = [count / total_actions for count in action_counts]

        # Get targets from context
        target_ratios_dict = context.target_ratios
        # Map dict to list based on ACTION_INDEX_NAMES order
        target_ratios = [
            target_ratios_dict.get(name, 1.0 / len(self.ACTION_INDEX_NAMES))
            for name in self.ACTION_INDEX_NAMES
        ]

        # Normalize targets if they don't sum to 1 (safety check)
        total_target = sum(target_ratios)
        if total_target > 0:
            target_ratios = [t / total_target for t in target_ratios]

        signed_deviations = [
            actual - target for actual, target in zip(action_ratios, target_ratios)
        ]
        abs_deviations = [abs(dev) for dev in signed_deviations]

        rms_deviation = (
            sum(dev**2 for dev in signed_deviations) / len(signed_deviations)
        ) ** 0.5
        max_abs_deviation = max(abs_deviations)
        max_over_deviation = max(
            (dev for dev in signed_deviations if dev > 0), default=0.0
        )
        max_under_deviation = min(
            (dev for dev in signed_deviations if dev < 0), default=0.0
        )

        balance_broken_threshold = self._get_setting_float(
            context, "forced_balance_threshold", 0.15
        )
        is_imbalanced = max_abs_deviation > balance_broken_threshold

        # Emergency intervention penalty
        emergency_penalty = 0.0
        if context.behavioral_penalty_calculator:
            emergency_penalty = (
                context.behavioral_penalty_calculator.calculate_emergency_intervention()
            )

        if emergency_penalty < 0:
            self.last_reward_details["emergency_intervention"] = emergency_penalty

        # State logging logic
        state_parts = []
        for idx, dev in enumerate(signed_deviations):
            if abs(dev) > balance_broken_threshold:
                direction = "over" if dev > 0 else "under"
                state_parts.append(
                    f"{self.ACTION_INDEX_NAMES[idx]}_{direction}:{abs(dev):.3f}"
                )
        current_state = "|".join(state_parts) if state_parts else "balanced"

        if should_log_detailed or current_state != self._forced_balance_last_state:
            deviations_str = ", ".join(
                f"{name}={ratio:.3f} ({dev:+.3f})"
                for name, ratio, dev in zip(
                    self.ACTION_INDEX_NAMES, action_ratios, signed_deviations
                )
            )
            self.logger.warning(
                f"Forced balance: total_actions={total_actions}, rms_dev={rms_deviation:.3f}, "
                f"max_dev={max_abs_deviation:.3f}, max_over={max_over_deviation:.3f}, "
                f"max_under={abs(max_under_deviation):.3f}, state={current_state}, deviations=[{deviations_str}]"
            )
            self._forced_balance_last_state = current_state

        if (
            context.step - self._forced_balance_last_summary_step
            >= self._forced_balance_summary_interval
        ):
            ratios_str = ", ".join(f"{ratio:.3f}" for ratio in action_ratios)
            deviation_summary = ", ".join(f"{dev:+.3f}" for dev in signed_deviations)
            self.logger.info(
                f"Forced balance SUMMARY [Step {context.step}]: total_actions={total_actions}, "
                f"ratios=[{ratios_str}], signed_dev=[{deviation_summary}], "
                f"rms_dev={rms_deviation:.3f}, max_dev={max_abs_deviation:.3f}, "
                f"state={current_state}, counts={action_counts}"
            )
            self._forced_balance_last_summary_step = context.step

        if not is_imbalanced:
            balanced_reward = self._get_setting_float(
                context, "forced_balance_balanced_reward", 2.0
            )
            final_reward = balanced_reward + emergency_penalty
            self.last_reward_details["base_reward"] = final_reward
            return final_reward

        global_penalty_scale = self._get_setting_float(
            context, "forced_balance_global_penalty_scale", 0.0
        )
        global_pressure = -global_penalty_scale * max_abs_deviation

        # Map action values (ACTION_HOLD=0, ACTION_BUY=1, ACTION_SELL=-1; legacy 2=SELL)
        # to the [HOLD, BUY, SELL] index used by signed_deviations.
        action_val = context.action
        if action_val == 2:
            action_val = ACTION_SELL
        if action_val == ACTION_HOLD:
            action_idx = 0
        elif action_val == ACTION_BUY:
            action_idx = 1
        elif action_val == ACTION_SELL:
            action_idx = 2
        else:
            action_idx = 0

        current_deviation = signed_deviations[action_idx]

        # Trend-aware penalty adjustment: reduce penalty if action aligns with trend
        trend_signal = 0.0
        trend_strength = 0.0
        try:
            if (
                hasattr(context, "behavioral_penalty_calculator")
                and context.behavioral_penalty_calculator
            ):
                td = context.behavioral_penalty_calculator.trend_detector
                if td and hasattr(td, "get_trend_signal"):
                    trend_signal = float(td.get_trend_signal() or 0.0)
                trend_strength = (
                    float(
                        getattr(
                            context, "behavioral_penalty_calculator", None
                        ).trend_adjustment_strength
                    )
                    if hasattr(context, "behavioral_penalty_calculator")
                    and getattr(
                        context.behavioral_penalty_calculator,
                        "trend_adjustment_strength",
                        None,
                    )
                    is not None
                    else 0.1
                )
        except Exception:
            trend_signal = 0.0
            trend_strength = 0.0

        if current_deviation > 0:
            penalty = self._map_forced_balance_penalty(
                context, current_deviation, max_abs_deviation
            )
            # If the trend favors this action (eg. trend_signal >0 for BUY), reduce penalty
            try:
                favored = (action_val == ACTION_BUY and trend_signal > 0) or (
                    action_val == ACTION_SELL and trend_signal < 0
                )
                if favored and trend_strength:
                    # reduce penalty proportionally to trend strength
                    reduction = min(0.8, abs(trend_signal) * trend_strength)
                    adjusted_penalty = penalty * (1.0 - reduction)
                else:
                    adjusted_penalty = penalty
            except Exception:
                adjusted_penalty = penalty

            reward = global_pressure - adjusted_penalty + emergency_penalty
            self.last_reward_details["imbalance_penalty"] = -penalty
        elif current_deviation < 0:
            bonus = self._map_forced_balance_bonus(
                context, abs(current_deviation), max_abs_deviation
            )
            reward = global_pressure + bonus + emergency_penalty
            self.last_reward_details["corrective_bonus"] = bonus
        else:
            on_target_reward = self._get_setting_float(
                context, "forced_balance_on_target_reward", 2.0
            )
            reward = global_pressure + on_target_reward + emergency_penalty
            self.last_reward_details["on_target_bonus"] = on_target_reward

        self.last_reward_details["base_reward"] = reward

        if should_log_detailed:
            self.logger.debug(
                "Forced balance decision: action=%s, deviation=%.3f, global_pressure=%.3f, reward=%.3f",
                self.ACTION_INDEX_NAMES[action_idx],
                current_deviation,
                global_pressure,
                reward,
            )

        return reward
