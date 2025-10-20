"""
Reward Stages - Curriculum learning stage reward calculations.

This module contains reward calculation methods for different curriculum learning stages.
"""

from typing import Any, Dict, List, Optional, cast

from ztb.trading.constants import ACTION_BUY, ACTION_HOLD, ACTION_SELL
from ztb.trading.environment.components.reward_components import RewardComponents
from ztb.trading.environment.components.reward_utils import RewardUtils


class RewardStages:
    """Curriculum learning stage reward calculations."""

    def __init__(
        self,
        reward_settings: Optional[Dict[str, Any]] = None,
        action_counts: Optional[List[int]] = None,
    ):
        """Initialize reward stages with settings and action tracking."""
        self.reward_settings = cast(Dict[str, Any], reward_settings or {})
        self._action_counts = action_counts or [0, 0, 0]  # [HOLD, BUY, SELL]
        self.components = RewardComponents(reward_settings)

    def update_action_counts(self, action: int) -> None:
        """Update action counts for balance tracking."""
        self._action_counts[action] += 1

    def get_action_counts(self) -> List[int]:
        """Get current action counts."""
        return self._action_counts.copy()

    def _calculate_forced_balance_reward(self, action: int) -> float:
        """Stage: Forced balance reward that strictly enforces 33/33/33 action distribution."""
        self._action_counts[action] += 1
        total_actions = sum(self._action_counts)

        if total_actions >= 30:  # Wait for some actions to accumulate
            action_ratios = [count / total_actions for count in self._action_counts]
            target_ratio = 1.0 / 3.0  # 33.33% for each action

            # Calculate balance penalty as max deviation from target
            balance_penalty = max(abs(ratio - target_ratio) for ratio in action_ratios)

            # Reward based on balance quality
            if balance_penalty < 0.05:  # Very balanced (within 5%)
                base_reward = 50.0
            elif balance_penalty < 0.1:  # Good balance (within 10%)
                base_reward = 20.0
            elif balance_penalty < 0.15:  # Moderate balance
                base_reward = 5.0
            elif balance_penalty < 0.2:  # Poor balance
                base_reward = 1.0
            else:  # Very poor balance
                base_reward = -10.0

            # Add small bonus for taking actions to encourage exploration
            exploration_bonus = 0.5
            base_reward += exploration_bonus

            return base_reward
        else:
            # Early exploration phase - encourage all actions equally
            return 2.0

    def _calculate_balanced_transition_reward(
        self,
        action: int,
        atr_normalised: float,
        portfolio_return: float,
        position: float,
        effective_max_position: float,
        current_price: float,
        atr: float,
        pnl: float,
        reward_scaling: float,
    ) -> float:
        """Stage 1: Normal reward with balance penalty."""
        self._action_counts[action] += 1
        total_actions = sum(self._action_counts)

        # Get penalty and tolerance from settings
        tolerance = RewardUtils.get_setting_float(
            self.reward_settings, "balance_penalty_tolerance", 0.05
        )
        penalty = RewardUtils.get_setting_float(
            self.reward_settings, "balance_penalty", 4.0
        )
        balance_penalty = 0.0

        if total_actions >= 10:
            action_ratios = [count / total_actions for count in self._action_counts]
            # Adjust target ratios: HOLD 40%, BUY 30%, SELL 30% (encourage trading)
            target_ratios = [0.4, 0.3, 0.3]  # [HOLD, BUY, SELL]

            for i, ratio in enumerate(action_ratios):
                deviation = abs(ratio - target_ratios[i])
                if deviation > tolerance:
                    # Penalty proportional to deviation beyond tolerance
                    excess_deviation = deviation - tolerance
                    balance_penalty += penalty * excess_deviation

        # Calculate base reward
        base_reward = self.components._calculate_base_reward(
            action,
            atr_normalised,
            portfolio_return,
            position,
            effective_max_position,
            current_price,
            atr,
            pnl,
        )

        final_reward = base_reward - balance_penalty
        return final_reward * reward_scaling

    def _calculate_trading_focused_reward(
        self,
        action: int,
        atr_normalised: float,
        portfolio_return: float,
        position: float,
        effective_max_position: float,
        current_price: float,
        atr: float,
        pnl: float,
        reward_scaling: float,
    ) -> float:
        """Stage: Trading-focused reward that heavily penalizes HOLD and encourages trading."""
        self._action_counts[action] += 1
        total_actions = sum(self._action_counts)

        # Extreme HOLD penalty: HOLD 5%, BUY 47.5%, SELL 47.5%
        target_ratios = [0.05, 0.475, 0.475]  # [HOLD, BUY, SELL]
        tolerance = RewardUtils.get_setting_float(
            self.reward_settings, "balance_penalty_tolerance", 0.1
        )
        penalty = RewardUtils.get_setting_float(
            self.reward_settings, "balance_penalty", 10.0
        )
        balance_penalty = 0.0

        if total_actions >= 10:
            action_ratios = [count / total_actions for count in self._action_counts]

            for i, ratio in enumerate(action_ratios):
                deviation = abs(ratio - target_ratios[i])
                if deviation > tolerance:
                    excess_deviation = deviation - tolerance
                    balance_penalty += penalty * excess_deviation

        # Calculate base reward
        base_reward = self.components._calculate_base_reward(
            action,
            atr_normalised,
            portfolio_return,
            position,
            effective_max_position,
            current_price,
            atr,
            pnl,
        )

        # Strong HOLD penalty (but not as extreme as trading_focused)
        hold_penalty_rate = RewardUtils.get_setting_float(
            self.reward_settings, "hold_penalty_rate", 0.05
        )
        if action == ACTION_HOLD:
            hold_penalty = (
                hold_penalty_rate * abs(position) / max(effective_max_position, 0.01)
            )
            base_reward -= hold_penalty

        # Moderate trading bonuses
        trading_bonus_multiplier = RewardUtils.get_setting_float(
            self.reward_settings, "trading_bonus_multiplier", 5.0
        )
        if action in [ACTION_BUY, ACTION_SELL]:
            trading_bonus = (
                RewardUtils.get_setting_float(
                    self.reward_settings, "trading_bonus", 0.02
                )
                * trading_bonus_multiplier
            )
            base_reward += trading_bonus

        final_reward = base_reward - balance_penalty
        return final_reward * reward_scaling

    def _calculate_profit_optimized_reward(
        self,
        action: int,
        atr_normalised: float,
        portfolio_return: float,
        position: float,
        effective_max_position: float,
        current_price: float,
        atr: float,
        pnl: float,
        reward_scaling: float,
    ) -> float:
        """Stage: Profit-optimized reward that maximizes profitable trading while minimizing losses."""
        self._action_counts[action] += 1
        total_actions = sum(self._action_counts)

        # Profit-optimized balance: HOLD 15%, BUY 42.5%, SELL 42.5% (slight edge to trading)
        target_ratios = [0.15, 0.425, 0.425]  # [HOLD, BUY, SELL]
        tolerance = RewardUtils.get_setting_float(
            self.reward_settings, "balance_penalty_tolerance", 0.05
        )
        penalty = RewardUtils.get_setting_float(
            self.reward_settings, "balance_penalty", 6.0
        )
        balance_penalty = 0.0

        if total_actions >= 10:
            action_ratios = [count / total_actions for count in self._action_counts]

            for i, ratio in enumerate(action_ratios):
                deviation = abs(ratio - target_ratios[i])
                if deviation > tolerance:
                    excess_deviation = deviation - tolerance
                    balance_penalty += penalty * excess_deviation

        # Calculate base reward
        base_reward = self.components._calculate_base_reward(
            action,
            atr_normalised,
            portfolio_return,
            position,
            effective_max_position,
            current_price,
            atr,
            pnl,
        )

        # Profit/loss based reward adjustment
        profit_multiplier = RewardUtils.get_setting_float(
            self.reward_settings, "profit_multiplier", 2.0
        )
        loss_penalty_multiplier = RewardUtils.get_setting_float(
            self.reward_settings, "loss_penalty_multiplier", 1.5
        )

        if pnl > 0:
            # Boost profitable trades
            profit_bonus = pnl * profit_multiplier
            base_reward += profit_bonus
        elif pnl < 0:
            # Penalize losing trades more heavily
            loss_penalty = abs(pnl) * loss_penalty_multiplier
            base_reward -= loss_penalty

        # Strong HOLD penalty (but not as extreme as trading_focused)
        hold_penalty_rate = RewardUtils.get_setting_float(
            self.reward_settings, "hold_penalty_rate", 0.02
        )
        if action == ACTION_HOLD:
            hold_penalty = (
                hold_penalty_rate * abs(position) / max(effective_max_position, 0.01)
            )
            base_reward -= hold_penalty

        # Moderate trading bonuses
        trading_bonus_multiplier = RewardUtils.get_setting_float(
            self.reward_settings, "trading_bonus_multiplier", 3.0
        )
        if action in [ACTION_BUY, ACTION_SELL]:
            trading_bonus = (
                RewardUtils.get_setting_float(
                    self.reward_settings, "trading_bonus", 0.01
                )
                * trading_bonus_multiplier
            )
            base_reward += trading_bonus

        final_reward = base_reward - balance_penalty
        return final_reward * reward_scaling

    def _calculate_ultra_profit_reward(
        self,
        action: int,
        atr_normalised: float,
        portfolio_return: float,
        position: float,
        effective_max_position: float,
        current_price: float,
        atr: float,
        pnl: float,
        reward_scaling: float,
    ) -> float:
        """Stage: Ultra-profit reward that COMPLETELY FORCES trading - HOLD is banned."""
        # COMPLETE HOLD BAN: Any HOLD action gets massive negative reward
        if action == ACTION_HOLD:
            return -100.0  # Massive penalty for HOLD

        # Only allow BUY and SELL actions
        if action not in [ACTION_BUY, ACTION_SELL]:
            return -100.0  # Should never happen, but just in case

        # Calculate base reward
        base_reward = self.components._calculate_base_reward(
            action,
            atr_normalised,
            portfolio_return,
            position,
            effective_max_position,
            current_price,
            atr,
            pnl,
        )

        # Massive trading bonus for ANY trading action
        trading_bonus = 10.0  # Even bigger bonus
        base_reward += trading_bonus

        # Position size bonus (reward larger positions)
        position_size_bonus = abs(position) / max(effective_max_position, 0.01) * 1.0
        base_reward += position_size_bonus

        # Minimal PnL weighting - focus on trading frequency over perfect timing
        ultra_profit_multiplier = RewardUtils.get_setting_float(
            self.reward_settings, "ultra_profit_multiplier", 0.1
        )  # Very low
        if pnl > 0:
            profit_bonus = pnl * ultra_profit_multiplier
            base_reward += profit_bonus
        elif pnl < 0:
            loss_penalty = (
                abs(pnl) * ultra_profit_multiplier * 0.05
            )  # Very light penalty
            base_reward -= loss_penalty

        final_reward = base_reward * reward_scaling
        return final_reward
