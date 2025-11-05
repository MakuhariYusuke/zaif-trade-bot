"""
Reward Calculator - Handles reward calculation logic for trading environment.

This module separates the complex reward calculation logic from the main environment class.
Refactored to follow SOLID principles with component-based architecture.
"""

import collections
from typing import List, Optional

import numpy as np

from ztb.trading.constants import ACTION_BUY, ACTION_HOLD, ACTION_SELL
from ztb.trading.environment.constants import (
    DEFAULT_ACTION_BALANCE_TARGET,
    DEFAULT_BALANCE_PENALTY_SCALE,
    DEFAULT_REDUNDANT_TRADE_PENALTY,
    EPSILON,
)
from ztb.trading.environment.utils.config import EnvironmentConfig, RewardSettings
from ztb.utils.logging_utils import get_logger

from .reward.action_penalty import ActionPenaltyCalculator
from .reward.base_reward_calculator import BaseRewardCalculator
from .reward.diversity_bonus import DiversityBonusCalculator
from .reward.drawdown_penalty import DrawdownPenaltyCalculator
from .reward.growth_bonus import GrowthBonusCalculator
from .reward.pnl_focused_reward import PnLFocusedRewardCalculator
from .reward.position_penalty import PositionPenaltyCalculator
from .reward.stagnation_penalty import StagnationPenaltyCalculator
from .reward.win_rate_bonus import WinRateBonusCalculator
from .reward.win_streak_bonus import WinStreakBonusCalculator


class RewardCalculator(BaseRewardCalculator):
    """
    Calculates rewards for trading actions with curriculum learning stages.

    This class orchestrates reward calculation using specialized components:
    - MarketRegimeDetector: Detects market regimes
    - DynamicRewardShaper: Applies market-aware reward shaping
    - SignalIntegrator: Integrates signal-based rewards
    - AsymmetricRewardScaler: Applies position-based scaling

    Follows SOLID principles for maintainability and testability.
    """

    def __init__(
        self,
        config: EnvironmentConfig,  # EnvironmentConfig
        reward_settings: RewardSettings,
        initial_portfolio_value: float,
    ):
        """
        Initialize RewardCalculator with component-based architecture.

        Args:
            config: Environment configuration
            reward_settings: Dictionary of reward settings
            initial_portfolio_value: Initial portfolio value
        """
        # Initialize base components
        super().__init__(config, reward_settings, initial_portfolio_value)

        # Initialize reward calculation components
        self.pnl_focused_calculator = PnLFocusedRewardCalculator(
            config, reward_settings, initial_portfolio_value
        )
        self.action_penalty_calculator = ActionPenaltyCalculator()
        self.position_penalty_calculator = PositionPenaltyCalculator()
        self.diversity_bonus_calculator = DiversityBonusCalculator()
        self.win_rate_bonus_calculator = WinRateBonusCalculator()
        self.drawdown_penalty_calculator = DrawdownPenaltyCalculator()
        self.stagnation_penalty_calculator = StagnationPenaltyCalculator()
        self.growth_bonus_calculator = GrowthBonusCalculator()
        self.win_streak_bonus_calculator = WinStreakBonusCalculator()

        # Initialize logger
        self.logger = get_logger(__name__)

        # Initialize base components
        super().__init__(config, reward_settings, initial_portfolio_value)

    def reset(self) -> None:
        """
        Reset the reward calculator state for a new episode.

        This method resets all internal state variables to their initial values.
        """
        # Reset action tracking
        self._action_counts = [0, 0, 0]  # [BUY, SELL, HOLD]
        self._consecutive_idle_steps = 0
        self._consecutive_position_hold_steps = 0
        self._win_count = 0
        self._loss_count = 0
        self._recent_actions = []  # Clear recent actions history
        self.last_signal_strength = 0.0
        self.last_signal_reward = 0.0
        self._previous_portfolio_value = self.initial_portfolio_value

    def _get_behavior_opt(self, key: str, default: float) -> float:
        """
        Get behavior optimization setting with fallback to config and default.

        Args:
            key: Setting key
            default: Default value

        Returns:
            Configured value
        """
        behavior_opts = getattr(self.config, "behavior_optimization", {}) or {}
        return behavior_opts.get(key, getattr(self.config, key, default))

    def calculate_reward(
        self,
        action: int,
        current_price: float,
        position: float,
        portfolio_value: float,
        atr: float,
        transaction_cost: float,
        reward_scaling: float,
        pnl: float,
        old_position: float,
        step: int,
        observation: Optional[np.ndarray],
        reward_history: List[float],
        portfolio_value_history: List[float],
    ) -> float:
        """
        Calculate reward using modular components.

        Args:
            action: Action taken (0=HOLD, 1=BUY, -1=SELL)
            current_price: Current market price
            position: Current position
            portfolio_value: Current portfolio value
            atr: Average True Range
            transaction_cost: Transaction cost
            reward_scaling: Reward scaling factor
            pnl: Profit/Loss from action
            old_position: Position before action
            step: Current step number
            observation: Current observation (for feature access)
            reward_history: History of rewards
            portfolio_value_history: History of portfolio values

        Returns:
            Calculated reward value
        """
        self.logger.debug(
            f"calculate_reward called with action={action}, curriculum_stage={self.config.curriculum_stage}"
        )

        # Update win/loss counts
        if pnl > 0:
            self._win_count += 1
        elif pnl < 0:
            self._loss_count += 1

        # Track recent actions for frequency penalty
        self._recent_actions.append(action)
        if (
            len(self._recent_actions) > 20
        ):  # Increased from 10 to 20 for forced_balance penalty
            self._recent_actions.pop(0)

        # Get curriculum stage from config
        curriculum_stage = getattr(self.config, "curriculum_stage", "simple")
        behavior_opts = getattr(self.config, "behavior_optimization", {}) or {}
        action_bonuses_cfg = getattr(self.config, "action_bonuses", {}) or {}

        # Calculate base reward using appropriate component
        if self.reward_settings.use_simple_reward or curriculum_stage == "pnl_focused":
            if curriculum_stage == "pnl_focused":
                # Calculate required parameters for PnL focused reward
                atr_normalised = atr / (current_price + EPSILON)
                portfolio_return = (portfolio_value - self.initial_portfolio_value) / (
                    self.initial_portfolio_value + EPSILON
                )
                effective_max_position = self.config.max_position_size

                base_reward = self.pnl_focused_calculator.calculate_pnl_focused_reward(
                    action,
                    atr_normalised,
                    portfolio_return,
                    position,
                    effective_max_position,
                    current_price,
                    atr,
                    pnl,
                    reward_scaling,
                    observation,
                    step,
                )
            else:
                # For simple reward, use a simple reward calculation
                base_reward = pnl * reward_scaling
        else:
            # For other stages, use a simple reward calculation
            base_reward = pnl * reward_scaling

        # For simple reward mode, return only base reward without additional components
        if self.reward_settings.use_simple_reward:
            return np.clip(
                base_reward,
                -self.reward_settings.reward_clip_value,
                self.reward_settings.reward_clip_value,
            )

        # Apply forced balance penalty if in forced_balance curriculum stage
        # Support multiple curriculum stage names that enable balance penalty
        balance_penalty = 0.0
        balance_penalty_enabled_stages = (
            "forced_balance",
            "balanced_penalty",
            "balance_optimization",
            "balance_penalty",
        )
        if curriculum_stage in balance_penalty_enabled_stages:
            self.logger.debug(f"Balance penalty stage detected: {curriculum_stage}")
            # Calculate action distribution imbalance
            total_actions = len(self._recent_actions)
            if (
                total_actions >= 10
            ):  # Reduced from 20 to 10 to match action history length
                counter = collections.Counter(self._recent_actions)
                buy_count = counter[ACTION_BUY]
                sell_count = counter[ACTION_SELL]
                hold_count = counter[ACTION_HOLD]

                # Target distribution: roughly 35% each for balance
                target_ratio = self._get_behavior_opt(
                    "action_balance_target", DEFAULT_ACTION_BALANCE_TARGET
                )
                buy_ratio = buy_count / total_actions
                sell_ratio = sell_count / total_actions
                hold_ratio = hold_count / total_actions

                # Penalize actions that deviate from target balanced distribution
                # Use asymmetric targets to favor BUY: BUY=0.4, SELL=0.25, HOLD=0.35
                # This creates asymmetric penalties where SELL-heavy distributions get higher penalties
                balance_penalty_scale = self._get_behavior_opt(
                    "balance_penalty", DEFAULT_BALANCE_PENALTY_SCALE
                )

                # Use asymmetric targets to break symmetry and favor BUY
                # BUY target higher (0.4), SELL lower (0.25), HOLD middle (0.35)
                # This means:
                # ALL_SELL: |0-0.4| + |1-0.25| + |0-0.35| = 0.4 + 0.75 + 0.35 = 1.5 ← HIGH penalty
                # ALL_BUY:  |1-0.4| + |0-0.25| + |0-0.35| = 0.6 + 0.25 + 0.35 = 1.2 ← LOW penalty
                # Balanced: |0.4-0.4| + |0.25-0.25| + |0.35-0.35| = 0.0 ← NO penalty (ideal)
                buy_target = 0.4
                sell_target = 0.25
                hold_target = 0.35

                deviation_buy = abs(buy_ratio - buy_target)
                deviation_sell = abs(sell_ratio - sell_target)
                deviation_hold = abs(hold_ratio - hold_target)

                total_deviation = deviation_buy + deviation_sell + deviation_hold
                balance_penalty = total_deviation * balance_penalty_scale

                # Debug logging
                if (
                    total_actions % 10 == 0
                ):  # Changed from 50 to 10 for more frequent logging
                    self.logger.info(
                        f"BALANCE_PENALTY ({curriculum_stage}): total_actions={total_actions}, buy={buy_ratio:.3f}, sell={sell_ratio:.3f}, hold={hold_ratio:.3f}, targets=[BUY:{buy_target:.3f}, SELL:{sell_target:.3f}, HOLD:{hold_target:.3f}], deviations=[{deviation_buy:.3f}, {deviation_sell:.3f}, {deviation_hold:.3f}], total_dev={total_deviation:.3f}, penalty={balance_penalty:.6f}"
                    )

        redundant_trade_penalty = 0.0
        redundant_trade_cost = self._get_behavior_opt(
            "redundant_trade_penalty", DEFAULT_REDUNDANT_TRADE_PENALTY
        )
        if redundant_trade_cost > 0.0:
            max_position = getattr(self.config, "max_position_size", 1.0)
            if (
                action == ACTION_BUY
                and old_position >= max_position - EPSILON
                and position >= max_position - EPSILON
            ):
                redundant_trade_penalty = redundant_trade_cost
            elif (
                action == ACTION_SELL
                and old_position <= -max_position + EPSILON
                and position <= -max_position + EPSILON
            ):
                redundant_trade_penalty = redundant_trade_cost

        # Apply penalties and bonuses using components
        base_action_penalty = self.get_setting_float(
            "base_action_penalty", getattr(self.config, "base_action_penalty", 0.015)
        )
        buy_action_bonus = self.get_setting_float(
            "action_bonuses.buy_action_bonus",
            action_bonuses_cfg.get("buy_action_bonus", 0.0),
        )
        sell_action_bonus = self.get_setting_float(
            "action_bonuses.sell_action_bonus",
            action_bonuses_cfg.get("sell_action_bonus", 0.0),
        )
        hold_action_bonus = self.get_setting_float(
            "action_bonuses.hold_action_bonus",
            action_bonuses_cfg.get("hold_action_bonus", 0.0),
        )

        action_penalty = self.action_penalty_calculator.calculate(
            action,
            position,
            self.config.max_position_size,
            current_price,
            atr,
            base_action_penalty=base_action_penalty,
            buy_action_bonus=buy_action_bonus,
            sell_action_bonus=sell_action_bonus,
            hold_action_bonus=hold_action_bonus,
        )
        position_penalty = self.position_penalty_calculator.calculate(
            position, self.config.max_position_size
        )
        diversity_bonus = self.diversity_bonus_calculator.calculate(
            self._recent_actions
        )
        win_rate_bonus = self.win_rate_bonus_calculator.calculate(action, pnl)
        drawdown_penalty = self.drawdown_penalty_calculator.calculate(reward_history)
        stagnation_penalty = self.stagnation_penalty_calculator.calculate(
            portfolio_value_history
        )
        growth_bonus = self.growth_bonus_calculator.calculate(portfolio_value_history)
        win_streak_bonus = self.win_streak_bonus_calculator.calculate(reward_history)

        # Combine all components
        self.logger.debug(
            f"Before total_reward calc: base_reward={base_reward:.6f}, balance_penalty={balance_penalty:.6f}, redundant_trade_penalty={redundant_trade_penalty:.6f}, action_penalty={action_penalty:.6f}"
        )
        total_reward = (
            base_reward
            - action_penalty
            - position_penalty
            + diversity_bonus
            + win_rate_bonus
            - drawdown_penalty
            - stagnation_penalty
            + growth_bonus
            + win_streak_bonus
            - balance_penalty
            - redundant_trade_penalty
        )
        self.logger.debug(f"After total_reward calc: total_reward={total_reward:.6f}")

        # Apply asymmetric scaling
        total_reward = self.asymmetric_reward_scaler.scale_reward(
            total_reward, position, pnl
        )

        # Apply signal integration
        total_reward = self.signal_integrator.integrate_signal(
            total_reward, observation, action, step
        )

        # Clip reward to prevent extreme values
        # reward_clip_min = self.reward_settings.reward_clip_min
        # reward_clip_max = self.reward_settings.reward_clip_max
        # total_reward = np.clip(total_reward, reward_clip_min, reward_clip_max)

        self.logger.debug(
            f"final reward={total_reward:.6f}, balance_penalty={balance_penalty:.6f}, redundant_trade_penalty={redundant_trade_penalty:.6f}"
        )
        return total_reward

    def get_current_regime(self, current_price: float, step: int) -> str:
        """Get the current market regime for external diagnostics."""
        if self.market_regime_detector:
            return self.market_regime_detector.detect_regime(current_price, step)
        return "unknown"
