"""
Reward Calculator - Handles reward calculation logic for trading environment.

This module separates the complex reward calculation logic from the main environment class.
"""

# mypy: disable-error-code=literal-required

import math
from typing import Any, List, Optional

import numpy as np

from ztb.utils.logging_utils import get_logger
from ztb.trading.constants import (
    ACTION_HOLD,
    ACTION_BUY,
    ACTION_SELL,
    MULTIPLIER_INDEX_BUY,
    MULTIPLIER_INDEX_SELL,
    MULTIPLIER_INDEX_HOLD,
)
from ztb.trading.environment.utils.config import RewardSettings


class RewardCalculator:
    """
    Calculates rewards for trading actions with curriculum learning stages.
    
    This class encapsulates all reward-related logic including:
    - Curriculum learning stages (forced_balance, balanced_transition, etc.)
    - Profit bonuses and loss penalties
    - Position penalties
    - Drawdown, stagnation, growth, and win streak calculations
    """

    def __init__(
        self,
        config: Any,  # EnvironmentConfig
        reward_settings: RewardSettings,
        initial_portfolio_value: float,
    ):
        """
        Initialize RewardCalculator.
        
        Args:
            config: Environment configuration
            reward_settings: Dictionary of reward settings
            initial_portfolio_value: Initial portfolio value
        """
        self.config = config
        self.reward_settings = reward_settings
        self.initial_portfolio_value = initial_portfolio_value
        self.logger = get_logger("ztb.trading.environment.reward")
        
        # Internal state for tracking
        self._action_counts: List[int] = [0, 0, 0]  # [HOLD, BUY, SELL]
        self._consecutive_idle_steps = 0
        self._consecutive_position_hold_steps = 0
        self._win_count = 0
        self._loss_count = 0
        self._recent_actions: List[int] = []  # Track recent actions for frequency penalty

    def get_setting_int(self, key: str, default: int) -> int:
        """Get integer reward setting with fallback."""
        if self.reward_settings and key in self.reward_settings:
            value = self.reward_settings[key]
            if isinstance(value, (int, float)):
                return int(value)
        return default

    def get_setting_float(self, key: str, default: float) -> float:
        """Get float reward setting with fallback."""
        if self.reward_settings and key in self.reward_settings:
            value = self.reward_settings[key]
            if isinstance(value, (int, float)):
                return float(value)
        return default

    def get_setting_bool(self, key: str, default: bool) -> bool:
        """Get boolean reward setting with fallback."""
        if self.reward_settings and key in self.reward_settings:
            value = self.reward_settings[key]
            if isinstance(value, bool):
                return value
            if isinstance(value, (int, float)):
                return bool(value)
            if isinstance(value, str):
                return value.lower() in {"true", "1", "yes", "y", "on"}
        return default

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
        observation: Optional[Any],
        reward_history: List[float],
        portfolio_value_history: List[float],
    ) -> float:
        """
        Calculate reward with curriculum learning stages.
        
        Args:
            action: Action taken (0=HOLD, 1=BUY, 2=SELL)
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
        # Debug logging for reward calculation inputs
        self.logger.debug(
            f"Reward calc inputs: action={action}, pnl={pnl:.2f}, position={position:.4f}, "
            f"portfolio_value={portfolio_value:.2f}, atr={atr:.2f}, current_price={current_price:.2f}, "
            f"old_position={old_position:.4f}, step={step}"
        )
        
        # Update win/loss counts
        if pnl > 0:
            self._win_count += 1
        elif pnl < 0:
            self._loss_count += 1
            
        # Track recent actions for frequency penalty
        self._recent_actions.append(action)
        if len(self._recent_actions) > 10:  # Keep last 10 actions
            self._recent_actions.pop(0)
        
        # Check if simple reward is enabled
        use_simple_reward = self.get_setting_bool("use_simple_reward", False)
        
        if use_simple_reward:
            return self.calculate_reward_simple(
                pnl, portfolio_value, position, old_position, action, reward_history, portfolio_value_history
            )
        
        # Original complex reward function
        curriculum_stage = self.config.curriculum_stage
        self.logger.info(
            "Curriculum stage: %s, position: %.2f, action: %d",
            curriculum_stage, position, action
        )

        eps = self.get_setting_float("eps", 1e-8)
        atr = atr if atr > eps else 1.0
        max_position_size = max(eps, self.config.max_position_size)
        
        # Calculate effective max position considering capital constraints
        effective_max_position = min(max_position_size, self.initial_portfolio_value / max(current_price, eps))
        self.logger.debug(f"Position calculations: max_position_size={max_position_size:.4f}, effective_max_position={effective_max_position:.4f}, initial_portfolio_value={self.initial_portfolio_value:.2f}, current_price={current_price:.2f}")
        
        # Adapt reward scaling based on max position size to prevent clipping
        scale_adjustment_base = self.get_setting_float("scale_adjustment_base", 1.0)
        scale_adjustment = scale_adjustment_base / max(0.01, max_position_size)
        reward_scaling = reward_scaling * scale_adjustment

        atr_normalised = pnl / atr
        portfolio_return = pnl / max(abs(self.initial_portfolio_value), eps)

        # Curriculum learning stages
        if curriculum_stage == "forced_balance":
            reward = self._calculate_forced_balance_reward(action)
            # Use smaller scaling for forced balance to prevent extreme values
            forced_balance_scaling = self.get_setting_float("forced_balance_scaling", 1.0)
            reward *= forced_balance_scaling
            # Apply clipping
            reward_clip_min = self.get_setting_float("reward_clip_min", -80.0)
            reward_clip_max = self.get_setting_float("reward_clip_max", 80.0)
            reward = np.clip(reward, reward_clip_min, reward_clip_max)
            self.logger.info(f"Forced balance reward: {reward}, action_counts: {self._action_counts}")
            self.logger.debug(f"Final reward: {reward}")
            return reward
        elif curriculum_stage == "balanced_transition":
            reward = self._calculate_balanced_transition_reward(
                action, atr_normalised, portfolio_return, position,
                effective_max_position, current_price, atr, pnl, reward_scaling
            )
            # Apply clipping
            reward_clip_min = self.get_setting_float("reward_clip_min", -80.0)
            reward_clip_max = self.get_setting_float("reward_clip_max", 80.0)
            reward = np.clip(reward, reward_clip_min, reward_clip_max)
            return reward
        elif curriculum_stage == "trading_focused":
            reward = self._calculate_trading_focused_reward(
                action, atr_normalised, portfolio_return, position,
                effective_max_position, current_price, atr, pnl, reward_scaling
            )
            # Apply clipping
            reward_clip_min = self.get_setting_float("reward_clip_min", -80.0)
            reward_clip_max = self.get_setting_float("reward_clip_max", 80.0)
            reward = np.clip(reward, reward_clip_min, reward_clip_max)
            return reward
        elif curriculum_stage == "profit_optimized":
            reward = self._calculate_profit_optimized_reward(
                action, atr_normalised, portfolio_return, position,
                effective_max_position, current_price, atr, pnl, reward_scaling, observation
            )
            # Apply clipping
            reward_clip_min = self.get_setting_float("reward_clip_min", -80.0)
            reward_clip_max = self.get_setting_float("reward_clip_max", 80.0)
            reward = np.clip(reward, reward_clip_min, reward_clip_max)
            return reward
        elif curriculum_stage == "ultra_profit":
            reward = self._calculate_ultra_profit_reward(
                action, atr_normalised, portfolio_return, position,
                effective_max_position, current_price, atr, pnl, reward_scaling
            )
            # Apply clipping
            reward_clip_min = self.get_setting_float("reward_clip_min", -80.0)
            reward_clip_max = self.get_setting_float("reward_clip_max", 80.0)
            reward = np.clip(reward, reward_clip_min, reward_clip_max)
            return reward
        elif curriculum_stage == "pnl_focused":
            reward = self._calculate_pnl_focused_reward(
                action, atr_normalised, portfolio_return, position,
                effective_max_position, current_price, atr, pnl, observation
            )
            reward *= reward_scaling
            # Apply clipping
            reward_clip_min = self.get_setting_float("reward_clip_min", -80.0)
            reward_clip_max = self.get_setting_float("reward_clip_max", 80.0)
            reward = np.clip(reward, reward_clip_min, reward_clip_max)
            return reward
        else:
            # Default stage
            reward = self._calculate_default_reward(
                action, atr_normalised, portfolio_return, position,
                effective_max_position, current_price, atr, pnl
            )
            reward *= reward_scaling
            # Apply clipping
            reward_clip_min = self.get_setting_float("reward_clip_min", -80.0)
            reward_clip_max = self.get_setting_float("reward_clip_max", 80.0)
            reward = np.clip(reward, reward_clip_min, reward_clip_max)
            return reward

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
            
            self.logger.debug(f"Forced balance: ratios={action_ratios}, penalty={balance_penalty:.3f}, reward={base_reward:.3f}")
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
        tolerance = self.get_setting_float("balance_penalty_tolerance", 0.05)
        penalty = self.get_setting_float("balance_penalty", 4.0)
        self.logger.info(f"Penalty settings: penalty={penalty}, tolerance={tolerance}")
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
                    self.logger.info(f"Balance penalty applied: {balance_penalty:.3f}, ratios: {action_ratios}, targets: {target_ratios}")
        
        # Calculate base reward
        base_reward = self._calculate_base_reward(
            action, atr_normalised, portfolio_return, position,
            effective_max_position, current_price, atr, pnl
        )
        
        final_reward = base_reward - balance_penalty
        self.logger.info(f"Balanced transition: base_reward={base_reward:.3f}, balance_penalty={balance_penalty:.3f}, final_reward={final_reward:.3f}, action_counts={self._action_counts}")
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

        # Trading-focused balance penalty: HOLD 10%, BUY 45%, SELL 45%
        target_ratios = [0.1, 0.45, 0.45]  # [HOLD, BUY, SELL] - minimize HOLD
        tolerance = self.get_setting_float("balance_penalty_tolerance", 0.05)
        penalty = self.get_setting_float("balance_penalty", 8.0)  # Higher penalty for trading focus
        balance_penalty = 0.0

        if total_actions >= 10:
            action_ratios = [count / total_actions for count in self._action_counts]

            for i, ratio in enumerate(action_ratios):
                deviation = abs(ratio - target_ratios[i])
                if deviation > tolerance:
                    # Penalty proportional to deviation beyond tolerance
                    excess_deviation = deviation - tolerance
                    balance_penalty += penalty * excess_deviation

            self.logger.info(f"Trading focused penalty applied: {balance_penalty:.3f}, ratios: {action_ratios}, targets: {target_ratios}")

        # Calculate base reward
        base_reward = self._calculate_base_reward(
            action, atr_normalised, portfolio_return, position,
            effective_max_position, current_price, atr, pnl
        )

        # Add strong HOLD penalty
        hold_penalty_rate = self.get_setting_float("hold_penalty_rate", 0.01)
        if action == ACTION_HOLD:
            # Strong penalty for HOLD action
            hold_penalty = hold_penalty_rate * abs(position) / max(effective_max_position, 0.01)
            base_reward -= hold_penalty
            self.logger.debug(f"HOLD penalty applied: {hold_penalty:.3f}")

        # Boost trading bonuses
        trading_bonus_multiplier = self.get_setting_float("trading_bonus_multiplier", 2.0)
        if action in [ACTION_BUY, ACTION_SELL]:
            trading_bonus = self.get_setting_float("trading_bonus", 0.01) * trading_bonus_multiplier
            base_reward += trading_bonus
            self.logger.debug(f"Trading bonus applied: {trading_bonus:.3f}")

        final_reward = base_reward - balance_penalty
        self.logger.info(f"Trading focused: base_reward={base_reward:.3f}, balance_penalty={balance_penalty:.3f}, final_reward={final_reward:.3f}, action_counts={self._action_counts}")
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
        observation: np.ndarray = None,
    ) -> float:
        """Stage: Profit-optimized reward that maximizes profitable trading while minimizing losses."""
        self._action_counts[action] += 1
        total_actions = sum(self._action_counts)

        # Profit-optimized balance: HOLD 15%, BUY 42.5%, SELL 42.5% (slight edge to trading)
        target_ratios = [0.15, 0.425, 0.425]  # [HOLD, BUY, SELL]
        tolerance = self.get_setting_float("balance_penalty_tolerance", 0.05)
        penalty = self.get_setting_float("balance_penalty", 6.0)
        balance_penalty = 0.0

        if total_actions >= 10:
            action_ratios = [count / total_actions for count in self._action_counts]

            for i, ratio in enumerate(action_ratios):
                deviation = abs(ratio - target_ratios[i])
                if deviation > tolerance:
                    excess_deviation = deviation - tolerance
                    balance_penalty += penalty * excess_deviation

        # Calculate base reward
        base_reward = self._calculate_base_reward(
            action, atr_normalised, portfolio_return, position,
            effective_max_position, current_price, atr, pnl, observation
        )

        # Profit/loss based reward adjustment
        profit_multiplier = self.get_setting_float("profit_multiplier", 2.0)
        loss_penalty_multiplier = self.get_setting_float("loss_penalty_multiplier", 1.5)

        if pnl > 0:
            # Boost profitable trades
            profit_bonus = pnl * profit_multiplier
            base_reward += profit_bonus
            self.logger.debug(f"Profit bonus applied: {profit_bonus:.3f} for pnl={pnl:.3f}")
        elif pnl < 0:
            # Penalize losing trades more heavily
            loss_penalty = abs(pnl) * loss_penalty_multiplier
            base_reward -= loss_penalty
            self.logger.debug(f"Loss penalty applied: {loss_penalty:.3f} for pnl={pnl:.3f}")

        # Strong HOLD penalty (but not as extreme as trading_focused)
        hold_penalty_rate = self.get_setting_float("hold_penalty_rate", 0.02)
        if action == ACTION_HOLD:
            hold_penalty = hold_penalty_rate * abs(position) / max(effective_max_position, 0.01)
            base_reward -= hold_penalty
            self.logger.debug(f"HOLD penalty applied: {hold_penalty:.3f}")

        # Moderate trading bonuses
        trading_bonus_multiplier = self.get_setting_float("trading_bonus_multiplier", 3.0)
        if action in [ACTION_BUY, ACTION_SELL]:
            trading_bonus = self.get_setting_float("trading_bonus", 0.01) * trading_bonus_multiplier
            base_reward += trading_bonus
            self.logger.debug(f"Trading bonus applied: {trading_bonus:.3f}")

        final_reward = base_reward - balance_penalty

        final_reward = base_reward - balance_penalty
        self.logger.info(f"Profit optimized: base_reward={base_reward:.3f}, balance_penalty={balance_penalty:.3f}, pnl={pnl:.3f}, final_reward={final_reward:.3f}")
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
        """Stage: Ultra-profit reward that maximizes profitability with balanced trading.
        
        Modified to allow necessary HOLD actions and ensure balanced BUY/SELL rewards.
        """
        self._action_counts[action] += 1
        total_actions = sum(self._action_counts)

        # Balanced target: HOLD 10%, BUY 45%, SELL 45% (allow necessary HOLD)
        target_ratios = [0.10, 0.45, 0.45]  # [HOLD, BUY, SELL]
        tolerance = self.get_setting_float("balance_penalty_tolerance", 0.15)  # More lenient
        penalty = self.get_setting_float("balance_penalty", 1.0)  # Lower penalty
        balance_penalty = 0.0

        if total_actions >= 10:
            action_ratios = [count / total_actions for count in self._action_counts]

            deviation = abs(action_ratios[action] - target_ratios[action])
            if deviation > tolerance:
                excess_deviation = deviation - tolerance
                balance_penalty = penalty * excess_deviation

        # Adjust pnl based on action for balanced BUY/SELL rewards
        # BUY action: pnl remains as is (predicting price up)
        # SELL action: pnl remains as is (predicting price down)
        adjusted_pnl = pnl

        # Calculate base reward
        base_reward = self._calculate_base_reward(
            action, atr_normalised, portfolio_return, position,
            effective_max_position, current_price, atr, adjusted_pnl
        )

        # Balanced profit/loss adjustments (equal treatment for BUY/SELL)
        profit_multiplier = self.get_setting_float("profit_multiplier", 3.0)
        loss_penalty_multiplier = self.get_setting_float("loss_penalty_multiplier", 3.0)  # Equal penalty

        if adjusted_pnl > 0:
            # Equal boost for profitable trades regardless of BUY/SELL
            profit_bonus = adjusted_pnl * profit_multiplier
            base_reward += profit_bonus
            self.logger.debug(f"Balanced profit bonus applied: {profit_bonus:.3f} for adjusted_pnl={adjusted_pnl:.3f}")
        elif adjusted_pnl < 0:
            # Equal penalty for losing trades regardless of BUY/SELL
            loss_penalty = abs(adjusted_pnl) * loss_penalty_multiplier
            base_reward -= loss_penalty
            self.logger.debug(f"Balanced loss penalty applied: {loss_penalty:.3f} for adjusted_pnl={adjusted_pnl:.3f}")

        # Moderate HOLD penalty (allow necessary HOLD for risk management)
        hold_penalty_rate = self.get_setting_float("hold_penalty_rate", 0.01)  # Reduced penalty
        if action == ACTION_HOLD:
            # Reduce penalty if position is large (risk management HOLD)
            position_size_factor = min(1.0, abs(position) / max(effective_max_position, 0.01))
            hold_penalty = hold_penalty_rate * position_size_factor
            base_reward -= hold_penalty
            self.logger.debug(f"Moderate HOLD penalty applied: {hold_penalty:.3f} (position_factor: {position_size_factor:.3f})")

        # Equal trading bonuses for BUY and SELL
        trading_bonus_multiplier = self.get_setting_float("trading_bonus_multiplier", 2.0)
        if action in [ACTION_BUY, ACTION_SELL]:
            trading_bonus = self.get_setting_float("trading_bonus", 0.01) * trading_bonus_multiplier
            base_reward += trading_bonus
            self.logger.debug(f"Equal trading bonus applied: {trading_bonus:.3f} for action {action}")

        final_reward = base_reward - balance_penalty
        self.logger.info(f"Balanced ultra profit: base_reward={base_reward:.3f}, balance_penalty={balance_penalty:.3f}, adjusted_pnl={adjusted_pnl:.3f}, final_reward={final_reward:.3f}")
        return final_reward * reward_scaling

    def _calculate_pnl_focused_reward(
        self,
        action: int,
        atr_normalised: float,
        portfolio_return: float,
        position: float,
        effective_max_position: float,
        current_price: float,
        atr: float,
        pnl: float,
        observation: Optional[Any]
    ) -> float:
        """Stage 2: PnL-focused reward with trend analysis."""
        base_profit_bonus = max(0.0, self.get_setting_float("base_profit_bonus_atr_coeff", 1.5) * atr_normalised + self.get_setting_float("base_profit_bonus_portfolio_coeff", 1.2) * portfolio_return) if pnl > 0 else 0.0

        # Trend analysis if observation available
        trend_multiplier = 1.0
        if observation is not None and hasattr(observation, '__getitem__'):
            try:
                # Assume RSI and MACD are in observation
                rsi_idx = -2 if len(observation) > 2 else None
                macd_idx = -1 if len(observation) > 1 else None
                
                if rsi_idx is not None and macd_idx is not None:
                    rsi = float(observation[rsi_idx])
                    macd_hist = float(observation[macd_idx])
                    
                    # Trend ratio
                    trend_ratio = (rsi / 50.0) * (1.0 + macd_hist / 100.0)
                    
                    if trend_ratio > 1.0 and action == ACTION_BUY:
                        trend_multiplier = 1.2
                    elif trend_ratio < 1.0 and action == ACTION_SELL:
                        trend_multiplier = 1.2
                    
                    # Oversold/Overbought signals
                    if rsi > 60.0 and trend_ratio > 1.0 and action == ACTION_SELL:
                        trend_multiplier *= 1.3
                    elif rsi < 40.0 and trend_ratio < 1.0 and action == ACTION_SELL:
                        trend_multiplier *= 1.3
            except (IndexError, TypeError, ValueError):
                pass

        # Balance BUY/SELL actions
        # profit_bonus_multipliers array order: [BUY, SELL, HOLD]
        multipliers_raw = self.reward_settings.get("profit_bonus_multipliers", [1.0, 1.0, 0.8])
        if len(multipliers_raw) >= 3:
            multipliers = [float(x) for x in multipliers_raw[:3]]
        else:
            multipliers = [1.0, 1.0, 0.8]
        
        if action == ACTION_BUY:
            profit_bonus = base_profit_bonus * multipliers[MULTIPLIER_INDEX_BUY] * trend_multiplier
        elif action == ACTION_SELL:
            profit_bonus = base_profit_bonus * multipliers[MULTIPLIER_INDEX_SELL] * trend_multiplier
        else:  # HOLD
            profit_bonus = base_profit_bonus * multipliers[MULTIPLIER_INDEX_HOLD] * trend_multiplier

        # Action penalties with configurable per-action penalties
        # Get action-specific penalties from settings (default: 0.0)
        hold_penalty = self.get_setting_float("hold_action_penalty", 0.0)
        buy_penalty = self.get_setting_float("buy_action_penalty", 0.0)
        sell_penalty = self.get_setting_float("sell_action_penalty", 0.0)
        
        # Base action penalties (legacy behavior)
        base_action_penalty = self.get_setting_float("base_action_penalty", 0.015) if action in [ACTION_BUY, ACTION_SELL] else 0.0
        
        if action == ACTION_HOLD:
            position_size_factor = abs(position) / effective_max_position
            volatility_factor = min(atr / (current_price * 0.01), 1.0)
            base_action_penalty = self.get_setting_float("hold_penalty_base", 0.01) + (self.get_setting_float("hold_penalty_position_factor", 0.04) * position_size_factor * volatility_factor)
            # Add configured HOLD penalty
            action_penalty = base_action_penalty + hold_penalty
        elif action == ACTION_BUY:
            # Add configured BUY penalty (negative value = reward)
            action_penalty = base_action_penalty + buy_penalty
        else:  # ACTION_SELL
            # Add configured SELL penalty (negative value = reward)
            action_penalty = base_action_penalty + sell_penalty

        # Loss penalty
        loss_penalty = self.get_setting_float("loss_penalty_coeff", -0.2) * abs(atr_normalised) if pnl < 0 else 0.0

        # Position penalty
        position_penalty = self._calculate_position_penalty(position, effective_max_position)

        reward = profit_bonus - action_penalty + loss_penalty - position_penalty
        self.logger.debug(f"PnL focused reward components: profit_bonus={profit_bonus:.4f}, action_penalty={action_penalty:.4f}, loss_penalty={loss_penalty:.4f}, position_penalty={position_penalty:.4f}, final={reward:.4f}")
        return reward

    def _calculate_default_reward(
        self,
        action: int,
        atr_normalised: float,
        portfolio_return: float,
        position: float,
        effective_max_position: float,
        current_price: float,
        atr: float,
        pnl: float,
        observation: np.ndarray = None,
    ) -> float:
        """Default reward calculation."""
        base_profit_bonus = max(0.0, self.get_setting_float("base_profit_bonus_atr_coeff", 1.5) * atr_normalised + self.get_setting_float("base_profit_bonus_portfolio_coeff", 1.2) * portfolio_return) if pnl > 0 else 0.0

        # profit_bonus_multipliers array order: [BUY, SELL, HOLD]
        multipliers = self.reward_settings.get("profit_bonus_multipliers", [1.0, 1.0, 0.8])
        
        # Get action-specific penalties from settings (default: 0.0)
        hold_penalty = self.get_setting_float("hold_action_penalty", 0.0)
        buy_penalty = self.get_setting_float("buy_action_penalty", 0.0)
        sell_penalty = self.get_setting_float("sell_action_penalty", 0.0)
        
        if action == ACTION_BUY:
            profit_bonus = base_profit_bonus * multipliers[MULTIPLIER_INDEX_BUY]
            action_penalty = 0.015 + buy_penalty
        elif action == ACTION_SELL:
            profit_bonus = base_profit_bonus * multipliers[MULTIPLIER_INDEX_SELL]
            action_penalty = 0.015 + sell_penalty
        else:  # HOLD
            profit_bonus = base_profit_bonus * multipliers[MULTIPLIER_INDEX_HOLD]
            position_size_factor = abs(position) / effective_max_position
            volatility_factor = min(atr / (current_price * 0.01), 1.0)
            action_penalty = 0.01 + (0.04 * position_size_factor * volatility_factor) + hold_penalty

        loss_penalty = -0.2 * abs(atr_normalised) if pnl < 0 else 0.0
        
        # Forced diversity bonus
        diversity_bonus = 0.0
        if self.get_setting_bool("enable_forced_diversity", False):
            diversity_bonus = self._calculate_diversity_bonus(action)
        
        # Win rate bonus
        total_trades = self._win_count + self._loss_count
        win_rate = self._win_count / max(total_trades, 1)
        win_rate_bonus = self.get_setting_float("win_rate_bonus", 0.0) * win_rate
        
        # Momentum bonus (based on RSI and MACD from observation)
        momentum_bonus = 0.0
        if observation is not None and len(observation) >= 2:
            try:
                rsi_idx = -2 if len(observation) > 2 else None
                macd_idx = -1 if len(observation) > 1 else None
                if rsi_idx is not None and macd_idx is not None:
                    rsi = float(observation[rsi_idx])
                    macd_hist = float(observation[macd_idx])
                    momentum_multiplier = self.get_setting_float("momentum_bonus", 0.0)
                    if action == ACTION_BUY and rsi > 50 and macd_hist > 0:
                        momentum_bonus = momentum_multiplier
                    elif action == ACTION_SELL and rsi < 50 and macd_hist < 0:
                        momentum_bonus = momentum_multiplier
            except (IndexError, TypeError, ValueError):
                pass
        
        # Volatility penalty
        volatility_penalty = self.get_setting_float("volatility_penalty", 0.0) * (atr / current_price)
        
        # Action frequency penalty
        action_frequency_penalty = 0.0
        if len(self._recent_actions) >= 5:
            recent_action_count = sum(1 for a in self._recent_actions[-5:] if a != ACTION_HOLD)
            frequency_penalty_rate = self.get_setting_float("action_frequency_penalty", 0.0)
            action_frequency_penalty = frequency_penalty_rate * (recent_action_count / 5.0)
        
        # Diversity bonus
        diversity_bonus_value = self.get_setting_float("diversity_bonus", 0.0)
        unique_actions = len(set(self._recent_actions))
        diversity_bonus = diversity_bonus_value * (unique_actions / 3.0)  # Normalize by total action types
        
        # Trading bonus
        trading_bonus = 0.0
        if action in [ACTION_BUY, ACTION_SELL]:
            trading_bonus = self.get_setting_float("trading_bonus", 0.01)

        position_penalty = self._calculate_position_penalty(position, effective_max_position)

        reward = (profit_bonus - action_penalty + loss_penalty + diversity_bonus + win_rate_bonus + 
                 momentum_bonus - volatility_penalty - action_frequency_penalty + trading_bonus - position_penalty)
        self.logger.debug(f"Comprehensive reward components: profit_bonus={profit_bonus:.4f}, action_penalty={action_penalty:.4f}, loss_penalty={loss_penalty:.4f}, diversity_bonus={diversity_bonus:.4f}, win_rate_bonus={win_rate_bonus:.4f}, momentum_bonus={momentum_bonus:.4f}, volatility_penalty={volatility_penalty:.4f}, action_frequency_penalty={action_frequency_penalty:.4f}, trading_bonus={trading_bonus:.4f}, position_penalty={position_penalty:.4f}, final={reward:.4f}")
        return reward

    def _calculate_base_reward(
        self,
        action: int,
        atr_normalised: float,
        portfolio_return: float,
        position: float,
        effective_max_position: float,
        current_price: float,
        atr: float,
        pnl: float,
        observation: np.ndarray = None,
    ) -> float:
        """Calculate base reward components."""
        return self._calculate_default_reward(
            action, atr_normalised, portfolio_return, position,
            effective_max_position, current_price, atr, pnl, observation
        )

    def _calculate_position_penalty(self, position: float, effective_max_position: float) -> float:
        """Calculate penalty for excessive position usage."""
        position_utilisation = abs(position) / effective_max_position
        soft_cap = self.get_setting_float("position_soft_cap", 0.8)
        
        if position_utilisation > soft_cap:
            overuse = position_utilisation - soft_cap
            scale = self.get_setting_float("position_penalty_scale", 0.5)
            exponent = self.get_setting_float("position_penalty_exp", 2.0)
            return scale * (math.exp(exponent * overuse) - 1.0)
        
        return 0.0

    def _calculate_diversity_bonus(self, action: int) -> float:
        """Calculate bonus for action diversity."""
        if len(self._recent_actions) < 3:
            return 0.1  # Small bonus for early exploration
        
        unique_recent = len(set(self._recent_actions[-10:]))  # Last 10 actions
        diversity_score = unique_recent / 3.0  # Normalize by action types
        
        # Bonus for maintaining diversity
        base_bonus = 0.05
        diversity_multiplier = diversity_score ** 2  # Quadratic scaling
        
        return base_bonus * diversity_multiplier

    def _calculate_diversity_bonus(self, action: int) -> float:
        """Calculate bonus for action diversity."""
        self._action_counts[action] += 1
        total_actions = sum(self._action_counts)
        
        if total_actions < 5:
            return 0.0

        action_ratios = [count / total_actions for count in self._action_counts]
        min_required_ratio = self.get_setting_float("diversity_min_required_ratio", 0.1)

        unused_penalty = sum(1.0 for count in self._action_counts if count == 0)
        
        underused_penalty = sum(
            max(0.0, min_required_ratio - ratio) * 2.0
            for ratio in action_ratios
        )

        return -(unused_penalty + underused_penalty)

    def calculate_drawdown_penalty(self, reward_history: List[float]) -> float:
        """Calculate drawdown penalty (when drawdown exceeds 50%)."""
        drawdown_window = self.get_setting_int("drawdown_window", 20)
        if len(reward_history) < drawdown_window:
            return 0.0

        recent_rewards = reward_history[-drawdown_window:]
        cumulative_reward = sum(recent_rewards)

        stagnation_window = self.get_setting_int("stagnation_window", 30)
        if len(reward_history) >= stagnation_window:
            initial_window = stagnation_window - drawdown_window
            initial_rewards = reward_history[-stagnation_window:-initial_window]
            initial_cumulative = sum(initial_rewards)

            if initial_cumulative > 0:
                drawdown_ratio = (initial_cumulative - cumulative_reward) / initial_cumulative
                drawdown_threshold = 0.5
                if drawdown_ratio > drawdown_threshold:
                    drawdown_penalty_coeff = self.get_setting_float("drawdown_penalty_coeff", 0.05)
                    return drawdown_ratio * drawdown_penalty_coeff

        return 0.0

    def calculate_stagnation_penalty(self, portfolio_value_history: List[float]) -> float:
        """Calculate stagnation penalty (when portfolio isn't growing)."""
        stagnation_window = self.get_setting_int("stagnation_window", 30)
        if len(portfolio_value_history) < stagnation_window:
            return 0.0

        recent_values = portfolio_value_history[-stagnation_window:]
        initial_value = recent_values[0]
        final_value = recent_values[-1]

        if initial_value > 0:
            growth_rate = (final_value - initial_value) / initial_value
            stagnation_threshold = self.get_setting_float("stagnation_threshold", -0.005)

            if growth_rate < stagnation_threshold:
                stagnation_penalty_max = self.get_setting_float("stagnation_penalty_max", 0.02)
                return min(stagnation_penalty_max, abs(growth_rate - stagnation_threshold) * 0.5)

        return 0.0

    def calculate_growth_bonus(self, portfolio_value_history: List[float]) -> float:
        """Calculate growth bonus (when portfolio is increasing)."""
        stagnation_window = self.get_setting_int("stagnation_window", 30)
        if len(portfolio_value_history) < stagnation_window:
            return 0.0

        recent_values = portfolio_value_history[-stagnation_window:]
        initial_value = recent_values[0]
        final_value = recent_values[-1]

        if initial_value > 0:
            growth_rate = (final_value - initial_value) / initial_value
            growth_threshold = self.get_setting_float("growth_threshold", 0.005)

            if growth_rate > growth_threshold:
                growth_bonus_max = self.get_setting_float("growth_bonus_max", 0.05)
                return min(growth_bonus_max, growth_rate * 0.5)

        return 0.0

    def calculate_win_streak_bonus(self, reward_history: List[float]) -> float:
        """Calculate win streak bonus."""
        win_streak_window = self.get_setting_int("win_streak_window", 5)
        if len(reward_history) < win_streak_window:
            return 0.0

        recent_rewards = reward_history[-win_streak_window:]
        win_count = sum(1 for r in recent_rewards if r > 0)

        win_streak_min = 3
        if win_count >= win_streak_min:
            win_streak_bonus_per_win = self.get_setting_float("win_streak_bonus_per_win", 0.01)
            return win_count * win_streak_bonus_per_win

        return 0.0

    def reset(self) -> None:
        """Reset internal state."""
        self._action_counts = [0, 0, 0]
        self._consecutive_idle_steps = 0
        self._consecutive_position_hold_steps = 0

    def _convert_continuous_to_discrete_action(self, action: Any) -> int:
        """
        Convert continuous action from SAC to discrete action.
        
        Args:
            action: Continuous action value from SAC (typically in [-1, 1])
            
        Returns:
            Discrete action (0=HOLD, 1=BUY, 2=SELL)
        """
        if isinstance(action, (int, np.integer)):
            # Already discrete
            return int(action)
        else:
            # Convert continuous action to discrete
            # SAC outputs continuous values in [-1, 1], map to discrete actions
            action_threshold_buy = self.get_setting_float("action_threshold_buy", 0.2)
            action_threshold_sell = self.get_setting_float("action_threshold_sell", -0.2)
            
            if action < action_threshold_sell:
                discrete_action = ACTION_SELL  # Strong sell signal
            elif action > action_threshold_buy:
                discrete_action = ACTION_BUY   # Strong buy signal
            else:
                discrete_action = ACTION_HOLD  # Hold/weak signal
            
            self.logger.debug(f"Continuous action {action:.3f} converted to discrete action {discrete_action}")
            return discrete_action

    def _calculate_win_rate_bonus(self, discrete_action: int, pnl: float) -> float:
        """
        Calculate win rate bonus based on recent trading performance.
        
        Uses continuous scaling instead of discrete thresholds for smoother learning.
        
        Args:
            discrete_action: Discrete action (0=HOLD, 1=BUY, 2=SELL)
            pnl: Profit/Loss from the action
            
        Returns:
            Win rate bonus value (continuous scaling)
        """
        win_rate_bonus = 0.0
        if discrete_action in [ACTION_BUY, ACTION_SELL] and pnl != 0:
            # Track trade outcomes for win rate calculation
            if not hasattr(self, "_trade_outcomes"):
                self._trade_outcomes = []

            # Record trade outcome (1 for win, 0 for loss)
            trade_outcome = 1 if pnl > 0 else 0
            self._trade_outcomes.append(trade_outcome)

            # Keep only recent trades (rolling window)
            max_window = self.get_setting_int("win_rate_window", 50)
            if len(self._trade_outcomes) > max_window:
                self._trade_outcomes = self._trade_outcomes[-max_window:]

            # Calculate current win rate when we have enough data
            min_trades = self.get_setting_int("win_rate_min_trades", 5)
            if len(self._trade_outcomes) >= min_trades:
                current_win_rate = sum(self._trade_outcomes) / len(self._trade_outcomes)

                baseline = self.get_setting_float("win_rate_baseline", 0.5)
                scaling = self.get_setting_float("win_rate_bonus_scale", 100.0)
                win_rate_bonus = (current_win_rate - baseline) * scaling

                # Optional clipping to keep reward bounded
                clip_min = self.get_setting_float("win_rate_bonus_min", -50.0)
                clip_max = self.get_setting_float("win_rate_bonus_max", 50.0)
                win_rate_bonus = float(np.clip(win_rate_bonus, clip_min, clip_max))

                self.logger.debug(
                    "Continuous win rate bonus: %.2f (win_rate=%.3f, baseline=%.3f, scale=%.1f)",
                    win_rate_bonus,
                    current_win_rate,
                    baseline,
                    scaling,
                )

        return win_rate_bonus

    def _calculate_action_balance_bonus(self, discrete_action: int) -> float:
        """
        Calculate action balance bonus to encourage balanced BUY/SELL usage.
        
        Args:
            discrete_action: Discrete action (0=HOLD, 1=BUY, 2=SELL)
            
        Returns:
            Action balance bonus value
        """
        action_balance_bonus = 0.0
        if discrete_action in [ACTION_BUY, ACTION_SELL]:
            # Track action counts in reward calculator
            self._action_counts[discrete_action] += 1
            total_recent_actions = sum(self._action_counts)
            
            if total_recent_actions >= 10:
                buy_ratio = self._action_counts[ACTION_BUY] / total_recent_actions
                sell_ratio = self._action_counts[ACTION_SELL] / total_recent_actions
                
                # Bonus for maintaining balance (ratios between 0.3 and 0.7)
                if 0.3 <= buy_ratio <= 0.7 and 0.3 <= sell_ratio <= 0.7:
                    action_balance_bonus = 3.0
                elif abs(buy_ratio - sell_ratio) <= 0.2:  # Difference within 20%
                    action_balance_bonus = 1.0
                # Common bonus for both BUY and SELL actions to ensure equality
                if discrete_action in [ACTION_BUY, ACTION_SELL]:
                    action_balance_bonus += 1.0  # Equal bonus for trading actions
        
        return action_balance_bonus

    def _calculate_position_size_bonus(self, position: float, old_position: float = 0.0) -> float:
        """
        Calculate position size bonus to reward optimal position sizes.

        For perfectly symmetric rewards, position sizing should not favor any action.
        This function now returns neutral bonuses to avoid any BUY/SELL bias.

        Args:
            position: Current position size
            old_position: Position size before action

        Returns:
            Position size bonus value (neutral)
        """
        # For perfect symmetry, make position sizing completely neutral
        # No bonuses or penalties based on position size to avoid any bias
        return 0.0

    def _calculate_drawdown_penalty(self, portfolio_value: float, portfolio_value_history: List[float]) -> float:
        """
        Calculate drawdown penalty to prevent large losses.
        
        Args:
            portfolio_value: Current portfolio value
            portfolio_value_history: History of portfolio values
            
        Returns:
            Drawdown penalty value
        """
        drawdown_penalty = 0.0
        if len(portfolio_value_history) >= 5:
            peak_value = max(portfolio_value_history[-20:]) if len(portfolio_value_history) >= 20 else max(portfolio_value_history)
            current_drawdown = (peak_value - portfolio_value) / max(peak_value, 1.0)
            
            if current_drawdown > 0.05:  # 5% drawdown threshold
                drawdown_penalty = -current_drawdown * 100
        
        return drawdown_penalty

    def calculate_reward_simple(
        self,
        pnl: float,
        portfolio_value: float,
        position: float,
        old_position: float,
        action: int = 0,
        reward_history: Optional[List[float]] = None,
        portfolio_value_history: Optional[List[float]] = None,
    ) -> float:
        """
        Sharpe ratio-based reward function for better risk-adjusted performance.
        
        This reward function considers:
        1. Risk-adjusted returns (Sharpe-like ratio)
        2. Position sizing efficiency
        3. Trading frequency balance
        4. Drawdown control
        
        Args:
            pnl: Profit/Loss from action (in currency units)
            portfolio_value: Current portfolio value
            position: Current position size
            old_position: Position before action
            action: Action taken (0=HOLD, 1=BUY, 2=SELL)
            reward_history: History of rewards
            portfolio_value_history: History of portfolio values
            
        Returns:
            Risk-adjusted reward in range [clip_min, clip_max]
        """
        # Initialize history if not provided
        if reward_history is None:
            reward_history = []
        if portfolio_value_history is None:
            portfolio_value_history = [portfolio_value]
        
        # 1. Base PNL reward (scaled)
        reward_scale = self.get_setting_float("reward_scale", 1000.0)
        pnl_reward = pnl * reward_scale
        
        # 2. Risk adjustment using rolling volatility
        risk_adjusted_reward = pnl_reward * 0.1  # Default fallback
        
        if len(portfolio_value_history) >= 10:
            returns = []
            for i in range(1, len(portfolio_value_history)):
                ret = (portfolio_value_history[i] - portfolio_value_history[i-1]) / max(portfolio_value_history[i-1], 1.0)
                returns.append(ret)
            
            if returns:
                # Calculate rolling volatility (standard deviation of returns)
                volatility = np.std(returns) if len(returns) > 1 else 0.01
                
                # Sharpe-like ratio: reward / volatility
                if volatility > 0.001:  # Avoid division by zero
                    sharpe_ratio = pnl_reward / (volatility * 100)  # Scale for meaningful values
                    risk_adjusted_reward = sharpe_ratio * 10  # Scale up for learning
                else:
                    risk_adjusted_reward = pnl_reward * 0.1  # Fallback for low volatility
        
        # 3. Position sizing bonus (reward optimal position sizes)
        position_size_bonus = self._calculate_position_size_bonus(position, old_position)
        
        # Handle continuous actions (convert to discrete for reward calculation)
        discrete_action = self._convert_continuous_to_discrete_action(action)

        # 4. Action balance bonus (encourage balanced BUY/SELL usage)
        action_balance_bonus = self._calculate_action_balance_bonus(discrete_action)
        
        # 5. Win rate bonus (aggressive reward for high win rate target of 80%+)
        win_rate_bonus = self._calculate_win_rate_bonus(discrete_action, pnl)
        
        # 6. Trading activity bonus (encourage but don't force trading)
        trading_bonus = 0.0
        if discrete_action in [ACTION_BUY, ACTION_SELL]:
            position_change = abs(position - old_position)
            if position_change > 0.001:  # Meaningful position change
                trading_bonus = 2.0
        
        # 7. Drawdown penalty (prevent large losses)
        drawdown_penalty = self._calculate_drawdown_penalty(portfolio_value, portfolio_value_history)
        
        # Combine all reward components
        total_reward = (
            risk_adjusted_reward +
            position_size_bonus +
            action_balance_bonus +
            win_rate_bonus +
            trading_bonus +
            drawdown_penalty
        )
        
        # Clip to reasonable range (expanded for aggressive rewards)
        clip_min = self.get_setting_float("reward_clip_min", -40.0)
        clip_max = self.get_setting_float("reward_clip_max", 40.0)
        total_reward = max(clip_min, min(clip_max, total_reward))
        
        # Log detailed reward breakdown for debugging
        self.logger.debug(
            f"Sharpe-based reward: {total_reward:.4f} "
            f"(pnl={pnl_reward:.2f}, risk_adj={risk_adjusted_reward:.2f}, "
            f"pos_bonus={position_size_bonus:.2f}, action_balance={action_balance_bonus:.2f}, "
            f"win_rate_bonus={win_rate_bonus:.2f}, trade_bonus={trading_bonus:.2f}, "
            f"drawdown_pen={drawdown_penalty:.2f})"
        )
        
        return total_reward

    def test_reward_calculation(self) -> None:
        """Test reward calculation with sample inputs."""
        # Sample inputs
        test_cases = [
            {
                "action": ACTION_HOLD,
                "pnl": 100.0,
                "position": 0.01,
                "portfolio_value": 200000.0,
                "atr": 500.0,
                "current_price": 5000000.0,
                "old_position": 0.0,
                "step": 1,
                "description": "HOLD with small profit"
            },
            {
                "action": ACTION_BUY,
                "pnl": 200.0,
                "position": 0.02,
                "portfolio_value": 200000.0,
                "atr": 500.0,
                "current_price": 5000000.0,
                "old_position": 0.01,
                "step": 1,
                "description": "BUY with profit"
            },
            {
                "action": ACTION_SELL,
                "pnl": -100.0,
                "position": 0.0,
                "portfolio_value": 200000.0,
                "atr": 500.0,
                "current_price": 5000000.0,
                "old_position": 0.02,
                "step": 1,
                "description": "SELL with loss"
            }
        ]
        
        print("Testing reward calculation...")
        for i, case in enumerate(test_cases):
            reward = self.calculate_reward(
                action=case["action"],
                current_price=case["current_price"],
                position=case["position"],
                portfolio_value=case["portfolio_value"],
                atr=case["atr"],
                transaction_cost=10.0,
                reward_scaling=10.0,  # Scale rewards for testing
                pnl=case["pnl"],
                old_position=case["old_position"],
                step=case["step"],
                observation=None,
                reward_history=[],
                portfolio_value_history=[case["portfolio_value"]] * 30
            )
            print(f"Test {i+1}: {case['description']} -> Reward: {reward:.4f}")
            print(f"  Components: pnl={case['pnl']:.2f}, position={case['position']:.4f}, atr={case['atr']:.2f}")


__all__ = ["RewardCalculator"]
