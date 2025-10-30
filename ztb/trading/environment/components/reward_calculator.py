"""
Reward Calculator - Handles reward calculation logic for trading environment.

This module separates the complex reward calculation logic from the main environment class.
Refactored to follow SOLID principles with component-based architecture.
"""

# mypy: disable-error-code=literal-required

import math
from typing import List, Optional, Union

import numpy as np

from ztb.trading.constants import (
    ACTION_BUY,
    ACTION_HOLD,
    ACTION_SELL,
    MULTIPLIER_INDEX_BUY,
    MULTIPLIER_INDEX_HOLD,
    MULTIPLIER_INDEX_SELL,
)
from ztb.trading.environment.constants import EPSILON
from ztb.trading.environment.utils.config import EnvironmentConfig, RewardSettings
from ztb.utils.logging_utils import get_logger

from .asymmetric_reward_scaler import AsymmetricRewardScaler
from .dynamic_reward_shaper import DynamicRewardShaper
from .market_regime_detector import MarketRegimeDetector
from .signal_integrator import SignalIntegrator


class RewardCalculator:
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
        self._recent_actions: List[
            int
        ] = []  # Track recent actions for frequency penalty
        self.last_signal_strength: float = 0.0
        self.last_signal_reward: float = 0.0
        self._previous_portfolio_value = initial_portfolio_value  # Track previous portfolio value for delta calculation

        # Initialize components following SOLID principles

        # 1. Market Regime Detector
        self.logger.debug("Initializing MarketRegimeDetector")
        regime_detection_window = self.get_setting_int(
            "dynamic_reward_shaping.regime_detection_window", 20
        )
        adaptation_frequency = self.get_setting_int(
            "dynamic_reward_shaping.adaptation_frequency", 10
        )
        high_volatility_threshold = self.get_setting_float(
            "dynamic_reward_shaping.volatility_coefficients.high_volatility_threshold",
            0.02,
        )
        low_volatility_threshold = self.get_setting_float(
            "dynamic_reward_shaping.volatility_coefficients.low_volatility_threshold",
            0.005,
        )
        trend_strength_threshold = self.get_setting_float(
            "dynamic_reward_shaping.trend_coefficients.trend_strength_threshold", 0.001
        )

        self.market_regime_detector = MarketRegimeDetector(
            regime_detection_window=regime_detection_window,
            adaptation_frequency=adaptation_frequency,
            high_volatility_threshold=high_volatility_threshold,
            low_volatility_threshold=low_volatility_threshold,
            trend_strength_threshold=trend_strength_threshold,
        )

        # 2. Dynamic Reward Shaper
        dynamic_reward_shaping_enabled = self.get_setting_bool(
            "dynamic_reward_shaping.enabled", False
        )
        market_regime_awareness = self.get_setting_bool(
            "dynamic_reward_shaping.market_regime_awareness", False
        )
        volatility_adjusted_rewards = self.get_setting_bool(
            "dynamic_reward_shaping.volatility_adjusted_rewards", False
        )
        trend_strength_bonus = self.get_setting_bool(
            "dynamic_reward_shaping.trend_strength_bonus", False
        )

        # Regime coefficients
        bull_market_bonus_coeff = self.get_setting_float(
            "dynamic_reward_shaping.regime_coefficients.bull_market_bonus_coeff", 1.2
        )
        bear_market_penalty_coeff = self.get_setting_float(
            "dynamic_reward_shaping.regime_coefficients.bear_market_penalty_coeff", 0.8
        )
        sideways_market_penalty_coeff = self.get_setting_float(
            "dynamic_reward_shaping.regime_coefficients.sideways_market_penalty_coeff",
            0.9,
        )
        volatile_market_bonus_coeff = self.get_setting_float(
            "dynamic_reward_shaping.regime_coefficients.volatile_market_bonus_coeff",
            1.1,
        )

        # Volatility coefficients
        high_volatility_bonus = self.get_setting_float(
            "dynamic_reward_shaping.volatility_coefficients.high_volatility_bonus", 1.3
        )
        low_volatility_penalty = self.get_setting_float(
            "dynamic_reward_shaping.volatility_coefficients.low_volatility_penalty", 0.7
        )

        # Trend coefficients
        strong_trend_bonus = self.get_setting_float(
            "dynamic_reward_shaping.trend_coefficients.strong_trend_bonus", 1.2
        )
        weak_trend_penalty = self.get_setting_float(
            "dynamic_reward_shaping.trend_coefficients.weak_trend_penalty", 0.9
        )

        self.dynamic_reward_shaper = DynamicRewardShaper(
            market_regime_detector=self.market_regime_detector,
            enabled=dynamic_reward_shaping_enabled,
            market_regime_awareness=market_regime_awareness,
            volatility_adjusted_rewards=volatility_adjusted_rewards,
            trend_strength_bonus=trend_strength_bonus,
            bull_market_bonus_coeff=bull_market_bonus_coeff,
            bear_market_penalty_coeff=bear_market_penalty_coeff,
            sideways_market_penalty_coeff=sideways_market_penalty_coeff,
            volatile_market_bonus_coeff=volatile_market_bonus_coeff,
            high_volatility_threshold=high_volatility_threshold,
            low_volatility_threshold=low_volatility_threshold,
            high_volatility_bonus=high_volatility_bonus,
            low_volatility_penalty=low_volatility_penalty,
            trend_strength_threshold=trend_strength_threshold,
            strong_trend_bonus=strong_trend_bonus,
            weak_trend_penalty=weak_trend_penalty,
        )

        # 3. Signal Integrator
        signal_guide_enabled = getattr(self.config, "signal_guidance_enabled", False)
        signal_guidance_config = getattr(self.config, "signal_guidance", {})
        guidance_level_str = signal_guidance_config.get("guidance_level", "strong")
        signal_bonus_weight = signal_guidance_config.get("signal_bonus_weight", 0.1)
        signal_penalty_weight = signal_guidance_config.get(
            "signal_penalty_weight", 0.05
        )
        granville_weight = signal_guidance_config.get("granville_weight", 1.2)
        dow_theory_weight = signal_guidance_config.get("dow_theory_weight", 1.5)
        heikin_ashi_weight = signal_guidance_config.get("heikin_ashi_weight", 1.0)
        enable_advanced_integration = signal_guidance_config.get(
            "enable_advanced_integration", True
        )

        self.signal_integrator = SignalIntegrator(
            config=config,
            enabled=signal_guide_enabled,
            guidance_level=guidance_level_str,
            signal_bonus_weight=signal_bonus_weight,
            signal_penalty_weight=signal_penalty_weight,
            granville_weight=granville_weight,
            dow_theory_weight=dow_theory_weight,
            heikin_ashi_weight=heikin_ashi_weight,
            enable_advanced_integration=enable_advanced_integration,
        )

        # Reference to signal integration for backward compatibility
        self.signal_integration = self.signal_integrator.signal_integration
        self.signal_guide = self.signal_integrator.signal_guide

        # 4. Asymmetric Reward Scaler
        long_position_reward_multiplier = self.get_setting_float(
            "long_position_reward_multiplier", 1.3
        )
        short_position_reward_multiplier = self.get_setting_float(
            "short_position_reward_multiplier", 0.7
        )
        long_position_penalty_multiplier = self.get_setting_float(
            "long_position_penalty_multiplier", 0.9
        )
        short_position_penalty_multiplier = self.get_setting_float(
            "short_position_penalty_multiplier", 0.95
        )

        self.asymmetric_reward_scaler = AsymmetricRewardScaler(
            long_position_reward_multiplier=long_position_reward_multiplier,
            short_position_reward_multiplier=short_position_reward_multiplier,
            long_position_penalty_multiplier=long_position_penalty_multiplier,
            short_position_penalty_multiplier=short_position_penalty_multiplier,
        )

        self.logger.info(
            f"Dynamic reward shaping enabled: {dynamic_reward_shaping_enabled}"
        )
        if dynamic_reward_shaping_enabled:
            self.logger.info(f"Market regime awareness: {market_regime_awareness}")
            self.logger.info(
                f"Volatility adjusted rewards: {volatility_adjusted_rewards}"
            )
            self.logger.info(f"Trend strength bonus: {trend_strength_bonus}")
        self.logger.info(f"Signal guide enabled: {signal_guide_enabled}")

        # Initialize behavior optimization parameters
        self.action_balance_target = self.get_setting_float(
            "behavior_optimization.action_balance_target", 0.8
        )
        self.entropy_regularization = self.get_setting_float(
            "behavior_optimization.entropy_regularization", 0.01
        )
        self.action_smoothing = self.get_setting_float(
            "behavior_optimization.action_smoothing", 0.1
        )
        self.consistency_penalty = self.get_setting_float(
            "behavior_optimization.consistency_penalty", 0.05
        )
        self.balance_penalty = self.get_setting_float(
            "behavior_optimization.balance_penalty", 1.0
        )

        # Validate and clamp action_balance_target to prevent invalid target ratios
        original_target = self.action_balance_target
        self.action_balance_target = max(0.05, min(0.45, self.action_balance_target))
        if self.action_balance_target != original_target:
            self.logger.warning(
                f"action_balance_target {original_target} is out of valid range [0.05, 0.45], "
                f"clamped to {self.action_balance_target}"
            )

        self.logger.info(
            f"Behavior optimization - Balance target: {self.action_balance_target}, "
            f"Entropy regularization: {self.entropy_regularization}, "
            f"Action smoothing: {self.action_smoothing}, "
            f"Consistency penalty: {self.consistency_penalty}, "
            f"Balance penalty: {self.balance_penalty}"
        )

    def get_setting_float(self, key: str, default: float) -> float:
        """Get float reward setting with fallback. Supports nested keys with dot notation."""
        if self.reward_settings:
            value = self._get_nested_setting(key)
            if isinstance(value, (int, float)):
                return float(value)
        return default

    def get_setting_int(self, key: str, default: int) -> int:
        """Get integer reward setting with fallback. Supports nested keys with dot notation."""
        if self.reward_settings:
            value = self._get_nested_setting(key)
            if isinstance(value, (int, float)):
                return int(value)
        return default

    def get_setting_bool(self, key: str, default: bool) -> bool:
        """Get boolean reward setting with fallback. Supports nested keys with dot notation."""
        if self.reward_settings:
            value = self._get_nested_setting(key)
            if isinstance(value, bool):
                return value
            if isinstance(value, (int, float)):
                return bool(value)
        return default

    def _get_nested_setting(self, key: str) -> Optional[Union[int, float, bool, str]]:
        """Get nested setting value using dot notation."""
        keys = key.split(".")
        value = self.reward_settings

        try:
            for k in keys:
                if isinstance(value, dict):
                    value = value[k]
                else:
                    return None
            return value
        except (KeyError, TypeError):
            return None

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
        self.last_signal_strength = 0.0
        self.last_signal_reward = 0.0

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
                pnl,
                portfolio_value,
                position,
                old_position,
                action,
                reward_history,
                portfolio_value_history,
                current_price,
                step,
                transaction_cost,
            )

        # Original complex reward function
        curriculum_stage = self.config.curriculum_stage
        self.logger.info(
            "Curriculum stage: %s, position: %.2f, action: %d",
            curriculum_stage,
            position,
            action,
        )

        eps = self.get_setting_float("eps", 1e-8)
        atr = atr if atr > eps else 1.0
        max_position_size = max(eps, self.config.max_position_size)

        # Calculate effective max position considering capital constraints
        effective_max_position = min(
            max_position_size, self.initial_portfolio_value / max(current_price, eps)
        )
        self.logger.debug(
            f"Position calculations: max_position_size={max_position_size:.4f}, effective_max_position={effective_max_position:.4f}, initial_portfolio_value={self.initial_portfolio_value:.2f}, current_price={current_price:.2f}"
        )

        # Adapt reward scaling based on max position size to prevent clipping
        scale_adjustment_base = self.get_setting_float("scale_adjustment_base", 1.0)
        scale_adjustment = scale_adjustment_base / max(0.01, max_position_size)
        reward_scaling = reward_scaling * scale_adjustment

        atr_normalised = pnl / atr
        portfolio_return = pnl / max(abs(self.initial_portfolio_value), eps)

        # Adjust pnl for transaction costs if a trade occurred
        # Note: transaction_cost is already deducted in position_manager, so we don't deduct again
        # if abs(position - old_position) > eps:  # Trade occurred
        #     adjusted_pnl = pnl - transaction_cost
        #     self.logger.debug(f"Trade occurred: adjusted pnl from {pnl:.4f} to {adjusted_pnl:.4f} (transaction_cost: {transaction_cost:.4f})")
        #     pnl = adjusted_pnl
        #     atr_normalised = pnl / atr
        #     portfolio_return = pnl / max(abs(self.initial_portfolio_value), eps)

        # Curriculum learning stages
        if curriculum_stage == "forced_balance":
            reward = self._calculate_forced_balance_reward(action)
            # Use smaller scaling for forced balance to prevent extreme values
            forced_balance_scaling = self.get_setting_float(
                "forced_balance_scaling", 1.0
            )
            reward *= forced_balance_scaling
            # Apply asymmetric scaling for v435 enhancement
            reward = self.asymmetric_reward_scaler.scale_reward(reward, position, pnl)
            # Apply clipping
            reward_clip_min = self.get_setting_float("reward_clip_min", -80.0)
            reward_clip_max = self.get_setting_float("reward_clip_max", 80.0)
            reward = np.clip(reward, reward_clip_min, reward_clip_max)
            self.logger.info(
                f"Forced balance reward: {reward}, action_counts: {self._action_counts}"
            )
            self.logger.debug(f"Final reward: {reward}")
            # Apply signal integration (v436 enhancement)
            reward = self.signal_integrator.integrate_signal(
                reward, observation, action, step
            )
            return reward
        elif curriculum_stage == "balanced_transition":
            reward = self._calculate_balanced_transition_reward(
                action,
                atr_normalised,
                portfolio_return,
                position,
                effective_max_position,
                current_price,
                atr,
                pnl,
                reward_scaling,
            )
            # Apply asymmetric scaling for v435 enhancement
            reward = self.asymmetric_reward_scaler.scale_reward(reward, position, pnl)
            # Apply clipping
            reward_clip_min = self.get_setting_float("reward_clip_min", -80.0)
            reward_clip_max = self.get_setting_float("reward_clip_max", 80.0)
            reward = np.clip(reward, reward_clip_min, reward_clip_max)
            # Apply signal integration (v436 enhancement)
            reward = self.signal_integrator.integrate_signal(
                reward, observation, action, step
            )
            return reward
        elif curriculum_stage == "pnl_focused":
            reward = self._calculate_pnl_focused_reward(
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
            # Apply asymmetric scaling for v435 enhancement
            reward = self.asymmetric_reward_scaler.scale_reward(reward, position, pnl)
            # Apply clipping
            reward_clip_min = self.get_setting_float("reward_clip_min", -80.0)
            reward_clip_max = self.get_setting_float("reward_clip_max", 80.0)
            reward = np.clip(reward, reward_clip_min, reward_clip_max)
            # Apply signal integration (v436 enhancement)
            reward = self.signal_integrator.integrate_signal(
                reward, observation, action, step
            )
            return reward
        elif curriculum_stage == "trading_focused":
            reward = self._calculate_trading_focused_reward(
                action,
                atr_normalised,
                portfolio_return,
                position,
                effective_max_position,
                current_price,
                atr,
                pnl,
                reward_scaling,
            )
            # Apply asymmetric scaling for v435 enhancement
            reward = self.asymmetric_reward_scaler.scale_reward(reward, position, pnl)
            # Apply clipping
            reward_clip_min = self.get_setting_float("reward_clip_min", -80.0)
            reward_clip_max = self.get_setting_float("reward_clip_max", 80.0)
            reward = np.clip(reward, reward_clip_min, reward_clip_max)
            # Apply signal integration (v436 enhancement)
            reward = self.signal_integrator.integrate_signal(
                reward, observation, action, step
            )
            return reward
        elif curriculum_stage == "profit_optimized":
            reward = self._calculate_profit_optimized_reward(
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
            )
            # Apply asymmetric scaling for v435 enhancement
            reward = self.asymmetric_reward_scaler.scale_reward(reward, position, pnl)
            # Apply clipping
            reward_clip_min = self.get_setting_float("reward_clip_min", -80.0)
            reward_clip_max = self.get_setting_float("reward_clip_max", 80.0)
            reward = np.clip(reward, reward_clip_min, reward_clip_max)
            # Apply signal integration (v436 enhancement)
            reward = self.signal_integrator.integrate_signal(
                reward, observation, action, step
            )
            return reward
        elif curriculum_stage == "ultra_profit":
            reward = self._calculate_ultra_profit_reward(
                action,
                atr_normalised,
                portfolio_return,
                position,
                effective_max_position,
                current_price,
                atr,
                pnl,
                reward_scaling,
            )
            # Apply asymmetric scaling for v435 enhancement
            reward = self.asymmetric_reward_scaler.scale_reward(reward, position, pnl)
            # Apply clipping
            reward_clip_min = self.get_setting_float("reward_clip_min", -80.0)
            reward_clip_max = self.get_setting_float("reward_clip_max", 80.0)
            reward = np.clip(reward, reward_clip_min, reward_clip_max)
            # Apply signal integration (v436 enhancement)
            reward = self.signal_integrator.integrate_signal(
                reward, observation, action, step
            )
            return reward
        elif curriculum_stage == "stability_optimized":
            reward = self._calculate_stability_optimized_reward(
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
            # Apply asymmetric scaling for v435 enhancement
            reward = self.asymmetric_reward_scaler.scale_reward(reward, position, pnl)
            # Apply clipping
            reward_clip_min = self.get_setting_float("reward_clip_min", -80.0)
            reward_clip_max = self.get_setting_float("reward_clip_max", 80.0)
            reward = np.clip(reward, reward_clip_min, reward_clip_max)
            # Apply signal integration (v436 enhancement)
            reward = self.signal_integrator.integrate_signal(
                reward, observation, action, step
            )
            return reward
        elif curriculum_stage == "backtest_optimization":
            # Calculate portfolio value delta for correlation improvement
            portfolio_value_delta = portfolio_value - self._previous_portfolio_value
            self._previous_portfolio_value = portfolio_value

            reward = self._calculate_backtest_optimization_reward(
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
                portfolio_value_delta,
            )
            # Apply asymmetric scaling for v435 enhancement
            reward = self.asymmetric_reward_scaler.scale_reward(reward, position, pnl)
            # Apply clipping
            reward_clip_min = self.get_setting_float("reward_clip_min", -80.0)
            reward_clip_max = self.get_setting_float("reward_clip_max", 80.0)
            reward = np.clip(reward, reward_clip_min, reward_clip_max)
            # Apply signal integration (v436 enhancement)
            reward = self.signal_integrator.integrate_signal(
                reward, observation, action, step
            )
            return reward
        else:
            # Default stage
            reward = self._calculate_default_reward(
                action,
                atr_normalised,
                portfolio_return,
                position,
                effective_max_position,
                current_price,
                atr,
                pnl,
            )
            reward *= reward_scaling
            # Apply asymmetric scaling for v435 enhancement
            reward = self.asymmetric_reward_scaler.scale_reward(reward, position, pnl)
            # Apply clipping
            reward_clip_min = self.get_setting_float("reward_clip_min", -80.0)
            reward_clip_max = self.get_setting_float("reward_clip_max", 80.0)
            reward = np.clip(reward, reward_clip_min, reward_clip_max)
            # Apply signal integration (v436 enhancement)
            reward = self.signal_integrator.integrate_signal(
                reward, observation, action, step
            )
            return reward

    def calculate_reward_simple(
        self,
        pnl: float,
        portfolio_value: float,
        position: float,
        old_position: float,
        action: int,
        reward_history: List[float],
        portfolio_value_history: List[float],
        current_price: float = 0.0,
        step: int = 0,
        transaction_cost: float = 0.0,
    ) -> float:
        """
        Calculate simple reward based on PnL with v431 enhancements and v440.1 dynamic shaping.

        Enhanced with v431 successful elements:
        - HOLD penalty multiplier for encouraging trading activity
        - Trade frequency bonus for promoting active trading
        - Reward scaling and clipping for stable learning
        - Dynamic reward shaping based on market conditions (v440.1)

        Args:
            pnl: Profit/Loss from action
            portfolio_value: Current portfolio value
            position: Current position
            old_position: Position before action
            action: Action taken
            reward_history: History of rewards
            portfolio_value_history: History of portfolio values
            current_price: Current market price (for regime detection)
            step: Current step number (for regime adaptation)
            transaction_cost: Transaction cost for the action

        Returns:
            Enhanced simple reward value
        """
        try:
            self.logger.debug(
                f"calculate_reward_simple called: pnl={pnl}, action={action}"
            )
            # Basic NaN/inf checks
            if np.isnan(pnl) or np.isinf(pnl):
                self.logger.warning(
                    "RewardCalculator failed, using simple reward: math range error"
                )
                return 0.0

            if np.isnan(portfolio_value) or np.isinf(portfolio_value):
                self.logger.warning(
                    "RewardCalculator failed, using simple reward: invalid portfolio_value"
                )
                return 0.0

            if np.isnan(position) or np.isinf(position):
                self.logger.warning(
                    "RewardCalculator failed, using simple reward: invalid position"
                )
                return 0.0

            # Get v431 enhancement parameters
            hold_penalty_multiplier = self.get_setting_float(
                "hold_penalty_multiplier", 1.0
            )
            trade_frequency_bonus = self.get_setting_float("trade_frequency_bonus", 0.0)
            reward_scaling = self.get_setting_float("reward_scaling", 1.0)
            reward_clip_value = self.get_setting_float("reward_clip_value", 10.0)

            # Adjust PnL for transaction costs if trade occurred
            # Note: transaction_cost is already deducted in position_manager, so we don't deduct again
            adjusted_pnl = pnl

            # Base PnL-based reward (scaled)
            if adjusted_pnl > 0:
                reward = adjusted_pnl * reward_scaling
            elif adjusted_pnl < 0:
                reward = adjusted_pnl * reward_scaling
            else:
                reward = 0.0

            # HOLD penalty (v431 enhancement) - encourage trading activity
            if action == ACTION_HOLD:  # 0 = HOLD
                reward *= hold_penalty_multiplier

            # Trade frequency bonus (v431 enhancement) - promote active trading
            if action in [ACTION_BUY, ACTION_SELL]:  # 1 = BUY, 2 = SELL
                reward += trade_frequency_bonus

            # Small penalty for large position changes to encourage stability
            position_change = abs(position - old_position)
            if position_change > 0.1:  # Large position change
                reward -= 0.1

            # Apply dynamic reward shaping (v440.1 enhancement)
            reward = self.dynamic_reward_shaper.shape_reward(
                reward, current_price, step, pnl
            )

            # Apply signal-based reward integration
            if self.signal_integrator.enabled:
                # Create observation from current state (simplified for now)
                observation = np.array([current_price, position, pnl])
                signal_reward = self.signal_integrator.integrate_signal(
                    reward=reward, observation=observation, action=action, step=step
                )
                reward = signal_reward  # integrate_signal returns the modified reward

            # Apply asymmetric reward scaling based on position
            reward = self.asymmetric_reward_scaler.scale_reward(reward, position, pnl)

            # Apply reward clipping
            # reward = np.clip(reward, -reward_clip_value, reward_clip_value)

            # Ensure reward is finite
            if not np.isfinite(reward):
                self.logger.warning(
                    "RewardCalculator failed, using simple reward: non-finite reward"
                )
                return 0.0

            return reward

        except Exception as e:
            self.logger.error(f"RewardCalculator failed, using simple reward: {e}")
            return 0.0

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

            self.logger.debug(
                f"Forced balance: ratios={action_ratios}, penalty={balance_penalty:.3f}, reward={base_reward:.3f}"
            )
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
        penalty = (
            self.balance_penalty
        )  # Use the initialized balance_penalty from behavior_optimization
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
                    self.logger.info(
                        f"Balance penalty applied: {balance_penalty:.3f}, ratios: {action_ratios}, targets: {target_ratios}"
                    )

        # Calculate base reward
        base_reward = self._calculate_base_reward(
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
        self.logger.info(
            f"Balanced transition: base_reward={base_reward:.3f}, balance_penalty={balance_penalty:.3f}, final_reward={final_reward:.3f}, action_counts={self._action_counts}"
        )
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
        penalty = self.get_setting_float(
            "balance_penalty", 8.0
        )  # Higher penalty for trading focus
        balance_penalty = 0.0

        if total_actions >= 10:
            action_ratios = [count / total_actions for count in self._action_counts]

            for i, ratio in enumerate(action_ratios):
                deviation = abs(ratio - target_ratios[i])
                if deviation > tolerance:
                    # Penalty proportional to deviation beyond tolerance
                    excess_deviation = deviation - tolerance
                    balance_penalty += penalty * excess_deviation

            self.logger.info(
                f"Trading focused penalty applied: {balance_penalty:.3f}, ratios: {action_ratios}, targets: {target_ratios}"
            )

        # Calculate base reward
        base_reward = self._calculate_base_reward(
            action,
            atr_normalised,
            portfolio_return,
            position,
            effective_max_position,
            current_price,
            atr,
            pnl,
        )

        # Add strong HOLD penalty
        hold_penalty_rate = self.get_setting_float("hold_penalty_rate", 0.01)
        if action == ACTION_HOLD:
            # Strong penalty for HOLD action
            hold_penalty = (
                hold_penalty_rate * abs(position) / max(effective_max_position, 0.01)
            )
            base_reward -= hold_penalty
            self.logger.debug(f"HOLD penalty applied: {hold_penalty:.3f}")

        # Boost trading bonuses
        trading_bonus_multiplier = self.get_setting_float(
            "trading_bonus_multiplier", 2.0
        )
        if action in [ACTION_BUY, ACTION_SELL]:
            trading_bonus = (
                self.get_setting_float("trading_bonus", 0.01) * trading_bonus_multiplier
            )
            base_reward += trading_bonus
            self.logger.debug(f"Trading bonus applied: {trading_bonus:.3f}")

        final_reward = base_reward - balance_penalty
        self.logger.info(
            f"Trading focused: base_reward={base_reward:.3f}, balance_penalty={balance_penalty:.3f}, final_reward={final_reward:.3f}, action_counts={self._action_counts}"
        )
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
            action,
            atr_normalised,
            portfolio_return,
            position,
            effective_max_position,
            current_price,
            atr,
            pnl,
            observation,
        )

        # Profit/loss based reward adjustment (normalized to prevent large values)
        profit_multiplier = self.get_setting_float("profit_multiplier", 2.0)
        loss_penalty_multiplier = self.get_setting_float("loss_penalty_multiplier", 1.5)

        # Normalize pnl by ATR and position scale to prevent extreme values
        pnl_normalizer = atr * effective_max_position * current_price
        normalized_pnl = pnl / max(pnl_normalizer, 1e-8)

        if pnl > 0:
            # Boost profitable trades (normalized)
            profit_bonus = normalized_pnl * profit_multiplier
            base_reward += profit_bonus
            self.logger.debug(
                f"Profit bonus applied: {profit_bonus:.3f} for normalized_pnl={normalized_pnl:.6f}"
            )
        elif pnl < 0:
            # Penalize losing trades more heavily (normalized)
            loss_penalty = abs(normalized_pnl) * loss_penalty_multiplier
            base_reward -= loss_penalty
            self.logger.debug(
                f"Loss penalty applied: {loss_penalty:.3f} for normalized_pnl={normalized_pnl:.6f}"
            )

        # Strong HOLD penalty (but not as extreme as trading_focused)
        hold_penalty_rate = self.get_setting_float(
            "hold_penalty_rate", 0.1
        )  # Use config value, default 0.1
        if action == ACTION_HOLD:
            hold_penalty = (
                hold_penalty_rate * abs(position) / max(effective_max_position, 0.01)
            )
            base_reward -= hold_penalty
            self.logger.debug(f"HOLD penalty applied: {hold_penalty:.3f}")

        # Moderate trading bonuses
        trading_bonus_multiplier = self.get_setting_float(
            "trading_bonus_multiplier", 3.0
        )
        if action in [ACTION_BUY, ACTION_SELL]:
            trading_bonus = (
                self.get_setting_float("trading_bonus", 0.01) * trading_bonus_multiplier
            )
            base_reward += trading_bonus
            self.logger.debug(f"Trading bonus applied: {trading_bonus:.3f}")

        # Position size bonus to encourage moderate trading activity (v438.1 enhancement)
        position_size_bonus_rate = self.get_setting_float(
            "position_size_bonus_rate", 0.05
        )
        position_utilization = abs(position) / max(effective_max_position, 0.01)
        if 0.1 <= position_utilization <= 0.8:  # Sweet spot for moderate positions
            position_size_bonus = position_size_bonus_rate * position_utilization
            base_reward += position_size_bonus
            self.logger.debug(
                f"Position size bonus applied: {position_size_bonus:.3f} (utilization: {position_utilization:.2f})"
            )

        # Activity incentive bonus for recent trading frequency (v438.1 enhancement)
        activity_bonus_rate = self.get_setting_float("activity_bonus_rate", 0.02)
        recent_trades = sum(
            1 for a in self._recent_actions[-5:] if a != ACTION_HOLD
        )  # Last 5 actions
        if recent_trades >= 2:  # At least 2 trades in last 5 actions
            activity_bonus = activity_bonus_rate * (recent_trades / 5.0)
            base_reward += activity_bonus
            self.logger.debug(
                f"Activity bonus applied: {activity_bonus:.3f} (recent trades: {recent_trades}/5)"
            )

        final_reward = base_reward - balance_penalty

        final_reward = base_reward - balance_penalty
        self.logger.info(
            f"Profit optimized: base_reward={base_reward:.3f}, balance_penalty={balance_penalty:.3f}, pnl={pnl:.3f}, final_reward={final_reward:.3f}"
        )
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
        """Simplified ultra-profit reward that focuses on basic trading principles."""

        # Basic reward components
        reward = 0.0

        # Profit/Loss component - normalized by ATR
        if pnl > 0:
            reward += atr_normalised * 2.0  # Increased reward for profits
        elif pnl < 0:
            reward -= abs(atr_normalised) * 1.0  # Penalty for losses

        # Position penalty - discourage excessive position sizes
        position_utilization = abs(position) / effective_max_position
        if position_utilization > 0.5:
            reward -= (position_utilization - 0.5) * 0.2

        # Action diversity encouragement - reduced penalty
        self._action_counts[action] += 1
        total_actions = sum(self._action_counts)

        if total_actions >= 10:
            action_ratios = [count / total_actions for count in self._action_counts]
            # Encourage balanced actions: roughly 20% HOLD, 40% BUY, 40% SELL
            target_ratios = [0.2, 0.4, 0.4]

            for i, ratio in enumerate(action_ratios):
                deviation = abs(ratio - target_ratios[i])
                if deviation > 0.15:  # More lenient deviation
                    reward -= deviation * 0.005

        # Strong trading bonus for BUY/SELL actions to encourage trading
        if action in [ACTION_BUY, ACTION_SELL]:
            reward += 0.1  # Increased trading bonus

        # Apply scaling and clipping
        reward *= reward_scaling
        reward_clip_min = self.get_setting_float("reward_clip_min", -1.0)
        reward_clip_max = self.get_setting_float("reward_clip_max", 1.0)
        reward = np.clip(reward, reward_clip_min, reward_clip_max)

        self.logger.debug(
            f"Simplified ultra profit reward: pnl={pnl:.4f}, atr_normalised={atr_normalised:.4f}, position={position:.4f}, action={action}, reward={reward:.6f}"
        )
        return reward

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
        reward_scaling: float,
        observation: Optional[np.ndarray],
        step: int,
    ) -> float:
        """Stage 2: PnL-focused reward with trend analysis."""
        base_profit_bonus = (
            max(
                0.0,
                self.get_setting_float("base_profit_bonus_atr_coeff", 1.5)
                * atr_normalised
                + self.get_setting_float("base_profit_bonus_portfolio_coeff", 1.2)
                * portfolio_return,
            )
            if pnl > 0
            else 0.0
        )

        # Trend analysis if observation available
        trend_multiplier = 1.0
        if observation is not None and hasattr(observation, "__getitem__"):
            try:
                # Assume RSI and MACD are in observation
                rsi_idx = -2 if len(observation) > 2 else None
                macd_idx = -1 if len(observation) > 1 else None

                if rsi_idx is not None and macd_idx is not None:
                    rsi = float(observation[rsi_idx])
                    macd_hist = float(observation[macd_idx])

                    # Check for NaN/inf
                    if (
                        np.isnan(rsi)
                        or np.isinf(rsi)
                        or np.isnan(macd_hist)
                        or np.isinf(macd_hist)
                    ):
                        trend_multiplier = 1.0
                    else:
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
        multipliers_raw = self.reward_settings.get(
            "profit_bonus_multipliers", [1.0, 1.0, 0.8]
        )
        if len(multipliers_raw) >= 3:
            multipliers = [float(x) for x in multipliers_raw[:3]]
        else:
            multipliers = [1.0, 1.0, 0.8]

        if action == ACTION_BUY:
            profit_bonus = (
                base_profit_bonus * multipliers[MULTIPLIER_INDEX_BUY] * trend_multiplier
            )
        elif action == ACTION_SELL:
            profit_bonus = (
                base_profit_bonus
                * multipliers[MULTIPLIER_INDEX_SELL]
                * trend_multiplier
            )
        else:  # HOLD
            profit_bonus = (
                base_profit_bonus
                * multipliers[MULTIPLIER_INDEX_HOLD]
                * trend_multiplier
            )

        # Action penalties with configurable per-action penalties
        # Get action-specific penalties from settings (default: 0.0)
        hold_penalty = self.get_setting_float("hold_action_penalty", 0.0)
        buy_penalty = self.get_setting_float("buy_action_penalty", 0.0)
        sell_penalty = self.get_setting_float("sell_action_penalty", 0.0)

        # Base action penalties (legacy behavior)
        base_action_penalty = (
            self.get_setting_float("base_action_penalty", 0.015)
            if action in [ACTION_BUY, ACTION_SELL]
            else 0.0
        )

        if action == ACTION_HOLD:
            position_size_factor = abs(position) / effective_max_position
            volatility_factor = min(atr / (current_price * 0.01), 1.0)
            base_action_penalty = self.get_setting_float("hold_penalty_base", 0.01) + (
                self.get_setting_float("hold_penalty_position_factor", 0.04)
                * position_size_factor
                * volatility_factor
            )
            base_action_penalty *= self.get_setting_float(
                "hold_penalty_multiplier", 1.0
            )
            # Add configured HOLD penalty
            action_penalty = base_action_penalty + hold_penalty
        elif action == ACTION_BUY:
            # Add configured BUY penalty (negative value = reward)
            action_penalty = base_action_penalty + buy_penalty
        else:  # ACTION_SELL
            # Add configured SELL penalty (negative value = reward)
            action_penalty = base_action_penalty + sell_penalty

        # Loss penalty
        loss_penalty = (
            self.get_setting_float("loss_penalty_coeff", -0.2) * abs(atr_normalised)
            if pnl < 0
            else 0.0
        )

        # Position penalty
        position_penalty = self._calculate_position_penalty(
            position, effective_max_position
        )

        reward = profit_bonus - action_penalty + loss_penalty - position_penalty
        self.logger.debug(
            f"PnL focused reward components: profit_bonus={profit_bonus:.4f}, action_penalty={action_penalty:.4f}, loss_penalty={loss_penalty:.4f}, position_penalty={position_penalty:.4f}, final={reward:.4f}"
        )
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
        """Simplified reward calculation with balanced structure."""
        # 1. 基本アクション報酬 - 各アクションに小さな固定報酬
        base_action_reward = self.get_setting_float("base_action_reward", 0.01)
        if action == ACTION_BUY:
            action_reward = base_action_reward
        elif action == ACTION_SELL:
            action_reward = base_action_reward
        else:  # HOLD
            action_reward = base_action_reward * 0.5  # HOLDは少し低い報酬

        # 2. パフォーマンス報酬 - 正規化されたパフォーマンス指標
        performance_reward = 0.0
        if abs(pnl) > EPSILON:  # EPSILONを使ってゼロ除算を防ぐ
            # PnLをATRで正規化
            normalized_pnl = pnl / (atr + EPSILON)
            # ポートフォリオリターンを考慮
            performance_reward = normalized_pnl + 0.1 * portfolio_return
            # 利益は正、損失は負
            performance_reward *= self.get_setting_float("performance_multiplier", 1.0)

        # 3. バランスペナルティ - アクション分布の不均衡に対するペナルティ
        balance_penalty = 0.0
        if len(self._recent_actions) >= 10:
            recent_actions = self._recent_actions[-10:]
            buy_count = recent_actions.count(ACTION_BUY)
            sell_count = recent_actions.count(ACTION_SELL)
            hold_count = recent_actions.count(ACTION_HOLD)

            # 理想的な分布からの偏差を計算
            total_actions = len(recent_actions)
            ideal_buy_ratio = 0.3
            ideal_sell_ratio = 0.3
            ideal_hold_ratio = 0.4

            actual_buy_ratio = buy_count / total_actions
            actual_sell_ratio = sell_count / total_actions
            actual_hold_ratio = hold_count / total_actions

            # 偏差の二乗和
            balance_penalty = (
                (actual_buy_ratio - ideal_buy_ratio) ** 2
                + (actual_sell_ratio - ideal_sell_ratio) ** 2
                + (actual_hold_ratio - ideal_hold_ratio) ** 2
            ) * self.get_setting_float("balance_penalty_multiplier", 0.1)

        # 4. リスクペナルティ - ポジションサイズやボラティリティに対するペナルティ
        risk_penalty = 0.0
        if abs(position) > EPSILON:
            # ポジションサイズペナルティ
            position_size_penalty = (
                abs(position) / effective_max_position
            ) * self.get_setting_float("position_size_penalty", 0.05)
            # ボラティリティペナルティ
            volatility_penalty = (
                atr / (current_price + EPSILON)
            ) * self.get_setting_float("volatility_penalty", 0.02)
            risk_penalty = position_size_penalty + volatility_penalty

        # 総報酬の計算
        total_reward = (
            action_reward + performance_reward - balance_penalty - risk_penalty
        )

        # 報酬のスケーリング
        reward_scale = self.get_setting_float("reward_scale", 1.0)
        total_reward *= reward_scale

        self.logger.debug(
            f"Simplified reward: action={action_reward:.4f}, performance={performance_reward:.4f}, balance_penalty={balance_penalty:.4f}, risk_penalty={risk_penalty:.4f}, total={total_reward:.4f}"
        )
        return total_reward

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
            action,
            atr_normalised,
            portfolio_return,
            position,
            effective_max_position,
            current_price,
            atr,
            pnl,
            observation,
        )

    def _calculate_position_penalty(
        self, position: float, effective_max_position: float
    ) -> float:
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
        diversity_multiplier = diversity_score**2  # Quadratic scaling

        return base_bonus * diversity_multiplier

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
                drawdown_ratio = (
                    initial_cumulative - cumulative_reward
                ) / initial_cumulative
                drawdown_threshold = 0.5
                if drawdown_ratio > drawdown_threshold:
                    drawdown_penalty_coeff = self.get_setting_float(
                        "drawdown_penalty_coeff", 0.05
                    )
                    return drawdown_ratio * drawdown_penalty_coeff

        return 0.0

    def calculate_stagnation_penalty(
        self, portfolio_value_history: List[float]
    ) -> float:
        """Calculate stagnation penalty (when portfolio isn't growing)."""
        stagnation_window = self.get_setting_int("stagnation_window", 30)
        if len(portfolio_value_history) < stagnation_window:
            return 0.0

        recent_values = portfolio_value_history[-stagnation_window:]
        initial_value = recent_values[0]
        final_value = recent_values[-1]

        if initial_value > 0:
            growth_rate = (final_value - initial_value) / initial_value
            stagnation_threshold = self.get_setting_float(
                "stagnation_threshold", -0.005
            )

            if growth_rate < stagnation_threshold:
                stagnation_penalty_max = self.get_setting_float(
                    "stagnation_penalty_max", 0.02
                )
                return min(
                    stagnation_penalty_max,
                    abs(growth_rate - stagnation_threshold) * 0.5,
                )

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
            win_streak_bonus_per_win = self.get_setting_float(
                "win_streak_bonus_per_win", 0.01
            )
            return win_count * win_streak_bonus_per_win

        return 0.0

    def reset(self) -> None:
        """Reset internal state."""
        self._action_counts = [0, 0, 0]
        self._consecutive_idle_steps = 0
        self._consecutive_position_hold_steps = 0
        self._win_count = 0
        self._loss_count = 0
        self._recent_actions = []

        # Reset signal integration stats (v436 enhancement)
        if self.signal_integration is not None:
            self.signal_integration.reset_stats()

    def _apply_signal_integration(
        self,
        reward: float,
        observation: Optional[np.ndarray],
        action: int,
        step: int,
    ) -> float:
        """
        Apply signal-based reward integration.

        Args:
            reward: Base reward value
            observation: Current observation
            action: Action taken
            step: Current step

        Returns:
            Modified reward with signal integration
        """
        return self.signal_integrator.integrate_signal(
            reward, observation, action, step
        )

    def _convert_continuous_to_discrete_action(
        self, action: Union[float, np.ndarray]
    ) -> int:
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
            action_threshold_sell = self.get_setting_float(
                "action_threshold_sell", -0.2
            )

            if action < action_threshold_sell:
                discrete_action = ACTION_SELL  # Strong sell signal
            elif action > action_threshold_buy:
                discrete_action = ACTION_BUY  # Strong buy signal
            else:
                discrete_action = ACTION_HOLD  # Hold/weak signal

            self.logger.debug(
                f"Continuous action {action:.3f} converted to discrete action {discrete_action}"
            )
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

    def _calculate_position_size_bonus(
        self, position: float, old_position: float = 0.0
    ) -> float:
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

    def _calculate_drawdown_penalty(
        self, portfolio_value: float, portfolio_value_history: List[float]
    ) -> float:
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
            peak_value = (
                max(portfolio_value_history[-20:])
                if len(portfolio_value_history) >= 20
                else max(portfolio_value_history)
            )
            current_drawdown = (peak_value - portfolio_value) / max(peak_value, 1.0)

            if current_drawdown > 0.05:  # 5% drawdown threshold
                drawdown_penalty = -current_drawdown * 100

        return drawdown_penalty

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
                "description": "HOLD with small profit",
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
                "description": "BUY with profit",
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
                "description": "SELL with loss",
            },
        ]

        self.logger.debug("Testing reward calculation...")
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
                portfolio_value_history=[case["portfolio_value"]] * 30,
            )
            self.logger.debug(
                f"Test {i+1}: {case['description']} -> Reward: {reward:.4f}"
            )
            self.logger.debug(
                f"  Components: pnl={case['pnl']:.2f}, position={case['position']:.4f}, atr={case['atr']:.2f}"
            )

    def _calculate_stability_optimized_reward(
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
        observation: Optional[np.ndarray],
        step: int,
    ) -> float:
        """
        Stability-optimized reward with enhanced balance mechanisms and market regime adaptation.

        Focuses on action balance, entropy regularization, and consistency penalties
        to achieve sustainable trading behavior. Now includes market regime detection
        for adaptive behavior optimization.
        """
        # Update action counts
        self._action_counts[action] += 1
        total_actions = sum(self._action_counts)

        # Get market regime if enabled
        market_regime = None
        regime_adaptive_params = None
        if (
            hasattr(self.config, "market_regime")
            and self.config.market_regime
            and self.config.market_regime.get("enabled", False)
        ):
            # Detect current market regime
            market_regime = self.market_regime_detector.detect_regime(
                current_price, step
            )

            # Get regime-adaptive parameters with normalized keys
            regime_config = self.config.market_regime.get(
                "regime_adaptive_behavior", {}
            )
            # Normalize regime keys: bull -> bull_market, bear -> bear_market, etc.
            regime_key_map = {
                "bull": "bull_market",
                "bear": "bear_market",
                "sideways": "sideways_market",
                "volatile": "volatile_market",
            }
            normalized_regime_key = regime_key_map.get(market_regime, market_regime)

            if normalized_regime_key in regime_config:
                regime_adaptive_params = regime_config[normalized_regime_key]

        # Enhanced balance mechanism with action_balance_target (regime-adaptive)
        balance_penalty = 0.0
        if total_actions >= 10:
            action_ratios = [count / total_actions for count in self._action_counts]

            # Use regime-adaptive action_balance_target if available
            if regime_adaptive_params:
                buy_sell_target = regime_adaptive_params.get(
                    "action_balance_target", self.action_balance_target
                )
            else:
                buy_sell_target = self.action_balance_target

            hold_target = 1.0 - 2 * buy_sell_target
            target_ratios = [
                hold_target,
                buy_sell_target,
                buy_sell_target,
            ]  # [HOLD, BUY, SELL]

            tolerance = self.get_setting_float("balance_penalty_tolerance", 0.05)

            # Use regime-adaptive balance_penalty if available
            if regime_adaptive_params:
                penalty_weight = regime_adaptive_params.get(
                    "balance_penalty", self.balance_penalty
                )
            else:
                penalty_weight = self.balance_penalty

            for i, ratio in enumerate(action_ratios):
                deviation = abs(ratio - target_ratios[i])
                if deviation > tolerance:
                    excess_deviation = deviation - tolerance
                    balance_penalty += penalty_weight * excess_deviation

        # Calculate base reward
        base_reward = self._calculate_base_reward(
            action,
            atr_normalised,
            portfolio_return,
            position,
            effective_max_position,
            current_price,
            atr,
            pnl,
            observation,
        )

        # Profit/loss based reward adjustment
        profit_multiplier = self.get_setting_float("profit_multiplier", 1.5)
        loss_penalty_multiplier = self.get_setting_float("loss_penalty_multiplier", 1.2)

        pnl_normalizer = atr * effective_max_position * current_price
        normalized_pnl = pnl / max(pnl_normalizer, 1e-8)

        if pnl > 0:
            profit_bonus = normalized_pnl * profit_multiplier
            base_reward += profit_bonus
        elif pnl < 0:
            loss_penalty = abs(normalized_pnl) * loss_penalty_multiplier
            base_reward -= loss_penalty

        # Moderate HOLD penalty
        hold_penalty_rate = self.get_setting_float("hold_penalty_rate", 0.05)
        if action == ACTION_HOLD:
            hold_penalty = (
                hold_penalty_rate * abs(position) / max(effective_max_position, 0.01)
            )
            base_reward -= hold_penalty

        # Moderate trading bonuses
        if action in [ACTION_BUY, ACTION_SELL]:
            trading_bonus = self.get_setting_float("trading_bonus", 0.005)
            base_reward += trading_bonus

        # Entropy regularization to encourage action diversity (regime-adaptive)
        entropy_bonus = 0.0
        entropy_reg = self.entropy_regularization
        if regime_adaptive_params:
            entropy_reg = regime_adaptive_params.get(
                "entropy_regularization", entropy_reg
            )

        if entropy_reg > 0.0 and total_actions >= 20:
            action_probs = np.array(action_ratios)
            entropy = -np.sum(action_probs * np.log(action_probs + 1e-8))
            max_entropy = np.log(3.0)  # 3 actions
            entropy_ratio = entropy / max_entropy
            # Only apply positive entropy bonus, avoid penalties
            if entropy_ratio > 0.5:
                entropy_bonus = entropy_reg * (entropy_ratio - 0.5)
            base_reward += entropy_bonus

        # Consistency penalty for repetitive actions (regime-adaptive)
        consistency_penalty_val = self.consistency_penalty
        if regime_adaptive_params:
            consistency_penalty_val = regime_adaptive_params.get(
                "consistency_penalty", consistency_penalty_val
            )

        if len(self._recent_actions) >= 5:
            recent_actions = self._recent_actions[-10:]  # Last 10 actions
            if len(recent_actions) >= 5:
                # Calculate action consistency (how often the same action is repeated)
                most_common_action = max(set(recent_actions), key=recent_actions.count)
                consistency_ratio = recent_actions.count(most_common_action) / len(
                    recent_actions
                )

                if consistency_ratio > 0.7:  # If same action > 70% of recent actions
                    excess_consistency = consistency_ratio - 0.7
                    consistency_penalty = consistency_penalty_val * excess_consistency
                    base_reward -= consistency_penalty

        # Action smoothing penalty for abrupt changes
        if len(self._recent_actions) >= 3:
            recent_changes = sum(
                1
                for i in range(1, len(self._recent_actions[-5:]))
                if self._recent_actions[-i] != self._recent_actions[-i - 1]
            )
            smoothing_penalty = self.action_smoothing * (
                recent_changes / 4.0
            )  # Normalize by max changes
            base_reward -= smoothing_penalty

        final_reward = base_reward - balance_penalty

        self.logger.debug(
            f"Stability optimized: base_reward={base_reward:.3f}, balance_penalty={balance_penalty:.3f}, "
            f"pnl={pnl:.3f}, entropy_bonus={entropy_bonus:.3f}, "
            f"consistency_penalty={consistency_penalty if 'consistency_penalty' in locals() else 0:.3f}, "
            f"market_regime={market_regime}, final_reward={final_reward:.3f}"
        )
        return final_reward * reward_scaling

    def _calculate_backtest_optimization_reward(
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
        observation: Optional[np.ndarray],
        step: int,
        portfolio_value_delta: float,
    ) -> float:
        """Backtest optimization stage: Direct PnL-focused reward for correlation improvement."""
        eps = self.get_setting_float("eps", 1e-8)

        # Portfolio value delta reward - positive for increases, negative for decreases
        # Use percentage change instead of absolute change for better correlation
        min_portfolio_value = 1e-6  # Prevent division by very small numbers
        if abs(self._previous_portfolio_value) > min_portfolio_value:
            portfolio_percentage_change = portfolio_value_delta / abs(
                self._previous_portfolio_value
            )
            # Clip extreme percentage changes to prevent reward explosion - extremely tight clipping for maximum stability
            portfolio_percentage_change = np.clip(
                portfolio_percentage_change, -0.001, 0.001
            )
            portfolio_reward = portfolio_percentage_change * self.get_setting_float(
                "portfolio_scaling", 100.0
            )
        else:
            portfolio_reward = 0.0

        # Position size bonus for active trading
        position_size_bonus = abs(position) * self.get_setting_float(
            "position_size_bonus", 0.1
        )

        # Action penalty to prevent excessive trading
        action_penalty = (
            self.get_setting_float("action_penalty", 0.01) if action != 0 else 0.0
        )

        # Combine rewards
        total_reward = portfolio_reward + position_size_bonus - action_penalty

        # Scale and clip
        total_reward *= reward_scaling

        self.logger.debug(
            f"Backtest optimization: portfolio_delta={portfolio_value_delta:.6f}, portfolio_pct_change={portfolio_percentage_change:.6f}, portfolio_reward={portfolio_reward:.3f}, "
            f"position_bonus={position_size_bonus:.3f}, action_penalty={action_penalty:.3f}, "
            f"total_reward={total_reward:.3f}"
        )

        return total_reward

    def get_current_regime(self, current_price: float, step: int) -> str:
        """Get the current market regime for external diagnostics."""
        if self.market_regime_detector:
            return self.market_regime_detector.detect_regime(current_price, step)
        return "unknown"
