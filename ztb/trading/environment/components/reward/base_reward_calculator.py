"""
Base Reward Calculator - Core reward calculation logic.

This module contains the base reward calculator class with common functionality
and component initialization.
"""

from typing import Any, List, Optional
from collections import deque

import numpy as np

from ztb.trading.constants import (
    ACTION_BUY,
    ACTION_HOLD,
    ACTION_SELL,
    MULTIPLIER_INDEX_BUY,
    MULTIPLIER_INDEX_HOLD,
    MULTIPLIER_INDEX_SELL,
)
from ztb.trading.environment.utils.config import EnvironmentConfig, RewardSettings
from ztb.utils.logging_utils import get_logger

from ..asymmetric_reward_scaler import AsymmetricRewardScaler
from ..dynamic_reward_shaper import DynamicRewardShaper
from ztb.trading.strategies.action_signal_guide.components.market_regime import MarketRegimeDetector
from ..signal_integrator import SignalIntegrator


class BaseRewardCalculator:
    """
    Base class for reward calculation with curriculum learning stages.

    This class orchestrates reward calculation using specialized components:
    - MarketRegimeDetector: Detects market regimes
    - DynamicRewardShaper: Applies market-aware reward shaping
    - SignalIntegrator: Integrates signal-based rewards
    - AsymmetricRewardScaler: Applies position-based scaling

    Follows SOLID principles for maintainability and testability.
    """

    def __init__(
        self,
        config: EnvironmentConfig,
        reward_settings: RewardSettings,
        initial_portfolio_value: float,
    ):
        """
        Initialize BaseRewardCalculator with component-based architecture.

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
        self._action_counts: List[int] = [0, 0, 0]  # [BUY, SELL, HOLD]
        self._consecutive_idle_steps = 0
        self._consecutive_position_hold_steps = 0
        self._win_count = 0
        self._loss_count = 0
        self._recent_actions = deque(maxlen=100)  # Track recent actions for frequency penalty
        self.last_signal_strength: float = 0.0
        self.last_signal_reward: float = 0.0
        self._previous_portfolio_value = initial_portfolio_value  # Track previous portfolio value for delta calculation

        # Initialize components following SOLID principles
        self._initialize_components()

    def _initialize_components(self):
        """Initialize all reward calculation components."""
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

        self.market_regime_detector = MarketRegimeDetector()

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
            config=self.config,
            enabled=signal_guide_enabled,
            guidance_level=guidance_level_str,
            signal_bonus_weight=signal_bonus_weight,
            signal_penalty_weight=signal_penalty_weight,
            granville_weight=granville_weight,
            dow_theory_weight=dow_theory_weight,
            heikin_ashi_weight=heikin_ashi_weight,
            enable_advanced_integration=enable_advanced_integration,
        )

        # 4. Asymmetric Reward Scaler
        long_position_reward_multiplier = self.get_setting_float(
            "asymmetric_reward_scaling.long_position_reward_multiplier", 1.3
        )
        short_position_reward_multiplier = self.get_setting_float(
            "asymmetric_reward_scaling.short_position_reward_multiplier", 0.7
        )
        long_position_penalty_multiplier = self.get_setting_float(
            "asymmetric_reward_scaling.long_position_penalty_multiplier", 0.9
        )
        short_position_penalty_multiplier = self.get_setting_float(
            "asymmetric_reward_scaling.short_position_penalty_multiplier", 0.95
        )

        self.asymmetric_reward_scaler = AsymmetricRewardScaler(
            long_position_reward_multiplier=long_position_reward_multiplier,
            short_position_reward_multiplier=short_position_reward_multiplier,
            long_position_penalty_multiplier=long_position_penalty_multiplier,
            short_position_penalty_multiplier=short_position_penalty_multiplier,
        )

    def _resolve_setting(self, key: str, default: Any) -> Any:
        """Resolve nested reward setting keys, including custom params."""
        if not key or self.reward_settings is None:
            return default

        parts = key.split(".")
        current: Any = self.reward_settings

        for part in parts:
            if isinstance(current, RewardSettings):
                if hasattr(current, part):
                    current = getattr(current, part)
                    continue
                custom_params = getattr(current, "custom_reward_params", {})
                if isinstance(custom_params, dict) and part in custom_params:
                    current = custom_params[part]
                    continue
                return default
            if isinstance(current, dict):
                if part in current:
                    current = current[part]
                    continue
                return default
            return default

        return default if current is None else current

    def get_setting_bool(self, key: str, default: bool = False) -> bool:
        """Get boolean setting with fallback."""
        value = self._resolve_setting(key, default)
        if isinstance(value, bool):
            return value
        if isinstance(value, (int, float)):
            return bool(value)
        if isinstance(value, str):
            normalized = value.strip().lower()
            if normalized in {"true", "1", "yes", "y", "on"}:
                return True
            if normalized in {"false", "0", "no", "n", "off"}:
                return False
        return bool(value)

    def get_setting_int(self, key: str, default: int = 0) -> int:
        """Get integer setting with fallback."""
        value = self._resolve_setting(key, default)
        try:
            return int(value)
        except (TypeError, ValueError):
            return default

    def get_setting_float(self, key: str, default: float = 0.0) -> float:
        """Get float setting with fallback."""
        value = self._resolve_setting(key, default)
        try:
            return float(value)
        except (TypeError, ValueError):
            return default

    def get_setting_str(self, key: str, default: str = "") -> str:
        """Get string setting with fallback."""
        value = self._resolve_setting(key, default)
        try:
            return str(value)
        except (TypeError, ValueError):
            return default

    def calculate_reward(
        self,
        action: int,
        observation: Optional[np.ndarray],
        reward: float,
        done: bool,
        info: dict,
        step: int,
    ) -> float:
        """
        Main reward calculation entry point.

        This method should be overridden by subclasses to implement specific reward strategies.
        """
        raise NotImplementedError("Subclasses must implement calculate_reward")

    def update_action_counts(self, action: int):
        """Update action tracking statistics."""
        if action == ACTION_BUY:
            self._action_counts[MULTIPLIER_INDEX_BUY] += 1
        elif action == ACTION_SELL:
            self._action_counts[MULTIPLIER_INDEX_SELL] += 1
        elif action == ACTION_HOLD:
            self._action_counts[MULTIPLIER_INDEX_HOLD] += 1

        # Track recent actions
        self._recent_actions.append(action)
        if len(self._recent_actions) > 100:  # Keep last 100 actions
            self._recent_actions.pop(0)

    def update_win_loss_counts(self, pnl: float):
        """Update win/loss tracking."""
        if pnl > 0:
            self._win_count += 1
        elif pnl < 0:
            self._loss_count += 1

    def reset_episode_state(self):
        """Reset episode-specific state."""
        self._action_counts = [0, 0, 0]  # [BUY, SELL, HOLD]
        self._consecutive_idle_steps = 0
        self._consecutive_position_hold_steps = 0
        self._win_count = 0
        self._loss_count = 0
        self._recent_actions.clear()
        self.last_signal_strength = 0.0
        self.last_signal_reward = 0.0
        self._previous_portfolio_value = self.initial_portfolio_value
