"""
Action Signal Guide - Training Wheels for RL Trading Agents

This module provides classical technical signal guidance to help reinforcement
learning agents learn basic trading patterns before discovering novel strategies.
"""

from typing import Any
import numpy as np
from enum import Enum

from ztb.trading.constants import ACTION_BUY, ACTION_HOLD, ACTION_SELL

from .signal_definitions import SignalDefinitions, SignalType
from ztb.utils.logging_utils import get_logger

class GuidanceMode(Enum):
    """Modes of signal guidance."""
    FULL_GUIDANCE = "full"      # Strong guidance for early training
    PARTIAL_GUIDANCE = "partial" # Moderate guidance
    MINIMAL_GUIDANCE = "minimal" # Light guidance for advanced training
    FADE_OUT = "fade_out"       # Guidance that fades out over time
    NO_GUIDANCE = "none"        # No guidance (pure RL)

class TimeFrame(Enum):
    """Time frames for signal evaluation."""
    SHORT = "short"     # Short-term signals (5-15 periods)
    MEDIUM = "medium"   # Medium-term signals (15-60 periods)
    LONG = "long"       # Long-term signals (60+ periods)

class ActionSignalGuide:
    """
    Provides classical technical signal guidance for trading actions.

    This serves as training wheels for RL agents, helping them learn basic
    BUY/SELL patterns from technical analysis before discovering novel strategies.
    """

    def __init__(self,
                 mode: GuidanceMode = GuidanceMode.FULL_GUIDANCE,
                 signal_weight: float = 1.0,
                 guidance_decay: float = 0.95,
                 feature_names: list[str] | None = None,
                 multi_timeframe: bool = True) -> None:
        """
        Initialize the action signal guide.

        Args:
            mode: Level of guidance to provide
            signal_weight: Weight multiplier for signal strength
            guidance_decay: Decay factor for guidance over time (0.0-1.0)
            feature_names: Names of features in observation vector
            multi_timeframe: Whether to use multi-timeframe signal evaluation
        """
        self.logger = get_logger("ActionSignalGuide")
        self.mode = mode
        self.signal_weight = signal_weight
        self.guidance_decay = guidance_decay
        self.feature_names = feature_names or []
        self.multi_timeframe = multi_timeframe

        # Initialize signal definitions
        self.signal_definitions = SignalDefinitions()

        # Setup multi-timeframe signal mappings if enabled
        if self.multi_timeframe:
            self._setup_timeframe_signals()

        # Guidance parameters based on mode
        self._setup_guidance_parameters()

        self.logger.info(f"Initialized ActionSignalGuide with mode: {mode.value}")

    def update_signal_confidence(self, observation: np.ndarray, action: int, reward: float) -> None:
        """
        Update signal confidence based on action outcomes.

        Args:
            observation: Market observation
            action: Action taken
            reward: Reward received (positive for good actions)
        """
        if not hasattr(self, 'signal_confidence'):
            self.signal_confidence = {}

        # Get active signals for this observation
        active_signals = []
        for signal_name, signal_config in self.signal_definitions.signals.items():
            signal_func = signal_config["function"]
            try:
                if signal_func(observation, self.feature_names):
                    active_signals.append(signal_name)
            except Exception:
                continue

        # Update confidence for active signals
        confidence_change = 0.01 if reward > 0 else -0.005  # Small positive/negative updates

        for signal_name in active_signals:
            if signal_name not in self.signal_confidence:
                self.signal_confidence[signal_name] = 1.0

            self.signal_confidence[signal_name] = np.clip(
                self.signal_confidence[signal_name] + confidence_change,
                0.1, 2.0  # Confidence bounds
            )

    def get_adaptive_signal_strength(self,
                                   observation: np.ndarray,
                                   action: int,
                                   step: int = 0,
                                   use_multi_timeframe: bool = True) -> float:
        """
        Get signal strength with adaptive confidence adjustments.

        Args:
            observation: Current market observation
            action: Action taken
            step: Training step
            use_multi_timeframe: Whether to use multi-timeframe analysis

        Returns:
            Adaptively adjusted signal strength
        """
        # Get base signal strength
        if use_multi_timeframe and self.multi_timeframe:
            base_strength = self.get_multi_timeframe_signal_strength(observation, action, step)
        else:
            base_strength = self.get_signal_strength(observation, action, step)

        # Apply confidence adjustments
        if hasattr(self, 'signal_confidence') and self.signal_confidence:
            confidence_multiplier = 1.0

            # Get active signals and apply their confidence
            for signal_name, signal_config in self.signal_definitions.signals.items():
                signal_func = signal_config["function"]
                try:
                    if signal_func(observation, self.feature_names):
                        signal_type = signal_config["type"]
                        if ((action == ACTION_BUY and signal_type == SignalType.BUY) or
                            (action == ACTION_SELL and signal_type == SignalType.SELL)):
                            confidence = self.signal_confidence.get(signal_name, 1.0)
                            confidence_multiplier *= confidence
                except Exception:
                    continue

            # Apply confidence adjustment (with dampening)
            confidence_adjustment = (confidence_multiplier - 1.0) * 0.1 + 1.0
            base_strength *= confidence_adjustment

        return np.clip(base_strength, 0.0, self.max_signal_strength)

    def _setup_guidance_parameters(self) -> None:
        """Setup guidance parameters based on current mode."""
        if self.mode == GuidanceMode.FULL_GUIDANCE:
            self.signal_threshold = 0.3  # Lower threshold for more guidance
            self.max_signal_strength = 1.0
            # Use instance guidance_decay if set, otherwise default
            if not hasattr(self, 'guidance_decay') or self.guidance_decay is None:
                self.guidance_decay = 0.95  # Slow decay
        elif self.mode == GuidanceMode.PARTIAL_GUIDANCE:
            self.signal_threshold = 0.5
            self.max_signal_strength = 0.8
            if not hasattr(self, 'guidance_decay') or self.guidance_decay is None:
                self.guidance_decay = 0.9
        elif self.mode == GuidanceMode.MINIMAL_GUIDANCE:
            self.signal_threshold = 0.7
            self.max_signal_strength = 0.5
            if not hasattr(self, 'guidance_decay') or self.guidance_decay is None:
                self.guidance_decay = 0.8
        elif self.mode == GuidanceMode.FADE_OUT:
            self.signal_threshold = 0.3  # Start with strong guidance
            self.max_signal_strength = 1.0
            # Use configured guidance_decay, default to 0.95 if not set
            if not hasattr(self, 'guidance_decay') or self.guidance_decay is None:
                self.guidance_decay = 0.95
        else:  # NO_GUIDANCE
            self.signal_threshold = 1.0  # Effectively disabled
            self.max_signal_strength = 0.0
            self.guidance_decay = 1.0

    def _setup_timeframe_signals(self) -> None:
        """Setup signal mappings for different timeframes."""
        # Define which signals are appropriate for each timeframe
        self.timeframe_signals = {
            TimeFrame.SHORT: [
                # Short-term signals (fast indicators)
                "rsi_oversold", "rsi_overbought",
                "stoch_oversold", "stoch_overbought",
                "williams_r_oversold", "williams_r_overbought",
                "cci_oversold", "cci_overbought",
                "bollinger_lower_touch", "bollinger_upper_touch"
            ],
            TimeFrame.MEDIUM: [
                # Medium-term signals (balanced indicators)
                "macd_bullish", "macd_bearish",
                "golden_cross", "death_cross",
                "plus_di_bullish", "minus_di_bearish",
                "adx_strong_trend"
            ],
            TimeFrame.LONG: [
                # Long-term signals (slow indicators)
                "range_bound", "low_volatility",
                "adx_strong_trend"
            ]
        }

        # Weights for combining signals from different timeframes
        self.timeframe_weights = {
            TimeFrame.SHORT: 0.4,   # 40% weight for short-term
            TimeFrame.MEDIUM: 0.4,  # 40% weight for medium-term
            TimeFrame.LONG: 0.2     # 20% weight for long-term
        }

        self.logger.info("Multi-timeframe signal evaluation enabled")

    def get_system_status(self) -> dict:
        """Return a compact system status summary used by scripts/tests."""
        return {
            "status": "ok",
            "mode": self.mode.value if hasattr(self, "mode") else None,
            "num_signals": len(getattr(self, "signal_definitions", {}).signals)
            if hasattr(self, "signal_definitions")
            else 0,
        }

    def set_feature_names(self, feature_names: list[str] | None) -> None:
        """set the feature names for signal evaluation."""
        self.feature_names = feature_names
        if feature_names is not None:
            self.logger.debug(f"set feature names: {len(feature_names)} features")
        else:
            self.logger.debug("Feature names set to None")

    def get_signal_strength(self,
                          observation: np.ndarray,
                          action: int,
                          step: int = 0) -> float:
        """
        Get signal strength for a given observation and action.

        Args:
            observation: Current market observation
            action: Action taken (0=HOLD, 1=BUY, 2=SELL)
            step: Current training step (for decay calculation)

        Returns:
            Signal strength multiplier (0.0 to 1.0)
        """
        if self.mode == GuidanceMode.NO_GUIDANCE or not self.feature_names:
            return 0.0

        # Evaluate all relevant signals
        buy_signals = self._evaluate_signals_for_action(observation, SignalType.BUY)
        sell_signals = self._evaluate_signals_for_action(observation, SignalType.SELL)

        # Calculate signal strength based on action
        if action == ACTION_BUY:  # BUY action
            signal_strength = buy_signals  # Direct BUY signal strength
        elif action == ACTION_SELL:  # SELL action
            # For SELL action, give positive strength if there are SELL signals,
            # regardless of BUY signals strength
            signal_strength = sell_signals  # Direct SELL signal strength
        else:  # HOLD action
            # HOLD gets weak positive signal if no strong BUY/SELL signals
            max_signal = max(buy_signals, sell_signals)
            signal_strength = max(0.0, 0.3 - max_signal)  # Weak signal if no strong directional signals

        # Apply guidance decay based on training progress
        decay_factor = self.guidance_decay ** (step / 10000)  # Decay over 10k steps
        signal_strength *= decay_factor

        # Apply signal weight and clamp
        signal_strength = min(signal_strength * self.signal_weight, self.max_signal_strength)

        return signal_strength

    def get_multi_timeframe_signal_strength(self,
                                          observation: np.ndarray,
                                          action: int,
                                          step: int = 0) -> float:
        """
        Get signal strength using multi-timeframe analysis.

        Args:
            observation: Current market observation
            action: Action taken (0=HOLD, 1=BUY, 2=SELL)
            step: Current training step (for decay calculation)

        Returns:
            Combined signal strength from multiple timeframes (0.0 to 1.0)
        """
        if self.mode == GuidanceMode.NO_GUIDANCE or not self.feature_names or not self.multi_timeframe:
            return self.get_signal_strength(observation, action, step)

        combined_strength = 0.0

        # Evaluate signals for each timeframe
        for timeframe, signals in self.timeframe_signals.items():
            timeframe_strength = self._evaluate_timeframe_signals(observation, action, signals)
            combined_strength += timeframe_strength * self.timeframe_weights[timeframe]

        # Apply guidance decay based on training progress
        decay_factor = self.guidance_decay ** (step / 10000)
        combined_strength *= decay_factor

        # Apply signal weight and clamp
        combined_strength = min(combined_strength * self.signal_weight, self.max_signal_strength)

        return combined_strength

    def _evaluate_timeframe_signals(self,
                                  observation: np.ndarray,
                                  action: int,
                                  signal_names: list[str]) -> float:
        """
        Evaluate signals for a specific timeframe.

        Args:
            observation: Current market observation
            action: Action taken
            signal_names: list of signal names to evaluate for this timeframe

        Returns:
            Signal strength for this timeframe
        """
        buy_signals = []
        sell_signals = []

        # Evaluate each signal in this timeframe
        for signal_name in signal_names:
            if signal_name in self.signal_definitions.signals:
                signal_config = self.signal_definitions.signals[signal_name]
                signal_func = signal_config["function"]

                try:
                    if signal_func(observation, self.feature_names):
                        signal_type = signal_config["type"]
                        signal_strength = signal_config["strength"].value

                        if signal_type == SignalType.BUY:
                            buy_signals.append(signal_strength)
                        elif signal_type == SignalType.SELL:
                            sell_signals.append(signal_strength)
                except Exception as e:
                    self.logger.debug(f"Error evaluating signal {signal_name}: {e}")
                    continue

        # Calculate timeframe signal strength based on action
        if action == ACTION_BUY:
            return max(buy_signals) if buy_signals else 0.0
        elif action == ACTION_SELL:
            return max(sell_signals) if sell_signals else 0.0
        else:  # HOLD
            max_signal = max(max(buy_signals or [0.0]), max(sell_signals or [0.0]))
            return max(0.0, 0.3 - max_signal)

    def get_action_recommendation(self, observation: np.ndarray) -> tuple[int, float]:
        """
        Get recommended action based on signal strength.

        Args:
            observation: Current market observation

        Returns:
            tuple of (recommended_action, confidence)
        """
        if not self.feature_names:
            return 0, 0.0  # Default to HOLD

        buy_strength = self._evaluate_signals_for_action(observation, SignalType.BUY)
        sell_strength = self._evaluate_signals_for_action(observation, SignalType.SELL)

        if buy_strength > sell_strength and buy_strength > self.signal_threshold:
            return ACTION_BUY, buy_strength  # BUY
        elif sell_strength > buy_strength and sell_strength > self.signal_threshold:
            return ACTION_SELL, sell_strength  # SELL
        else:
            return ACTION_HOLD, max(buy_strength, sell_strength)  # HOLD

    def get_multi_timeframe_action_recommendation(self, observation: np.ndarray) -> tuple[int, float]:
        """
        Get recommended action using multi-timeframe signal analysis.

        Args:
            observation: Current market observation

        Returns:
            tuple of (recommended_action, confidence)
        """
        if not self.feature_names or not self.multi_timeframe:
            return self.get_action_recommendation(observation)

        buy_strength_total = 0.0
        sell_strength_total = 0.0

        # Evaluate signals across all timeframes
        for timeframe, signals in self.timeframe_signals.items():
            buy_strength = self._evaluate_timeframe_signals(observation, ACTION_BUY, signals)
            sell_strength = self._evaluate_timeframe_signals(observation, ACTION_SELL, signals)

            weight = self.timeframe_weights[timeframe]
            buy_strength_total += buy_strength * weight
            sell_strength_total += sell_strength * weight

        # Determine recommendation based on combined strengths
        if buy_strength_total > sell_strength_total and buy_strength_total > self.signal_threshold:
            return ACTION_BUY, buy_strength_total
        elif sell_strength_total > buy_strength_total and sell_strength_total > self.signal_threshold:
            return ACTION_SELL, sell_strength_total
        else:
            return ACTION_HOLD, max(buy_strength_total, sell_strength_total)

    def _evaluate_signals_for_action(self,
                                   observation: np.ndarray,
                                   signal_type: SignalType) -> float:
        """
        Evaluate all signals of a given type and return combined strength.

        Args:
            observation: Current market observation
            signal_type: Type of signals to evaluate

        Returns:
            Combined signal strength (0.0 to 1.0)
        """
        signal_names = self.signal_definitions.get_signals_by_type(signal_type)
        total_strength = 0.0
        active_signals = 0

        for signal_name in signal_names:
            sig_type, strength = self.signal_definitions.evaluate_signal(
                signal_name, observation, self.feature_names
            )

            if sig_type == signal_type and strength > 0.0:
                total_strength += strength
                active_signals += 1

        # Return average strength if any signals are active
        if active_signals > 0:
            return min(total_strength / active_signals, 1.0)
        else:
            return 0.0

    def update_guidance_mode(self, mode: GuidanceMode) -> None:
        """Update the guidance mode."""
        self.mode = mode
        self._setup_guidance_parameters()
        self.logger.info(f"Updated guidance mode to: {mode.value}")

    def get_guidance_stats(self) -> dict[str, Any]:
        """Get current guidance statistics."""
        return {
            "mode": self.mode.value,
            "signal_weight": self.signal_weight,
            "signal_threshold": self.signal_threshold,
            "max_signal_strength": self.max_signal_strength,
            "guidance_decay": self.guidance_decay,
            "num_features": len(self.feature_names),
            "available_signals": len(self.signal_definitions.get_signal_names())
        }

    def reset(self) -> None:
        """Reset the signal guide state."""
        self.logger.debug("Reset ActionSignalGuide")
        # Any state reset logic would go here
