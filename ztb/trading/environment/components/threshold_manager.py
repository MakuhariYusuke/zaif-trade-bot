"""
Threshold Manager - Unified threshold management for trading actions and signals.

This module provides a comprehensive threshold management system that handles:
- Action discretization thresholds (continuous to discrete conversion)
- Adaptive threshold adjustment based on market volatility
- Market regime detection and signal threshold adaptation
- Performance-based threshold optimization

The ThresholdManager serves as the single source of truth for all threshold-related
logic in the trading system, eliminating code duplication and ensuring consistent
behavior across ActionSignalGuide and backtesting components.
"""

import logging
from collections import deque
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from ztb.trading.strategies.action_signal_guide.components.market_regime import (
    MarketRegimeDetector,
)
from ztb.types.common import ConfigDict
from ztb.utils.data.outlier_detection import calculate_z_score_single

logger = logging.getLogger(__name__)


class ThresholdManager:
    """
    Unified threshold management system for trading actions and signals.

    This class provides comprehensive threshold management capabilities:

    **Core Features:**
    - Action discretization: Converts continuous actions to discrete trades
    - Adaptive thresholds: Adjusts based on market volatility
    - Market regime detection: Identifies trending, ranging, and volatile conditions
    - Signal threshold adaptation: Optimizes signal confidence and strength thresholds
    - Performance tracking: Learns from signal execution results

    **Market Regimes:**
    - trending_bull: Strong upward price movement
    - trending_bear: Strong downward price movement
    - ranging: Sideways price movement with low volatility
    - volatile: High volatility regardless of trend direction

    **Usage Examples:**
    ```python
    # Basic action threshold
    threshold = manager.get_threshold(volatility=0.02, current_price=100.0)

    # Market regime detection
    regime = manager.detect_market_regime(market_data)

    # Adaptive signal thresholds
    thresholds = manager.calculate_adaptive_signal_thresholds(
        market_data, base_confidence=0.7, base_strength=0.4
    )

    # Performance update
    manager.update_performance({"profitable": True, "pnl": 0.05})
    ```
    """

    def __init__(self, config: ConfigDict) -> None:
        """
        Initialize the ThresholdManager.

        Args:
            config: Configuration object containing threshold settings.
        """
        self.config = config

        # Base threshold from config (legacy parameter name)
        self.base_threshold = getattr(config, "continuous_to_discrete_threshold", 0.01)

        # Adaptive settings
        self.adaptive_mode = getattr(config, "adaptive_threshold_mode", False)
        self.volatility_multiplier = getattr(
            config, "threshold_volatility_multiplier", 1.0
        )
        self.min_threshold = getattr(config, "min_action_threshold", 0.001)
        self.max_threshold = getattr(config, "max_action_threshold", 1.0)

        # Dynamic Thresholding (Z-Score) settings
        self.dynamic_threshold_mode = getattr(
            config, "dynamic_threshold_mode", "fixed"
        )  # fixed, volatility, z_score
        self.z_score_window = getattr(config, "z_score_window", 100)
        self.z_score_threshold = getattr(config, "z_score_threshold", 2.0)
        self.z_score_method = getattr(config, "z_score_method", "std")  # std or mad
        self.action_history = deque(maxlen=self.z_score_window)
        self.min_std = 1e-6  # Prevent division by zero

        # State tracking
        self.last_threshold = self.base_threshold
        self.last_volatility = 0.0
        self.z_score_trigger_count = 0

        # Market regime detection parameters
        self.regime_window = getattr(config, "regime_detection_window", 50)
        self.adaptation_rate = getattr(config, "threshold_adaptation_rate", 0.1)
        self.performance_memory = getattr(config, "performance_memory_size", 100)

        # Performance tracking for signal thresholds
        self.signal_history: List[Dict[str, Any]] = []

        # Market regime detector for advanced analysis
        regime_config = getattr(config, "regime_detection_config", {}) or {}
        regime_use_relative = bool(regime_config.get("use_relative", False))
        regime_ref_window = int(regime_config.get("reference_window", 1000))
        regime_percentile = float(regime_config.get("percentile_threshold", 0.8))
        self.regime_detector = MarketRegimeDetector(
            use_relative=regime_use_relative,
            reference_window=regime_ref_window,
            percentile_threshold=regime_percentile,
        )

        # Caching for performance optimization
        self._regime_cache: Optional[Dict[str, Any]] = None
        self._threshold_cache: Optional[Dict[str, Any]] = None

        # Regime detection parameters (legacy compatibility)
        self.trend_threshold = getattr(config, "trend_detection_threshold", 0.001)
        self.volatility_threshold = getattr(
            config, "volatility_detection_threshold", 0.02
        )

        logger.info(
            f"ThresholdManager initialized: base={self.base_threshold}, "
            f"adaptive={self.adaptive_mode}, "
            f"range=[{self.min_threshold}, {self.max_threshold}], "
            f"regime_window={self.regime_window}"
        )

        # Validate configuration
        self._validate_config()

    def _validate_config(self) -> None:
        """
        Validate configuration parameters for consistency and safety.

        Raises:
            ValueError: If configuration parameters are invalid
        """
        if self.min_threshold >= self.max_threshold:
            raise ValueError(
                f"min_threshold ({self.min_threshold}) must be less than max_threshold ({self.max_threshold})"
            )

        # If the configured base threshold is out-of-bounds, clamp it and warn rather than raising.
        if (
            self.base_threshold < self.min_threshold
            or self.base_threshold > self.max_threshold
        ):
            logger.warning(
                f"base_threshold ({self.base_threshold}) outside expected range[{self.min_threshold}, {self.max_threshold}]; clamping to range."
            )
            self.base_threshold = float(
                np.clip(self.base_threshold, self.min_threshold, self.max_threshold)
            )

        if self.regime_window < 10:
            raise ValueError(
                f"regime_window ({self.regime_window}) should be at least 10 for reliable analysis"
            )

        if self.performance_memory < 10:
            raise ValueError(
                f"performance_memory ({self.performance_memory}) should be at least 10 for meaningful adaptation"
            )

        if not (0 < self.trend_threshold < 1):
            raise ValueError(
                f"trend_threshold ({self.trend_threshold}) should be between 0 and 1"
            )

        if not (0 < self.volatility_threshold < 1):
            raise ValueError(
                f"volatility_threshold ({self.volatility_threshold}) should be between 0 and 1"
            )

    def update_action_stats(self, raw_action_value: float) -> None:
        """
        Update the history of raw action values for Z-score calculation.

        Args:
            raw_action_value: The raw output from the model (usually between -1 and 1).
        """
        self.action_history.append(abs(raw_action_value))

    def _calculate_z_score_threshold(
        self, raw_action_value: float, base_threshold: float
    ) -> float:
        """
        Calculate threshold based on Z-score of the current action value.

        If the action is statistically significant (Z-score > threshold),
        return a threshold that allows this action to pass (slightly lower than action value).
        Otherwise, return the base threshold.

        Args:
            raw_action_value: The current raw action value.
            base_threshold: The fallback base threshold.

        Returns:
            The calculated threshold.
        """
        if len(self.action_history) < 10:
            return base_threshold

        abs_action = abs(raw_action_value)
        history_array = np.array(self.action_history)
        # calculate_z_score_single returns (value - mean)/std; this supports NANs.
        z_score = calculate_z_score_single(
            abs_action, history_array, self.min_std, method=self.z_score_method
        )

        if z_score > self.z_score_threshold:
            # If significant, lower the threshold to just below the current action
            # to ensure it passes, but keep it above min_threshold.
            # We use 0.99 factor to be safe.
            dynamic_threshold_mag = abs_action * 0.99
            dynamic_threshold_mag = max(self.min_threshold, dynamic_threshold_mag)

            if base_threshold < 0:
                return -dynamic_threshold_mag
            # Record event for observability
            try:
                self.z_score_trigger_count += 1
            except Exception:
                pass
            return dynamic_threshold_mag

        return base_threshold

    def _apply_hybrid_regime_threshold_modifier(
        self, base_threshold: float, regime: Optional[str]
    ) -> float:
        """Apply HybridConfig regime_filter threshold modifier (v454 soft mode)."""
        if not regime:
            return base_threshold

        hybrid_config: Any = None
        if isinstance(self.config, dict):
            hybrid_config = self.config.get("hybrid_config")
        else:
            hybrid_config = getattr(self.config, "hybrid_config", None)

        if not isinstance(hybrid_config, dict) or not hybrid_config.get("enabled", False):
            return base_threshold

        regime_filter = hybrid_config.get("regime_filter", {})
        if not isinstance(regime_filter, dict) or not regime_filter.get("enabled", False):
            return base_threshold

        if str(regime_filter.get("mode", "hard")).lower() != "soft":
            return base_threshold

        constraints = regime_filter.get("regime_constraints", {})
        if not isinstance(constraints, dict):
            return base_threshold

        constraint = constraints.get(str(regime))
        if not isinstance(constraint, dict):
            return base_threshold

        permission = str(constraint.get("action_permission", "allow")).lower()
        if permission == "deny":
            return base_threshold

        modifier_raw = constraint.get("confidence_threshold_modifier")
        if modifier_raw is None:
            return base_threshold

        try:
            modifier = float(modifier_raw)
        except (TypeError, ValueError):
            return base_threshold

        if not np.isfinite(modifier) or modifier == 0.0:
            return base_threshold

        sign = 1.0 if base_threshold >= 0 else -1.0
        magnitude = abs(base_threshold)
        adjusted_magnitude = float(
            np.clip(magnitude + modifier, self.min_threshold, self.max_threshold)
        )
        return sign * adjusted_magnitude

    def get_threshold(
        self,
        volatility: Optional[float] = None,
        current_price: Optional[float] = None,
        regime: Optional[str] = None,
        base_value: Optional[float] = None,
        raw_action_value: Optional[float] = None,
    ) -> float:
        """
        Calculate the current action threshold.

        Args:
            volatility: Current market volatility (e.g., ATR).
            current_price: Current market price (used to normalize ATR).
            regime: Current market regime (e.g., 'trending', 'ranging').
            base_value: Optional base threshold to use instead of self.base_threshold.
            raw_action_value: Optional raw action value for Z-score calculation.

        Returns:
            The threshold value to use for action discretization.
        """
        base = base_value if base_value is not None else self.base_threshold
        base = self._apply_hybrid_regime_threshold_modifier(base, regime)

        # Z-Score based dynamic thresholding
        if self.dynamic_threshold_mode == "z_score" and raw_action_value is not None:
            return self._calculate_z_score_threshold(raw_action_value, base)

        if not self.adaptive_mode:
            # Even in non-adaptive mode, we might want to apply regime-based scaling?
            # For now, let's stick to the logic that adaptive_mode controls all dynamic behavior.
            return base

        if volatility is None or current_price is None or current_price <= 0:
            # Fallback to base threshold if data is missing
            adjusted_threshold = base
        else:
            # Calculate relative volatility (ATR / Price)
            relative_volatility = volatility / current_price

            # Adjust threshold based on volatility
            # Formula: base +/- (volatility_factor * multiplier)
            adjustment = relative_volatility * self.volatility_multiplier
            if base < 0:
                adjusted_threshold = base - adjustment
            else:
                adjusted_threshold = base + adjustment

        # Regime-based adjustment
        if regime:
            regime_upper = regime.upper()

            # Specific regime overrides for v451 optimization
            buy_favorable_regimes = [
                "BUY_BREAKOUT",
                "BUY_DIVERGENCE",
                "BUY_MOMENTUM_STRONG",
                "BUY_VOLUME_SURGE",
                "STRONG_BULL",
                "MODERATE_BULL",  # Added for v453 improvement
            ]
            sell_favorable_regimes = [
                "SELL_BREAKDOWN",
                "SELL_DIVERGENCE",
                "SELL_MOMENTUM_WEAK",
                "SELL_VOLUME_SURGE",
                "STRONG_BEAR",
                "MODERATE_BEAR",  # Added for v453 improvement
                "BREAKDOWN",      # Added for v453 improvement (covers BREAKDOWN_SETUP)
            ]

            if any(r in regime_upper for r in buy_favorable_regimes):
                if base > 0:  # Buy threshold
                    # Encourage Buy: Lower the threshold
                    adjusted_threshold *= 0.5
                    logger.debug(
                        f"Decreased BUY threshold for {regime}: {adjusted_threshold:.4f}"
                    )
                else:  # Sell threshold
                    # Discourage Sell: Make threshold more negative (larger magnitude)
                    # v453 Optimization: Relaxed from 10.0 to 1.0 to allow full counter-trend scalping (v3 behavior)
                    adjusted_threshold *= 1.0
                    logger.debug(
                        f"Increased SELL threshold (discourage) for {regime}: {adjusted_threshold:.4f}"
                    )

            elif any(r in regime_upper for r in sell_favorable_regimes):
                if base > 0:  # Buy threshold
                    # Discourage Buy: Raise the threshold
                    # v453 Optimization: Relaxed from 10.0 to 1.0 to allow full counter-trend scalping (v3 behavior)
                    adjusted_threshold *= 1.0
                    logger.debug(
                        f"Increased BUY threshold (discourage) for {regime}: {adjusted_threshold:.4f}"
                    )
                else:  # Sell threshold
                    # Encourage Sell: Make threshold less negative (smaller magnitude)
                    adjusted_threshold *= 0.5
                    logger.debug(
                        f"Decreased SELL threshold (encourage) for {regime}: {adjusted_threshold:.4f}"
                    )

            elif any(
                r in regime_upper
                for r in ["HIGH_VOLATILITY"]
            ):
                # Increase threshold in high volatility ranging to reduce noise entries
                # v453 Optimization: Relaxed from 10.0 to 1.0 to avoid missing opportunities (v3 behavior)
                adjusted_threshold *= 1.0
                logger.debug(
                    f"Increased threshold for high volatility market: {adjusted_threshold:.4f}"
                )
            elif any(
                r in regime_upper
                for r in ["SIDEWAYS", "RANGING", "CONSOLIDATION"]
            ):
                # Restore baseline behavior for normal ranging/consolidation
                # v453 Final Fix: Set to 1.0 to allow profitable range trading (same as "Unknown" regime)
                adjusted_threshold *= 1.0
                logger.debug(
                    f"Threshold multiplier 1.0 applied for ranging market: {adjusted_threshold:.4f}"
                )
            elif any(r in regime_upper for r in ["TRENDING", "BULL", "BEAR"]):
                # Slightly decrease threshold in strong trends to ensure we don't miss moves?
                # Or keep as is. Let's keep as is for now to avoid over-trading.
                pass

        # Clip to safe bounds
        if base < 0:
            # Negative threshold logic (SELL)
            # We want the magnitude to be within [min, max]
            # So the value should be within [-max, -min]
            final_threshold = np.clip(
                adjusted_threshold, -self.max_threshold, -self.min_threshold
            )
        else:
            # Positive threshold logic (BUY)
            final_threshold = np.clip(
                adjusted_threshold, self.min_threshold, self.max_threshold
            )

        self.last_threshold = final_threshold
        if volatility is not None and current_price is not None and current_price > 0:
            self.last_volatility = volatility / current_price

        return float(final_threshold)

    def get_state_info(self) -> Dict[str, Any]:
        """Return current state information for logging/debugging."""
        return {
            "current_threshold": self.last_threshold,
            "last_volatility": self.last_volatility,
            "is_adaptive": self.adaptive_mode,
            "signal_history_size": len(self.signal_history),
            "regime_cache_active": self._regime_cache is not None,
            "threshold_cache_active": self._threshold_cache is not None,
            "regime_detector_available": True,
        }

    def detect_market_regime(
        self, data: pd.DataFrame, current_index: Optional[int] = None
    ) -> str:
        """
        Detect current market regime using advanced MarketRegimeDetector.

        This method leverages the comprehensive MarketRegimeDetector for accurate
        regime classification, falling back to simple analysis if data is insufficient.
        Results are cached for performance optimization.

        Args:
            data: Recent market data (OHLCV format)
            current_index: Current index in data (optional, for backward compatibility)

        Returns:
            Market regime: 'trending_bull', 'trending_bear', 'ranging', 'volatile'
        """
        # Check cache first
        cache_key = f"{len(data)}_{data.index[-1] if len(data) > 0 else 'empty'}"
        if self._regime_cache and self._regime_cache.get("cache_key") == cache_key:
            return self._regime_cache["regime"]

        try:
            # Use advanced MarketRegimeDetector if sufficient data
            if len(data) >= 10:  # Reduced from 50 to match detector's capability
                regime = self.regime_detector.detect_regime(data)
                detected_regime = self._normalize_regime_label(regime)
            else:
                detected_regime = "ranging"
        except Exception as e:
            logger.warning(
                f"Advanced regime detection failed, defaulting to ranging: {e}"
            )
            logger.warning(f"Exception type: {type(e).__name__}")
            import traceback

            logger.warning(f"Traceback: {traceback.format_exc()}")
            detected_regime = "ranging"

        # Cache result
        self._regime_cache = {
            "cache_key": cache_key,
            "regime": detected_regime,
            "timestamp": pd.Timestamp.now(),
        }

        return detected_regime

    @staticmethod
    def _normalize_regime_label(regime: object) -> str:
        """Normalize diverse regime enum/string formats into threshold-manager labels."""
        regime_text = str(getattr(regime, "value", regime)).lower()
        if any(key in regime_text for key in ["bull", "buy_breakout", "buy_momentum"]):
            return "trending_bull"
        if any(
            key in regime_text
            for key in ["bear", "sell_breakdown", "sell_momentum", "breakdown"]
        ):
            return "trending_bear"
        if any(key in regime_text for key in ["volatile", "volatility", "extreme"]):
            return "volatile"
        if any(key in regime_text for key in ["rang", "sideways", "consolidation"]):
            return "ranging"
        if "breakout" in regime_text:
            return "trending_bull"
        return "ranging"

    def get_regime_adjustments(self, regime: str) -> Dict[str, float]:
        """
        Get threshold adjustments for a specific market regime.

        Args:
            regime: Market regime ('trending_bull', 'trending_bear', 'ranging', 'volatile')

        Returns:
            Dictionary with adjustment factors for confidence and strength thresholds
        """
        regime_adjustments = {
            "trending_bull": {"confidence_multiplier": 0.9, "strength_multiplier": 0.8},
            "trending_bear": {"confidence_multiplier": 0.9, "strength_multiplier": 0.8},
            "ranging": {
                "confidence_multiplier": 1.1,
                "strength_multiplier": 1.2,
            },  # Higher thresholds in ranging markets
            "volatile": {
                "confidence_multiplier": 1.2,
                "strength_multiplier": 1.3,
            },  # Much higher in volatile markets
            "unknown": {"confidence_multiplier": 1.0, "strength_multiplier": 1.0},
        }

        return regime_adjustments.get(
            regime, {"confidence_multiplier": 1.0, "strength_multiplier": 1.0}
        )

    def calculate_adaptive_signal_thresholds(
        self,
        data: pd.DataFrame,
        base_confidence: float = 0.7,
        base_strength: float = 0.4,
    ) -> Dict[str, float]:
        """
        Calculate adaptive signal thresholds based on market regime and performance.

        This method combines market regime analysis with performance-based adaptation
        to provide optimal signal filtering thresholds. Results are cached for performance.

        Args:
            data: Market data (OHLCV format)
            base_confidence: Base confidence threshold (default: 0.7)
            base_strength: Base signal strength threshold (default: 0.4)

        Returns:
            Dictionary with adaptive thresholds and metadata
        """
        # Check cache
        cache_key = f"{len(data)}_{base_confidence}_{base_strength}"
        if (
            self._threshold_cache
            and self._threshold_cache.get("cache_key") == cache_key
        ):
            return self._threshold_cache["thresholds"]

        regime = self.detect_market_regime(data)
        regime_adjustments = self.get_regime_adjustments(regime)
        performance_adjustment = self._calculate_performance_adjustment()

        confidence_threshold = (
            base_confidence
            * regime_adjustments["confidence_multiplier"]
            * performance_adjustment["confidence"]
        )
        signal_strength_threshold = (
            base_strength
            * regime_adjustments["strength_multiplier"]
            * performance_adjustment["strength"]
        )

        # Ensure reasonable bounds
        confidence_threshold = np.clip(confidence_threshold, 0.5, 0.9)
        signal_strength_threshold = np.clip(signal_strength_threshold, 0.2, 0.7)

        thresholds = {
            "confidence_threshold": confidence_threshold,
            "signal_strength_threshold": signal_strength_threshold,
            "regime": regime,
            "performance_adjustment": performance_adjustment,
        }

        # Cache result
        self._threshold_cache = {
            "cache_key": cache_key,
            "thresholds": thresholds,
            "timestamp": pd.Timestamp.now(),
        }

        return thresholds

    # Backwards compatibility alias
    def calculate_adaptive_thresholds(
        self,
        data: pd.DataFrame,
        base_confidence: float = 0.7,
        base_strength: float = 0.4,
    ) -> Dict[str, float]:
        """Backward-compatible alias for calculate_adaptive_signal_thresholds
        Some older tests or modules call calculate_adaptive_thresholds; maintain alias
        to avoid breaking older code while keeping new name for clarity.
        """
        return self.calculate_adaptive_signal_thresholds(
            data, base_confidence, base_strength
        )

    def _calculate_performance_adjustment(self) -> Dict[str, float]:
        """
        Calculate threshold adjustments based on recent performance.

        Returns:
            Performance-based adjustment factors
        """
        if len(self.signal_history) < 10:
            return {"confidence": 1.0, "strength": 1.0}

        recent_signals = self.signal_history[-20:]  # Last 20 signals
        win_rate = sum(1 for s in recent_signals if s.get("profitable", False)) / len(
            recent_signals
        )

        # Adjust thresholds based on win rate
        if win_rate > 0.6:  # Good performance, can be less strict
            adjustment = 0.9
        elif win_rate < 0.4:  # Poor performance, be more strict
            adjustment = 1.1
        else:
            adjustment = 1.0

        return {"confidence": adjustment, "strength": adjustment}

    def update_performance(self, signal_result: Dict[str, Any]):
        """
        Update performance tracking with signal result.

        Args:
            signal_result: Result of executed signal
        """
        self.signal_history.append(signal_result)

        # Keep memory limited
        if len(self.signal_history) > self.performance_memory:
            self.signal_history = self.signal_history[-self.performance_memory :]

    def reset(self) -> None:
        """
        Reset the ThresholdManager state for a new episode/training session.

        This clears performance history and cache while preserving configuration.
        """
        self.signal_history.clear()
        self._regime_cache = None
        self._threshold_cache = None
        self.last_threshold = self.base_threshold
        self.last_volatility = 0.0

        logger.info("ThresholdManager state reset")
