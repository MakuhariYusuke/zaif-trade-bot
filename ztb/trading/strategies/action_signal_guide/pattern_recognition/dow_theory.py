"""
Dow Theory pattern recognition for Action Signal Guide.

Based on Charles Dow's principles of market analysis. This implementation
focuses on trend confirmation and reversal signals using price action.
"""

from collections.abc import Mapping
from typing import TypedDict

import numpy as np
import pandas as pd

try:
    from ztb.features.generators.technical.trend.supertrend import (
        compute_supertrend_direction,
    )
except ImportError:

    def compute_supertrend_direction(df: pd.DataFrame) -> pd.Series:
        return pd.Series([0] * len(df), index=df.index)

from ztb.features.generators.technical.volatility.bollinger import compute_bb_width
from ztb.trading.strategies.action_signal_guide.pattern_recognition.base import (
    SignalResult,
    TrendPatternRecognizer,
)


class DowTrendState(TypedDict, total=False):
    direction: int
    strength: float
    slope: float
    period: int
    supertrend_direction: int
    volatility: float


class DowSignalPayload(TypedDict):
    direction: float
    strength: float
    description: str
    confidence: float


class DowTheoryRecognizer(TrendPatternRecognizer):
    """
    Recognizes patterns using Dow Theory principles.

    Implements core Dow Theory principles for trend analysis and confirmation.
    Focuses on primary trend identification and confirmation signals.
    """

    def __init__(self, config: Mapping[str, object] | None = None):
        cfg: dict[str, object] = dict(config) if config else {}
        super().__init__(cfg)
        self.pattern_type = "dow_theory"

        # Moving average periods for trend analysis
        self.primary_trend_period = self._to_int(cfg.get("primary_trend_period"), 50)
        self.secondary_trend_period = self._to_int(
            cfg.get("secondary_trend_period"), 20
        )
        self.short_trend_period = self._to_int(cfg.get("short_trend_period"), 10)

        # Confirmation thresholds
        self.trend_confirmation_threshold = self._to_float(
            cfg.get("trend_confirmation_threshold"), 0.002
        )
        self.reversal_threshold = self._to_float(cfg.get("reversal_threshold"), 0.02)

        # Confirmation options
        self.require_volume_confirmation = self._to_bool(
            cfg.get("require_volume_confirmation"), False
        )

        # Optional enhancement indicators
        self.use_supertrend = self._to_bool(cfg.get("use_supertrend"), True)
        self.use_bollinger = self._to_bool(cfg.get("use_bollinger"), True)
        self.bb_period = self._to_int(cfg.get("bb_period"), 20)
        self.volatility_threshold = self._to_float(cfg.get("volatility_threshold"), 0.1)

    def recognize(
        self,
        data: pd.DataFrame,
        index: int = -1,
        multi_timeframe_data: dict[str, object] | None = None,
    ) -> SignalResult | None:
        """
        Recognize Dow Theory patterns in the data.

        Args:
            data: OHLCV DataFrame
            index: Index to analyze (-1 for latest)

        Returns:
            SignalResult if pattern detected, None otherwise
        """
        del multi_timeframe_data

        if "close" not in data.columns:
            return None
        if len(data) < self.primary_trend_period + 10:
            return None

        resolved_index = self.resolve_analysis_index(
            len(data), index, min_required_index=self.primary_trend_period
        )
        if resolved_index is None:
            return None

        closes = data["close"].to_numpy(dtype="float64")

        # Analyze trends at different levels
        primary_trend = self._analyze_primary_trend(data, closes, resolved_index)
        secondary_trend = self._analyze_secondary_trend(data, closes, resolved_index)
        short_trend = self._analyze_short_trend(data, closes, resolved_index)

        # Check for trend confirmation or reversal
        signal = self._check_trend_confirmation(
            primary_trend, secondary_trend, short_trend, data, resolved_index
        )

        if signal is None:
            return None

        return SignalResult(
            signal_type="dow_theory",
            strength=abs(signal["strength"]),
            direction=signal["direction"],
            description=signal["description"],
            confidence=signal["confidence"],
        )

    def _analyze_primary_trend(
        self, data: pd.DataFrame, closes: np.ndarray, index: int
    ) -> DowTrendState:
        """Analyze primary (long-term) trend with optional enhancement indicators."""
        base_trend = self._analyze_trend(closes, index, self.primary_trend_period)
        window = data.iloc[max(0, index - self.primary_trend_period) : index + 1]

        slope_direction = base_trend["direction"]
        supertrend_direction = self._calculate_supertrend_direction(window)
        final_direction, strength_multiplier = self._combine_trend_directions(
            slope_direction, supertrend_direction
        )

        volatility = self._calculate_bollinger_volatility(window)
        volatility_multiplier = self._volatility_strength_multiplier(volatility)

        return {
            "direction": final_direction,
            "strength": base_trend["strength"] * strength_multiplier * volatility_multiplier,
            "slope": base_trend["slope"],
            "supertrend_direction": supertrend_direction,
            "volatility": volatility,
            "period": self.primary_trend_period,
        }

    def _analyze_secondary_trend(
        self, data: pd.DataFrame, closes: np.ndarray, index: int
    ) -> DowTrendState:
        """Analyze secondary (medium-term) trend."""
        del data
        trend = self._analyze_trend(closes, index, self.secondary_trend_period)
        trend["period"] = self.secondary_trend_period
        return trend

    def _analyze_short_trend(
        self, data: pd.DataFrame, closes: np.ndarray, index: int
    ) -> DowTrendState:
        """Analyze short-term trend."""
        del data
        trend = self._analyze_trend(closes, index, self.short_trend_period)
        trend["period"] = self.short_trend_period
        return trend

    def _analyze_trend(
        self, closes: np.ndarray, index: int, period: int
    ) -> DowTrendState:
        """Analyze trend over a configurable window using cached linear-regression weights."""
        start = max(0, index - period)
        window_closes = closes[start : index + 1]

        if window_closes.size < 2:
            return {"direction": 0, "strength": 0.0, "slope": 0.0, "period": period}

        normalized_slope = self.calculate_normalized_slope(window_closes)
        direction = self.slope_direction(
            normalized_slope, self.trend_confirmation_threshold
        )

        return {
            "direction": direction,
            "strength": abs(normalized_slope),
            "slope": normalized_slope,
            "period": period,
        }

    def _calculate_supertrend_direction(self, window: pd.DataFrame) -> int:
        """Calculate SuperTrend direction for trend confirmation."""
        if not self.use_supertrend or len(window) < 15:
            return 0

        try:
            st_direction = compute_supertrend_direction(window)
            if st_direction.empty:
                return 0
            return int(st_direction.iloc[-1])
        except Exception:
            return 0

    def _calculate_bollinger_volatility(self, window: pd.DataFrame) -> float:
        """Calculate Bollinger-band width based volatility."""
        if not self.use_bollinger or len(window) < self.bb_period:
            return 0.0

        try:
            bb_width = compute_bb_width(window, period=self.bb_period)
            if bb_width.empty:
                return 0.0
            return float(bb_width.iloc[-1])
        except Exception:
            return 0.0

    @staticmethod
    def _combine_trend_directions(
        slope_direction: int, supertrend_direction: int
    ) -> tuple[int, float]:
        """Combine slope and SuperTrend direction with confidence multipliers."""
        if supertrend_direction != 0 and slope_direction != 0:
            if supertrend_direction == slope_direction:
                return slope_direction, 1.5  # Strong confirmation
            return supertrend_direction, 0.8  # Conflict fallback
        if supertrend_direction != 0:
            return supertrend_direction, 1.2
        return slope_direction, 1.0

    def _volatility_strength_multiplier(self, volatility: float) -> float:
        """Adjust trend strength by current volatility regime."""
        if volatility > self.volatility_threshold:
            return 1.2
        if volatility < self.volatility_threshold * 0.5:
            return 0.9
        return 1.0

    def _check_trend_confirmation(
        self,
        primary: DowTrendState,
        secondary: DowTrendState,
        short: DowTrendState,
        data: pd.DataFrame,
        index: int,
    ) -> DowSignalPayload | None:
        """
        Check for trend confirmation or reversal signals based on Dow Theory.

        Key principles:
        - Trends continue until clear reversal signals
        - Multiple timeframes should confirm
        - Volume should confirm price action
        """
        # Check for bullish confirmation (primary OR secondary trend aligned)
        if primary["direction"] == 1 or secondary["direction"] == 1:
            if self.require_volume_confirmation and not self._check_volume_confirmation(
                data, index, 1
            ):
                return None

            strength = self._enforce_min_strength(
                max(primary["strength"], secondary["strength"])
            )
            return {
                "direction": 1.0,
                "strength": strength,
                "description": (
                    "Dow Theory: Bullish trend confirmed "
                    f"(primary or secondary aligned, strength: {strength:.3f})"
                ),
                "confidence": self._calculate_confidence(strength, cap=0.7, scale=0.8),
            }

        # Check for bearish confirmation (primary OR secondary trend aligned)
        if primary["direction"] == -1 or secondary["direction"] == -1:
            if self.require_volume_confirmation and not self._check_volume_confirmation(
                data, index, -1
            ):
                return None

            strength = self._enforce_min_strength(
                max(primary["strength"], secondary["strength"])
            )
            return {
                "direction": -1.0,
                "strength": strength,
                "description": (
                    "Dow Theory: Bearish trend confirmed "
                    f"(primary or secondary aligned, strength: {strength:.3f})"
                ),
                "confidence": self._calculate_confidence(strength, cap=0.7, scale=0.8),
            }

        reversal_signal = self._check_reversal_signals(primary, secondary, short, data, index)
        if reversal_signal:
            return reversal_signal

        divergence_signal = self._check_divergence_signals(
            primary, secondary, short, data, index
        )
        if divergence_signal:
            return divergence_signal

        if primary["direction"] == 0 and secondary["direction"] == 0:
            short_direction = short["direction"] if short["direction"] != 0 else 1
            strength = self._enforce_min_strength(short["strength"])
            return {
                "direction": float(short_direction),
                "strength": strength,
                "description": (
                    "Dow Theory: Weak trend signal "
                    f"(sideways market, strength: {strength:.6f})"
                ),
                "confidence": self._calculate_confidence(strength, cap=0.3, scale=0.5),
            }

        return None

    def _check_volume_confirmation(
        self, data: pd.DataFrame, index: int, _direction: int
    ) -> bool:
        """Check if volume confirms the price trend."""
        if index < 5 or "volume" not in data.columns:
            return False

        recent_volume = data["volume"].iloc[index - 4 : index + 1]
        avg_volume = recent_volume.mean()

        # Volume should be above average for trend confirmation
        return float(recent_volume.iloc[-1]) > float(avg_volume)

    def _check_reversal_signals(
        self,
        primary: DowTrendState,
        secondary: DowTrendState,
        short: DowTrendState,
        _data: pd.DataFrame,
        _index: int,
    ) -> DowSignalPayload | None:
        """
        Check for potential trend reversal signals.

        Dow Theory: Trends continue until clear reversal signals appear.
        """
        if abs(primary["slope"]) <= self.reversal_threshold:
            return None

        primary_bull_reversing = (
            primary["direction"] == 1
            and secondary["direction"] <= 0
            and short["direction"] <= 0
        )
        primary_bear_reversing = (
            primary["direction"] == -1
            and secondary["direction"] >= 0
            and short["direction"] >= 0
        )

        if not (primary_bull_reversing or primary_bear_reversing):
            return None

        direction = -1.0 if primary["direction"] == 1 else 1.0
        strength = min(0.8, abs(primary["slope"]))
        trend_type = "bullish" if direction == 1.0 else "bearish"

        return {
            "direction": direction,
            "strength": strength,
            "description": f"Dow Theory: Primary trend reversal signal ({trend_type})",
            "confidence": self._calculate_confidence(strength, cap=0.8),
        }

    def _check_divergence_signals(
        self,
        primary: DowTrendState,
        secondary: DowTrendState,
        short: DowTrendState,
        _data: pd.DataFrame,
        _index: int,
    ) -> DowSignalPayload | None:
        """
        Check for divergence signals that may indicate trend exhaustion.
        """
        del secondary

        short_reversal_threshold = self.trend_confirmation_threshold * 2.0

        if (
            primary["direction"] == 1
            and short["direction"] == -1
            and abs(short["slope"]) > short_reversal_threshold
        ):
            return {
                "direction": 0.0,
                "strength": 0.4,
                "description": "Dow Theory: Short-term divergence from primary bullish trend",
                "confidence": self._calculate_confidence(0.4, cap=0.5),
            }

        if (
            primary["direction"] == -1
            and short["direction"] == 1
            and abs(short["slope"]) > short_reversal_threshold
        ):
            return {
                "direction": 0.0,
                "strength": 0.4,
                "description": "Dow Theory: Short-term divergence from primary bearish trend",
                "confidence": self._calculate_confidence(0.4, cap=0.5),
            }

        return None

    @staticmethod
    def _enforce_min_strength(strength: float) -> float:
        return max(0.00001, float(strength))

    @staticmethod
    def _calculate_confidence(strength: float, cap: float, scale: float = 1.0) -> float:
        return min(cap, max(0.0001, float(strength) * scale))

    @staticmethod
    def _to_float(value: object, default: float) -> float:
        try:
            return float(value)
        except (TypeError, ValueError):
            return default

    @staticmethod
    def _to_int(value: object, default: int) -> int:
        try:
            return int(value)
        except (TypeError, ValueError):
            return default

    @staticmethod
    def _to_bool(value: object, default: bool) -> bool:
        if isinstance(value, bool):
            return value
        if isinstance(value, (int, float)):
            return bool(value)
        if isinstance(value, str):
            normalized = value.strip().lower()
            if normalized in {"1", "true", "yes", "on"}:
                return True
            if normalized in {"0", "false", "no", "off"}:
                return False
        return default
