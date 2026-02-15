"""
Granville's Law pattern recognition for Action Signal Guide.

Joseph Granville's trading rules based on the relationship between price and volume.
This implementation focuses on the core principles while providing configurable parameters.
"""

from collections.abc import Mapping
from typing import TypedDict

import pandas as pd

try:
    from ztb.features.generators.technical.volume.obv import compute_obv
except ImportError:

    def compute_obv(df: pd.DataFrame) -> pd.Series:
        return pd.Series([1000.0] * len(df), index=df.index)


from ztb.trading.strategies.action_signal_guide.pattern_recognition.base import (
    SignalResult,
    TrendPatternRecognizer,
)


class GranvilleSignal(TypedDict):
    direction: float
    strength: float
    description: str
    confidence: float


class GranvilleLawRecognizer(TrendPatternRecognizer):
    """
    Recognizes patterns using Granville's Law.

    Based on Joseph Granville's 8 rules for interpreting price and volume relationships.
    This implementation provides the core rules with configurable parameters.
    """

    def __init__(self, config: Mapping[str, object] | None = None):
        cfg: dict[str, object] = dict(config) if config else {}
        super().__init__(cfg)
        self.pattern_type = "granville_law"
        # Price change thresholds
        self.price_change_threshold = self._to_float(
            cfg.get("price_change_threshold"), 0.005
        )
        # Volume change thresholds
        self.volume_change_threshold = self._to_float(
            cfg.get("volume_change_threshold"), 0.1
        )
        # Trend determination period
        self.trend_period = self._to_int(cfg.get("trend_period"), 20)
        # Minimum volume for valid signals
        self.min_volume = self._to_float(cfg.get("min_volume"), 1000.0)
        # Use OBV for volume analysis
        self.use_obv = self._to_bool(cfg.get("use_obv"), True)

    def recognize(
        self,
        data: pd.DataFrame,
        index: int = -1,
        multi_timeframe_data: dict[str, object] | None = None,
    ) -> SignalResult | None:
        """
        Recognize Granville's Law patterns in the data.

        Args:
            data: OHLCV DataFrame
            index: Index to analyze (-1 for latest)

        Returns:
            SignalResult if pattern detected, None otherwise
        """
        del multi_timeframe_data

        if len(data) < self.trend_period + 2:
            return None

        resolved_index = self.resolve_analysis_index(
            len(data), index, min_required_index=self.trend_period + 1
        )
        if resolved_index is None:
            return None

        # Get required data window
        window_data = data.iloc[resolved_index - self.trend_period : resolved_index + 1]

        # Calculate price and volume changes
        price_change, volume_change = self._calculate_changes(window_data)

        # Determine market trend
        market_trend = self._determine_market_trend(window_data)

        # Apply Granville's rules
        signal = self._apply_granville_rules(
            price_change, volume_change, market_trend, window_data.iloc[-1]
        )

        if signal:
            return SignalResult(
                signal_type="granville_law",
                strength=abs(signal["strength"]),
                direction=signal["direction"],
                description=signal["description"],
                confidence=signal["confidence"],
            )

        return None

    def _calculate_changes(self, data: pd.DataFrame) -> tuple[float, float]:
        """
        Calculate price and volume changes over the analysis period.

        Returns:
            Tuple of (price_change_ratio, volume_change_ratio)
        """
        if len(data) < 2:
            return 0.0, 0.0

        # Price change (close to close)
        recent_close = data["close"].iloc[-1]
        previous_close = data["close"].iloc[-2]
        price_change = self.safe_ratio(
            float(recent_close - previous_close), float(previous_close), default=0.0
        )

        # Volume change - use OBV if enabled, otherwise raw volume
        if self.use_obv and len(data) >= 5:  # Need minimum data for OBV
            try:
                obv_series = compute_obv(data)
                recent_obv = obv_series.iloc[-1]
                previous_obv = obv_series.iloc[-2]
                volume_change = self.safe_ratio(
                    float(recent_obv - previous_obv), abs(float(previous_obv)), default=0.0
                )
            except Exception:
                # Fallback to raw volume if OBV calculation fails
                recent_volume = data["volume"].iloc[-1]
                previous_volume = data["volume"].iloc[-2]
                volume_change = self.safe_ratio(
                    float(recent_volume - previous_volume),
                    float(previous_volume),
                    default=0.0,
                )
        else:
            # Raw volume change
            recent_volume = data["volume"].iloc[-1]
            previous_volume = data["volume"].iloc[-2]
            volume_change = self.safe_ratio(
                float(recent_volume - previous_volume),
                float(previous_volume),
                default=0.0,
            )

        return price_change, volume_change

    def _determine_market_trend(self, data: pd.DataFrame) -> str:
        """
        Determine the current market trend (bullish/bearish).

        Returns:
            'bullish', 'bearish', or 'sideways'
        """
        if len(data) < self.trend_period:
            return "sideways"

        # Simple trend determination based on moving averages
        closes = data["close"]
        from ztb.features.generators.technical.trend.sma import compute_sma

        ma_short = compute_sma(data, period=min(5, len(closes)))
        ma_long = compute_sma(data, period=min(self.trend_period, len(closes)))

        if len(ma_short) < 2 or len(ma_long) < 2:
            return "sideways"

        # Compare recent short MA vs long MA
        recent_short = ma_short.iloc[-1]
        recent_long = ma_long.iloc[-1]

        if recent_short > recent_long * 1.005:  # 0.5% above
            return "bullish"
        elif recent_short < recent_long * 0.995:  # 0.5% below
            return "bearish"
        else:
            return "sideways"

    def _apply_granville_rules(
        self,
        price_change: float,
        volume_change: float,
        market_trend: str,
        current_data: pd.Series,
    ) -> GranvilleSignal | None:
        """
        Apply Granville's Law rules to generate trading signals.

        Core Rules:
        1. Price up + Volume up = Buy (strong bullish)
        2. Price down + Volume up = Sell (strong bearish)
        3. Price up + Volume down = Buy (in bear market only)
        4. Price down + Volume down = Sell (in bull market only)
        """
        # Check minimum volume
        if current_data.get("volume", 0) < self.min_volume:
            return None

        price_up = price_change > self.price_change_threshold
        price_down = price_change < -self.price_change_threshold
        volume_up = volume_change > self.volume_change_threshold
        volume_down = volume_change < -self.volume_change_threshold

        # Rule 1: Price up + Volume up = Strong Buy
        if price_up and volume_up:
            strength = min(1.0, abs(price_change) * abs(volume_change) * 100)
            return {
                "direction": 1.0,
                "strength": strength,
                "description": f"Granville Rule 1: Price ↑ + Volume ↑ = Strong Buy (trend: {market_trend})",
                "confidence": min(0.9, strength),
            }

        # Rule 2: Price down + Volume up = Strong Sell
        elif price_down and volume_up:
            strength = min(1.0, abs(price_change) * abs(volume_change) * 100)
            return {
                "direction": -1.0,
                "strength": strength,
                "description": f"Granville Rule 2: Price ↓ + Volume ↑ = Strong Sell (trend: {market_trend})",
                "confidence": min(0.9, strength),
            }

        # Rule 3: Price up + Volume down = Buy (bear market only)
        elif price_up and volume_down and market_trend == "bearish":
            strength = min(0.7, abs(price_change) * 50)
            return {
                "direction": 1.0,
                "strength": strength,
                "description": "Granville Rule 3: Price ↑ + Volume ↓ = Buy in Bear Market",
                "confidence": min(0.6, strength),
            }

        # Rule 4: Price down + Volume down = Sell (bull market only)
        elif price_down and volume_down and market_trend == "bullish":
            strength = min(0.7, abs(price_change) * 50)
            return {
                "direction": -1.0,
                "strength": strength,
                "description": "Granville Rule 4: Price ↓ + Volume ↓ = Sell in Bull Market",
                "confidence": min(0.6, strength),
            }

        # Additional rules for sideways/invalid signals
        # Rule 5: Price sideways + Volume up = Potential accumulation
        if not price_up and not price_down and volume_up:
            return {
                "direction": 0.0,
                "strength": 0.3,
                "description": "Granville Rule 5: Sideways Price + Volume ↑ = Accumulation",
                "confidence": 0.4,
            }

        # Rule 6: Price sideways + Volume down = Potential distribution
        elif not price_up and not price_down and volume_down:
            return {
                "direction": 0.0,
                "strength": 0.3,
                "description": "Granville Rule 6: Sideways Price + Volume ↓ = Distribution",
                "confidence": 0.4,
            }

        return None

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
