"""
Bollinger Bands Pattern Recognizer
ボリンジャーバンドパターン認識 - ボラティリティベースのシグナル生成
"""

import pandas as pd

try:
    from ztb.features.generators.technical.volatility.bollinger import (
        compute_bb_lower,
        compute_bb_middle,
        compute_bb_upper,
        compute_bb_width,
    )
except ImportError:
    # Mock functions if volatility module is not available
    def compute_bb_lower(
        df: pd.DataFrame, period: int = 20, std_dev: int = 2
    ) -> pd.Series:
        return pd.Series([df["close"].mean()] * len(df), index=df.index)

    def compute_bb_middle(df: pd.DataFrame, period: int = 20) -> pd.Series:
        return pd.Series([df["close"].mean()] * len(df), index=df.index)

    def compute_bb_upper(
        df: pd.DataFrame, period: int = 20, std_dev: int = 2
    ) -> pd.Series:
        return pd.Series([df["close"].mean()] * len(df), index=df.index)

    def compute_bb_width(
        df: pd.DataFrame, period: int = 20, std_dev: int = 2
    ) -> pd.Series:
        return pd.Series([0.1] * len(df), index=df.index)

from ztb.trading.constants import ACTION_HOLD

from .base import CandlestickPatternRecognizer, MultiTimeframeData, SignalResult

class BollingerBandsRecognizer(CandlestickPatternRecognizer):
    """
    Bollinger Bands pattern recognizer.
    ボリンジャーバンドベースのパターン認識
    価格のボラティリティとトレンドを分析
    """

    def __init__(self, config: dict[str, object] | None = None):
        super().__init__(config)
        self.pattern_type = "bollinger"
        self.period = int(self.config.get("period", 20))
        self.std_dev = float(self.config.get("std_dev", 2.0))
        self.squeeze_threshold = float(self.config.get(
            "squeeze_threshold", 0.05
        ))  # バンド幅の収縮閾値
        self.expansion_threshold = float(self.config.get(
            "expansion_threshold", 0.15
        ))  # バンド幅の拡大閾値
        self.touch_distance = float(self.config.get(
            "touch_distance", 0.001
        ))  # バンドタッチの距離閾値

    def recognize(
        self,
        data: pd.DataFrame,
        index: int = -1,
        multi_timeframe_data: MultiTimeframeData | None = None,
    ) -> SignalResult | None:
        """
        Recognize Bollinger Bands patterns.

        Args:
            data: OHLCV DataFrame
            index: Current index to analyze

        Returns:
            SignalResult with Bollinger Bands analysis
        """
        if not self.validate_data(data):
            return None

        resolved_index = self.resolve_analysis_index(
            len(data),
            index,
            min_required_index=max(0, self.period - 1),
        )
        if resolved_index is None:
            return SignalResult(
                signal_type="bb_insufficient_data",
                strength=0.0,
                direction=0.0,
                description=f"Insufficient data for Bollinger Bands (need {self.period} periods)",
                metadata={},
                validity_period=1,
                risk_level="low",
            )

        data, index = self._build_analysis_view(data, resolved_index)

        # Calculate market conditions for adaptive parameters
        lookback_data = data.iloc[max(0, index - 30) : index + 1]
        returns = lookback_data["close"].pct_change().dropna()
        current_volatility = float(returns.std()) if not returns.empty else 0.0

        from ztb.features.generators.technical.trend.sma import compute_sma

        rolling_vol = returns.rolling(window=20).std()
        avg_volatility = (
            float(rolling_vol.mean()) if len(returns) >= 20 else current_volatility
        )
        volatility_ratio = (
            current_volatility / avg_volatility if avg_volatility > 0 else 1.0
        )

        # Simple trend strength calculation
        try:
            sma_series = compute_sma(lookback_data, period=20)
            sma_20 = (
                float(sma_series.iloc[-1])
                if not sma_series.empty and not pd.isna(sma_series.iloc[-1])
                else 0.0
            )
        except Exception:
            sma_20 = float(lookback_data["close"].mean())

        trend_strength = (
            abs((float(lookback_data["close"].iloc[-1]) - sma_20) / sma_20)
            if sma_20 != 0
            else 0.5
        )

        if len(data) < self.period:
            return SignalResult(
                signal_type="bb_insufficient_data",
                strength=0.0,
                direction=0.0,
                description=f"Insufficient data for Bollinger Bands (need {self.period} periods)",
                metadata={},
                validity_period=1,
                risk_level="low",
            )

        try:
            # Calculate Bollinger Bands
            bb_upper = compute_bb_upper(data, self.period, self.std_dev)
            bb_lower = compute_bb_lower(data, self.period, self.std_dev)
            bb_middle = compute_bb_middle(data, self.period)
            bb_width = compute_bb_width(data, self.period, self.std_dev)

            current_price = data.iloc[index]["close"]
            current_upper = bb_upper.iloc[index]
            current_lower = bb_lower.iloc[index]
            current_middle = bb_middle.iloc[index]
            current_width = bb_width.iloc[index]

            # Get previous values for trend analysis
            if index > 0:
                prev_price = data.iloc[index - 1]["close"]
                prev_upper = bb_upper.iloc[index - 1]
                prev_lower = bb_lower.iloc[index - 1]
                prev_middle = bb_middle.iloc[index - 1]
                prev_width = bb_width.iloc[index - 1]
            else:
                prev_price = current_price
                prev_upper = current_upper
                prev_lower = current_lower
                prev_middle = current_middle
                prev_width = current_width

            # 1. Band Touch Signals (価格がバンドにタッチ)
            upper_touch_distance = abs(current_price - current_upper) / current_price
            lower_touch_distance = abs(current_price - current_lower) / current_price

            if upper_touch_distance <= self.touch_distance:
                # Upper band touch - potential sell signal
                base_strength = min(
                    0.8, (1 - upper_touch_distance / self.touch_distance) * 0.8
                )

                # Adaptive direction based on touch strength and market conditions
                touch_strength = 1 - upper_touch_distance / self.touch_distance
                direction_factor = -touch_strength * (0.7 + trend_strength * 0.3)
                direction = max(-1.0, direction_factor)

                # Calculate pattern completeness based on how close price is to the band
                pattern_completeness = (
                    touch_strength  # Closer touch = higher completeness
                )

                # Use pattern confidence calculation
                pattern_factors = {
                    "trend_strength": self._calculate_trend_strength(data, index, 15),
                    "candle_size": self._calculate_candle_size_confidence(
                        data, index, 0.7
                    ),  # Band touches involve larger candles
                    "price_movement": self._calculate_price_movement_confidence(
                        data, index, 0.8
                    ),  # Strong movement to touch band
                    "pattern_completeness": pattern_completeness,  # How close price is to the Bollinger band
                }

                # Adaptive strength with volatility boost
                volatility_boost = min(0.2, volatility_ratio * 0.1)
                confidence = self._calculate_pattern_confidence(
                    data,
                    index,
                    pattern_factors,
                    base_confidence=base_strength + volatility_boost,
                )

                return SignalResult(
                    signal_type="bb_upper_touch",
                    strength=confidence,
                    direction=direction,
                    description=f"Price touched upper Bollinger Band at {current_price:.4f}",
                    confidence=confidence,
                    metadata={
                        "band": "upper",
                        "price": current_price,
                        "band_value": current_upper,
                        "distance": upper_touch_distance,
                        "width": current_width,
                        "volatility_ratio": volatility_ratio,
                        "trend_strength": trend_strength,
                        "touch_strength": touch_strength,
                        "pattern_completeness": pattern_completeness,
                    },
                    validity_period=3,
                    risk_level="medium",
                )

            elif lower_touch_distance <= self.touch_distance:
                # Lower band touch - potential buy signal
                base_strength = min(
                    0.8, (1 - lower_touch_distance / self.touch_distance) * 0.8
                )

                # Adaptive direction based on touch strength and market conditions
                touch_strength = 1 - lower_touch_distance / self.touch_distance
                direction_factor = touch_strength * (0.7 + trend_strength * 0.3)
                direction = min(1.0, direction_factor)

                # Calculate pattern completeness based on how close price is to the band
                pattern_completeness = (
                    touch_strength  # Closer touch = higher completeness
                )

                # Use pattern confidence calculation
                pattern_factors = {
                    "trend_strength": self._calculate_trend_strength(data, index, 15),
                    "candle_size": self._calculate_candle_size_confidence(
                        data, index, 0.7
                    ),  # Band touches involve larger candles
                    "price_movement": self._calculate_price_movement_confidence(
                        data, index, 0.8
                    ),  # Strong movement to touch band
                    "pattern_completeness": pattern_completeness,  # How close price is to the Bollinger band
                }

                # Adaptive strength with volatility boost
                volatility_boost = min(0.2, volatility_ratio * 0.1)
                confidence = self._calculate_pattern_confidence(
                    data,
                    index,
                    pattern_factors,
                    base_confidence=base_strength + volatility_boost,
                )

                return SignalResult(
                    signal_type="bb_lower_touch",
                    strength=confidence,
                    direction=direction,
                    description=f"Price touched lower Bollinger Band at {current_price:.4f}",
                    confidence=confidence,
                    metadata={
                        "band": "lower",
                        "price": current_price,
                        "band_value": current_lower,
                        "distance": lower_touch_distance,
                        "width": current_width,
                        "volatility_ratio": volatility_ratio,
                        "trend_strength": trend_strength,
                        "touch_strength": touch_strength,
                        "pattern_completeness": pattern_completeness,
                    },
                    validity_period=3,
                    risk_level="medium",
                )

            # 2. Band Squeeze (バンドの収縮 - ボラティリティ低下)
            if current_width <= self.squeeze_threshold:
                base_strength = min(
                    0.6,
                    (self.squeeze_threshold - current_width) / self.squeeze_threshold,
                )

                # Calculate pattern completeness based on how squeezed the bands are
                pattern_completeness = (
                    self.squeeze_threshold - current_width
                ) / self.squeeze_threshold

                # Use pattern confidence calculation
                pattern_factors = {
                    "trend_strength": self._calculate_trend_strength(data, index, 20),
                    "candle_size": self._calculate_candle_size_confidence(
                        data, index, 0.4
                    ),  # Squeeze involves smaller candles
                    "price_movement": self._calculate_price_movement_confidence(
                        data, index, 0.5
                    ),  # Low movement during squeeze
                    "pattern_completeness": pattern_completeness,  # How squeezed the bands are
                }

                # Adaptive strength with volatility consideration
                volatility_boost = min(
                    0.1, (1 - volatility_ratio) * 0.1
                )  # Boost in low volatility
                confidence = self._calculate_pattern_confidence(
                    data,
                    index,
                    pattern_factors,
                    base_confidence=base_strength + volatility_boost,
                )

                return SignalResult(
                    signal_type="bb_squeeze",
                    strength=confidence,
                    direction=0.0,  # Neutral signal
                    description=f"Bollinger Bands squeeze detected (width: {current_width:.4f})",
                    metadata={
                        "width": current_width,
                        "threshold": self.squeeze_threshold,
                        "squeeze_ratio": current_width / self.squeeze_threshold,
                        "volatility_ratio": volatility_ratio,
                        "trend_strength": trend_strength,
                    },
                    validity_period=5,
                    risk_level="low",
                )

            # 3. Band Expansion (バンドの拡大 - ボラティリティ上昇)
            if current_width >= self.expansion_threshold:
                expansion_ratio = current_width / self.expansion_threshold
                pattern_completeness = min(
                    1.0, (expansion_ratio - 1.0) / 2.0
                )  # How much above threshold

                pattern_factors = {
                    "trend": self._calculate_trend_strength(
                        data, index, int(0.4 * 20)
                    ),  # Expansion can occur in any trend
                    "candle_size": self._calculate_candle_size_confidence(
                        data, index, 0.6
                    ),  # Larger candles during expansion
                    "price_movement": self._calculate_price_movement_confidence(
                        data, index, 0.7
                    ),  # High movement during expansion
                    "pattern_completeness": pattern_completeness,  # How expanded the bands are
                }

                # Adaptive confidence with volatility consideration
                volatility_boost = min(0.1, volatility_ratio * 0.1)
                confidence = self._calculate_pattern_confidence(
                    data, index, pattern_factors, base_confidence=0.6 + volatility_boost
                )

                return SignalResult(
                    signal_type="bb_expansion",
                    strength=confidence,
                    direction=0.0,  # Neutral signal - expansion indicates potential breakout
                    description=f"Bollinger Bands expansion detected (width: {current_width:.4f})",
                    metadata={
                        "width": current_width,
                        "threshold": self.expansion_threshold,
                        "expansion_ratio": expansion_ratio,
                        "pattern_completeness": pattern_completeness,
                        "volatility_ratio": volatility_ratio,
                        "trend_strength": trend_strength,
                    },
                    validity_period=3,
                    risk_level="high",
                )

            # 4. Middle Band Cross (ミドルバンドとのクロス)
            if prev_price <= prev_middle and current_price > current_middle:
                # Bullish cross of middle band
                cross_strength = abs(current_price - current_middle) / (
                    current_upper - current_middle
                )
                pattern_completeness = min(
                    1.0, cross_strength * 2.0
                )  # How strongly it crossed

                pattern_factors = {
                    "trend": self._calculate_trend_strength(
                        data, index, int(0.5 * 20)
                    ),  # Bullish cross favors uptrend
                    "candle_size": self._calculate_candle_size_confidence(
                        data, index, 0.5
                    ),  # Moderate candle size for cross
                    "price_movement": self._calculate_price_movement_confidence(
                        data, index, 0.6
                    ),  # Moderate movement for cross
                    "pattern_completeness": pattern_completeness,  # How strongly it crossed
                }

                # Adaptive direction based on cross strength and trend
                direction_factor = cross_strength * (0.6 + trend_strength * 0.4)
                direction = min(0.9, direction_factor)

                # Adaptive confidence with volatility consideration
                volatility_boost = min(0.1, volatility_ratio * 0.05)
                confidence = self._calculate_pattern_confidence(
                    data, index, pattern_factors, base_confidence=0.5 + volatility_boost
                )

                return SignalResult(
                    signal_type="bb_middle_cross_bullish",
                    strength=confidence,
                    direction=direction,
                    description=f"Price crossed above middle Bollinger Band at {current_price:.4f}",
                    metadata={
                        "cross_type": "bullish",
                        "middle_value": current_middle,
                        "prev_price": prev_price,
                        "current_price": current_price,
                        "cross_strength": cross_strength,
                        "pattern_completeness": pattern_completeness,
                        "volatility_ratio": volatility_ratio,
                        "trend_strength": trend_strength,
                    },
                    validity_period=2,
                    risk_level="medium",
                )

            elif prev_price >= prev_middle and current_price < current_middle:
                # Bearish cross of middle band
                cross_strength = abs(current_price - current_middle) / (
                    current_middle - current_lower
                )
                pattern_completeness = min(
                    1.0, cross_strength * 2.0
                )  # How strongly it crossed

                pattern_factors = {
                    "trend": self._calculate_trend_strength(
                        data, index, int(0.5 * 20)
                    ),  # Bearish cross favors downtrend
                    "candle_size": self._calculate_candle_size_confidence(
                        data, index, 0.5
                    ),  # Moderate candle size for cross
                    "price_movement": self._calculate_price_movement_confidence(
                        data, index, 0.6
                    ),  # Moderate movement for cross
                    "pattern_completeness": pattern_completeness,  # How strongly it crossed
                }

                # Adaptive direction based on cross strength and trend
                direction_factor = -cross_strength * (0.6 + trend_strength * 0.4)
                direction = max(-0.9, direction_factor)

                # Adaptive confidence with volatility consideration
                volatility_boost = min(0.1, volatility_ratio * 0.05)
                confidence = self._calculate_pattern_confidence(
                    data, index, pattern_factors, base_confidence=0.5 + volatility_boost
                )

                return SignalResult(
                    signal_type="bb_middle_cross_bearish",
                    strength=confidence,
                    direction=direction,
                    description=f"Price crossed below middle Bollinger Band at {current_price:.4f}",
                    metadata={
                        "cross_type": "bearish",
                        "middle_value": current_middle,
                        "prev_price": prev_price,
                        "current_price": current_price,
                        "cross_strength": cross_strength,
                        "pattern_completeness": pattern_completeness,
                        "volatility_ratio": volatility_ratio,
                        "trend_strength": trend_strength,
                    },
                    validity_period=2,
                    risk_level="medium",
                )

            # 5. Band Walk (バンド内での動き分析)
            # Price near upper band within bands
            if current_price > current_middle and current_price <= current_upper * 0.98:
                upper_position = (current_price - current_middle) / (
                    current_upper - current_middle
                )
                if upper_position > 0.7:
                    pattern_completeness = min(
                        1.0, (upper_position - 0.7) / 0.3
                    )  # How close to upper band

                    pattern_factors = {
                        "trend": self._calculate_trend_strength(
                            data, index, int(0.3 * 20)
                        ),  # Upper walk favors uptrend
                        "candle_size": self._calculate_candle_size_confidence(
                            data, index, 0.4
                        ),  # Smaller candles for walk
                        "price_movement": self._calculate_price_movement_confidence(
                            data, index, 0.3
                        ),  # Low movement for walk
                        "pattern_completeness": pattern_completeness,  # How close to upper band
                    }

                    # Adaptive confidence with trend consideration
                    trend_boost = min(0.1, trend_strength * 0.1)
                    confidence = self._calculate_pattern_confidence(
                        data, index, pattern_factors, base_confidence=0.3 + trend_boost
                    )

                    return SignalResult(
                        signal_type="bb_upper_walk",
                        strength=confidence,
                        direction=0.0,  # Neutral - monitoring position
                        description=f"Price walking upper band region (position: {upper_position:.2f})",
                        metadata={
                            "position": upper_position,
                            "region": "upper",
                            "middle": current_middle,
                            "upper": current_upper,
                            "pattern_completeness": pattern_completeness,
                            "volatility_ratio": volatility_ratio,
                            "trend_strength": trend_strength,
                        },
                        validity_period=1,
                        risk_level="low",
                    )

            # Price near lower band within bands
            elif (
                current_price < current_middle and current_price >= current_lower * 1.02
            ):
                lower_position = (current_middle - current_price) / (
                    current_middle - current_lower
                )
                if lower_position > 0.7:
                    pattern_completeness = min(
                        1.0, (lower_position - 0.7) / 0.3
                    )  # How close to lower band

                    pattern_factors = {
                        "trend": self._calculate_trend_strength(
                            data, index, int(0.3 * 20)
                        ),  # Lower walk favors downtrend
                        "candle_size": self._calculate_candle_size_confidence(
                            data, index, 0.4
                        ),  # Smaller candles for walk
                        "price_movement": self._calculate_price_movement_confidence(
                            data, index, 0.3
                        ),  # Low movement for walk
                        "pattern_completeness": pattern_completeness,  # How close to lower band
                    }

                    # Adaptive confidence with trend consideration
                    trend_boost = min(0.1, trend_strength * 0.1)
                    confidence = self._calculate_pattern_confidence(
                        data, index, pattern_factors, base_confidence=0.3 + trend_boost
                    )

                    return SignalResult(
                        signal_type="bb_lower_walk",
                        strength=confidence,
                        direction=0.0,  # Neutral - monitoring position
                        description=f"Price walking lower band region (position: {lower_position:.2f})",
                        metadata={
                            "position": lower_position,
                            "region": "lower",
                            "middle": current_middle,
                            "lower": current_lower,
                            "pattern_completeness": pattern_completeness,
                            "volatility_ratio": volatility_ratio,
                            "trend_strength": trend_strength,
                        },
                        validity_period=1,
                        risk_level="low",
                    )

            # 6. Neutral zone (ミドルバンド付近)
            middle_distance = abs(current_price - current_middle) / current_middle
            if middle_distance <= 0.01:  # Within 1% of middle band
                return SignalResult(
                    signal_type="bb_neutral",
                    strength=0.0,
                    direction=ACTION_HOLD,
                    description=f"Price near middle Bollinger Band (distance: {middle_distance:.4f})",
                    metadata={
                        "middle_distance": middle_distance,
                        "middle_value": current_middle,
                        "width": current_width,
                    },
                    validity_period=1,
                    risk_level="low",
                )

            # Default: no significant signal
            return SignalResult(
                signal_type="bb_neutral",
                strength=0.0,
                direction=ACTION_HOLD,
                description="Price within normal Bollinger Bands range",
                metadata={
                    "price": current_price,
                    "upper": current_upper,
                    "middle": current_middle,
                    "lower": current_lower,
                    "width": current_width,
                    "upper_distance": (current_upper - current_price) / current_price,
                    "lower_distance": (current_price - current_lower) / current_price,
                },
                validity_period=1,
                risk_level="low",
            )

        except Exception as e:
            return SignalResult(
                signal_type="bb_error",
                strength=0.0,
                direction=ACTION_HOLD,
                description=f"Bollinger Bands calculation error: {str(e)}",
                metadata={"error": str(e)},
                validity_period=1,
                risk_level="low",
            )

    def _build_analysis_view(
        self,
        data: pd.DataFrame,
        resolved_index: int,
    ) -> tuple[pd.DataFrame, int]:
        """Slice bounded history up to `resolved_index` for cheaper indicator recomputation."""
        configured_window = self.config.get("analysis_window")
        default_window = max(120, self.period * 8)
        try:
            window_size = (
                int(configured_window)
                if configured_window is not None
                else default_window
            )
        except (TypeError, ValueError):
            window_size = default_window

        window_size = max(self.period, min(window_size, 1200))
        start_idx = max(0, resolved_index - window_size + 1)
        view = data.iloc[start_idx : resolved_index + 1]
        if len(view) < self.period:
            view = data.iloc[: resolved_index + 1]
        return view, len(view) - 1
