"""
RSI (Relative Strength Index) Pattern Recognizer
既存のRSI特徴量クラスを使用したパターン認識
"""

from typing import Optional, TypedDict

import pandas as pd

from ztb.features.generators.technical.momentum.rsi import compute_rsi
from ztb.trading.strategies.action_signal_guide.pattern_recognition.base import (
    IndicatorPatternRecognizer,
    MultiTimeframeData,
    SignalResult,
)


class RSIRegimeThresholds(TypedDict):
    """Regime-adjusted RSI threshold levels."""

    overbought_level: float
    oversold_level: float


class RSIPatternRecognizer(IndicatorPatternRecognizer):
    """
    RSI-based pattern recognition using existing RSI feature class.
    既存のRSI特徴量クラスを使用したパターン認識
    """

    def __init__(self, config: Optional[dict[str, object]] = None):
        super().__init__(config)
        self.rsi_period = int(self.config.get("rsi_period", 14))
        self.overbought_level = float(self.config.get("overbought_level", 70))
        self.oversold_level = float(self.config.get("oversold_level", 30))
        self.divergence_lookback = int(self.config.get("divergence_lookback", 5))

    def recognize(
        self,
        data: pd.DataFrame,
        index: int = -1,
        multi_timeframe_data: Optional[MultiTimeframeData] = None,
    ) -> Optional[SignalResult]:
        """
        Recognize RSI-based patterns.
        RSIベースのパターン認識
        """
        min_periods = self.rsi_period + self.divergence_lookback
        resolved_index = self.resolve_indicator_index(
            data, index, min_required_periods=min_periods
        )
        if resolved_index is None:
            return None

        analysis_data, local_index = self.build_indicator_view(
            data,
            resolved_index,
            min_required_periods=min_periods,
            window_multiplier=10,
            min_window=140,
            max_window=800,
        )

        rsi_values = compute_rsi(analysis_data, period=self.rsi_period)
        if rsi_values.empty or rsi_values.isna().all():
            return None

        current_rsi = float(rsi_values.iloc[local_index])
        previous_rsi = float(rsi_values.iloc[max(0, local_index - 1)])
        if not pd.notna(current_rsi) or not pd.notna(previous_rsi):
            return None

        market_context = self.calculate_market_context(
            analysis_data, local_index, volatility_lookback=20, trend_window=20
        )
        volatility_ratio = market_context.volatility_ratio
        trend_strength = market_context.trend_strength

        mtf_confidence = self._analyze_multi_timeframe_rsi_alignment(
            current_rsi, previous_rsi, multi_timeframe_data
        )

        regime_adjusted_thresholds = (
            self._adjust_thresholds_for_regime(multi_timeframe_data)
            if self.regime_aware
            else {
                "overbought_level": self.overbought_level,
                "oversold_level": self.oversold_level,
            }
        )

        # Check for overbought/oversold signals
        if (
            current_rsi <= regime_adjusted_thresholds["oversold_level"]
            and previous_rsi > regime_adjusted_thresholds["oversold_level"]
        ):
            oversold_depth = self.clamp(
                self.safe_ratio(
                    self.oversold_level - current_rsi,
                    max(self.oversold_level, 1e-9),
                    default=0.0,
                ),
                0.0,
                1.0,
            )
            base_strength = min(1.0, oversold_depth * 0.8 + 0.2)

            direction_factor = oversold_depth * (0.8 + trend_strength * 0.2)
            direction = min(1.0, direction_factor)

            volatility_boost = min(0.2, volatility_ratio * 0.1)
            strength = min(1.0, base_strength + volatility_boost) * mtf_confidence

            return SignalResult(
                signal_type="RSI_oversold_mtf",
                strength=strength,
                direction=direction,
                description=f"RSI oversold signal MTF (RSI: {current_rsi:.2f}, MTF: {mtf_confidence:.2f})",
                metadata={
                    "rsi_value": current_rsi,
                    "volatility_ratio": volatility_ratio,
                    "trend_strength": trend_strength,
                    "oversold_depth": oversold_depth,
                    "mtf_confidence": mtf_confidence,
                    "regime_adjusted": True,
                },
                validity_period=3,
                risk_level="medium",
            )

        elif (
            current_rsi >= regime_adjusted_thresholds["overbought_level"]
            and previous_rsi < regime_adjusted_thresholds["overbought_level"]
        ):
            overbought_depth = self.clamp(
                self.safe_ratio(
                    current_rsi - self.overbought_level,
                    max(100.0 - self.overbought_level, 1e-9),
                    default=0.0,
                ),
                0.0,
                1.0,
            )
            base_strength = overbought_depth

            direction_factor = -overbought_depth * (0.8 + trend_strength * 0.2)
            direction = max(-1.0, direction_factor)

            volatility_boost = min(0.2, volatility_ratio * 0.1)
            strength = min(1.0, base_strength + volatility_boost) * mtf_confidence

            return SignalResult(
                signal_type="RSI_overbought_mtf",
                strength=strength,
                direction=direction,
                description=f"RSI overbought signal MTF (RSI: {current_rsi:.2f}, MTF: {mtf_confidence:.2f})",
                metadata={
                    "rsi_value": current_rsi,
                    "volatility_ratio": volatility_ratio,
                    "trend_strength": trend_strength,
                    "overbought_depth": overbought_depth,
                    "mtf_confidence": mtf_confidence,
                    "regime_adjusted": True,
                },
                validity_period=3,
                risk_level="medium",
            )

        divergence_signal = self._check_divergence(
            analysis_data,
            rsi_values,
            local_index,
            volatility_ratio,
            trend_strength,
            mtf_confidence,
        )
        if divergence_signal:
            return divergence_signal

        # Center line cross signals
        if previous_rsi <= 50 and current_rsi > 50:
            base_direction = 0.6
            trend_amplification = trend_strength * 0.4
            direction = min(1.0, base_direction + trend_amplification)

            base_strength = 0.5
            volatility_boost = min(0.1, volatility_ratio * 0.05)
            strength = min(0.8, base_strength + volatility_boost) * mtf_confidence

            return SignalResult(
                signal_type="RSI_centerline_bullish_mtf",
                strength=strength,
                direction=direction,
                description=f"RSI center line cross up MTF (RSI: {current_rsi:.2f}, MTF: {mtf_confidence:.2f})",
                metadata={
                    "rsi_value": current_rsi,
                    "volatility_ratio": volatility_ratio,
                    "trend_strength": trend_strength,
                    "mtf_confidence": mtf_confidence,
                    "regime_adjusted": True,
                },
                validity_period=2,
                risk_level="low",
            )

        if previous_rsi >= 50 and current_rsi < 50:
            base_direction = -0.6
            trend_amplification = trend_strength * 0.4
            direction = max(-1.0, base_direction - trend_amplification)

            base_strength = 0.5
            volatility_boost = min(0.1, volatility_ratio * 0.05)
            strength = min(0.8, base_strength + volatility_boost) * mtf_confidence

            return SignalResult(
                signal_type="RSI_centerline_bearish_mtf",
                strength=strength,
                direction=direction,
                description=f"RSI center line cross down MTF (RSI: {current_rsi:.2f}, MTF: {mtf_confidence:.2f})",
                metadata={
                    "rsi_value": current_rsi,
                    "volatility_ratio": volatility_ratio,
                    "trend_strength": trend_strength,
                    "mtf_confidence": mtf_confidence,
                    "regime_adjusted": True,
                },
                validity_period=2,
                risk_level="low",
            )

        return None

    def _check_divergence(
        self,
        data: pd.DataFrame,
        rsi_values: pd.Series,
        index: int,
        volatility_ratio: float = 1.0,
        trend_strength: float = 0.5,
        mtf_confidence: float = 1.0,
    ) -> Optional[SignalResult]:
        """
        Check for RSI divergence patterns.
        RSIダイバージェンスパターンのチェック
        """
        if len(rsi_values) < self.divergence_lookback + 2:
            return None

        start_idx = max(0, index - self.divergence_lookback)
        recent_prices = data["close"].iloc[start_idx : index + 1]
        recent_rsi = rsi_values.iloc[start_idx : index + 1]

        if len(recent_prices) < 2 or len(recent_rsi) < 2:
            return None

        # Bullish divergence (price down, RSI up)
        price_down = recent_prices.iloc[-1] < recent_prices.iloc[0]
        rsi_up = recent_rsi.iloc[-1] > recent_rsi.iloc[0]

        if price_down and rsi_up:
            base_strength = 0.5
            volatility_boost = min(0.2, volatility_ratio * 0.1)
            strength = min(0.8, base_strength + volatility_boost) * mtf_confidence

            base_direction = 0.7
            trend_amplification = trend_strength * 0.3
            direction = min(1.0, base_direction + trend_amplification)

            return SignalResult(
                signal_type="RSI_bullish_divergence_mtf",
                strength=strength,
                direction=direction,
                description=f"RSI bullish divergence detected MTF (MTF: {mtf_confidence:.2f})",
                metadata={
                    "volatility_ratio": volatility_ratio,
                    "trend_strength": trend_strength,
                    "mtf_confidence": mtf_confidence,
                    "divergence_type": "bullish",
                    "regime_adjusted": True,
                },
                validity_period=4,
                risk_level="medium",
            )

        # Bearish divergence (price up, RSI down)
        price_up = recent_prices.iloc[-1] > recent_prices.iloc[0]
        rsi_down = recent_rsi.iloc[-1] < recent_rsi.iloc[0]

        if price_up and rsi_down:
            base_strength = 0.5
            volatility_boost = min(0.2, volatility_ratio * 0.1)
            strength = min(0.8, base_strength + volatility_boost) * mtf_confidence

            base_direction = -0.7
            trend_amplification = trend_strength * 0.3
            direction = max(-1.0, base_direction - trend_amplification)

            return SignalResult(
                signal_type="RSI_bearish_divergence_mtf",
                strength=strength,
                direction=direction,
                description=f"RSI bearish divergence detected MTF (MTF: {mtf_confidence:.2f})",
                metadata={
                    "volatility_ratio": volatility_ratio,
                    "trend_strength": trend_strength,
                    "mtf_confidence": mtf_confidence,
                    "divergence_type": "bearish",
                    "regime_adjusted": True,
                },
                validity_period=4,
                risk_level="medium",
            )

        return None

    def _analyze_multi_timeframe_rsi_alignment(
        self,
        current_rsi: float,
        previous_rsi: float,
        multi_timeframe_data: Optional[MultiTimeframeData],
    ) -> float:
        """Analyze multi-timeframe RSI alignment for enhanced signal confidence."""
        rsi_change = current_rsi - previous_rsi
        return self.calculate_mtf_confidence(
            multi_timeframe_data,
            momentum_key="higher_timeframe_momentum",
            momentum_delta=rsi_change,
            regime_boost_clusters=(0,),
        )

    def _adjust_thresholds_for_regime(
        self,
        multi_timeframe_data: Optional[MultiTimeframeData],
        pattern_type: str = "general",
    ) -> RSIRegimeThresholds:
        """
        Adjust RSI thresholds based on market regime.

        Args:
            multi_timeframe_data: Multi-timeframe data containing regime info

        Returns:
            Adjusted thresholds dictionary
        """
        del pattern_type  # Reserved for future pattern-specific tuning.

        base_thresholds: RSIRegimeThresholds = {
            "overbought_level": self.overbought_level,
            "oversold_level": self.oversold_level,
        }

        if not self.regime_aware or not multi_timeframe_data:
            return base_thresholds

        regime_cluster = self._extract_regime_cluster(multi_timeframe_data, default=1)

        if regime_cluster == 0:
            return {
                "overbought_level": min(75.0, self.overbought_level + 5.0),
                "oversold_level": max(25.0, self.oversold_level - 5.0),
            }
        if regime_cluster == 2:
            return {
                "overbought_level": max(80.0, self.overbought_level + 10.0),
                "oversold_level": min(20.0, self.oversold_level - 10.0),
            }

        return base_thresholds
