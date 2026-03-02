"""
MACD (Moving Average Convergence Divergence) Pattern Recognizer
既存のMACD特徴量クラスを使用したパターン認識
"""

from typing import TypedDict

import pandas as pd

from ztb.features.generators.technical.momentum.macd import compute_macd
from ztb.trading.strategies.action_signal_guide.pattern_recognition.base import (
    IndicatorPatternRecognizer,
    MultiTimeframeData,
    SignalResult,
)

class MACDRegimeThresholds(TypedDict):
    """Regime-adjusted MACD thresholds."""

    histogram_threshold: float

class MACDPatternRecognizer(IndicatorPatternRecognizer):
    """
    MACD-based pattern recognition using existing MACD feature class.
    既存のMACD特徴量クラスを使用したパターン認識
    """

    def __init__(self, config: dict[str, object] | None = None):
        super().__init__(config)
        self.fast_period = int(self.config.get("fast_period", 12))
        self.slow_period = int(self.config.get("slow_period", 26))
        self.signal_period = int(self.config.get("signal_period", 9))
        self.histogram_threshold = float(self.config.get("histogram_threshold", 0.0))

    def recognize(
        self,
        data: pd.DataFrame,
        index: int = -1,
        multi_timeframe_data: MultiTimeframeData | None = None,
    ) -> SignalResult | None:
        """
        Recognize MACD-based patterns.
        MACDベースのパターン認識
        """
        min_periods = max(self.fast_period, self.slow_period) + self.signal_period
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
            min_window=180,
            max_window=1000,
        )

        market_context = self.calculate_market_context(
            analysis_data, local_index, volatility_lookback=30, trend_window=20
        )
        volatility_ratio = market_context.volatility_ratio
        trend_strength = market_context.trend_strength

        try:
            macd_hist = compute_macd(
                analysis_data,
                fast_period=self.fast_period,
                slow_period=self.slow_period,
                signal_period=self.signal_period,
            )
        except Exception:
            macd_hist = self._calculate_macd_manual(analysis_data)

        if macd_hist.empty or macd_hist.isna().all():
            return None

        current_hist = float(macd_hist.iloc[local_index])
        previous_hist = float(macd_hist.iloc[max(0, local_index - 1)])
        if not pd.notna(current_hist) or not pd.notna(previous_hist):
            return None

        hist_abs = macd_hist.abs().dropna()
        hist_abs_max = float(hist_abs.max()) if not hist_abs.empty else 1.0
        if hist_abs_max <= 0 or not pd.notna(hist_abs_max):
            hist_abs_max = 1.0

        hist_std_value = float(macd_hist.std()) if pd.notna(macd_hist.std()) else 0.0
        hist_std = abs(hist_std_value) if abs(hist_std_value) > 1e-9 else 1.0

        mtf_confidence = self._analyze_multi_timeframe_macd_alignment(
            current_hist, previous_hist, multi_timeframe_data
        )

        regime_adjusted_thresholds = (
            self._adjust_thresholds_for_regime(multi_timeframe_data)
            if self.regime_aware
            else {"histogram_threshold": self.histogram_threshold}
        )
        histogram_threshold = regime_adjusted_thresholds["histogram_threshold"]

        # Zero line cross signals
        if previous_hist <= 0 and current_hist > 0:
            base_strength = min(abs(current_hist) / hist_abs_max, 1.0)
            cross_strength = abs(current_hist) / hist_abs_max
            direction_factor = cross_strength * (0.8 + trend_strength * 0.2)
            direction = min(1.0, direction_factor)

            volatility_boost = min(0.2, volatility_ratio * 0.1)
            strength = min(1.0, base_strength + volatility_boost) * mtf_confidence

            return SignalResult(
                signal_type="MACD_zero_cross_bullish_mtf",
                strength=strength,
                direction=direction,
                description=f"MACD histogram zero line cross up MTF (Hist: {current_hist:.6f}, MTF: {mtf_confidence:.2f})",
                metadata={
                    "histogram_value": current_hist,
                    "volatility_ratio": volatility_ratio,
                    "trend_strength": trend_strength,
                    "cross_strength": cross_strength,
                    "mtf_confidence": mtf_confidence,
                    "regime_adjusted": True,
                },
                validity_period=4,
                risk_level="medium",
            )

        if previous_hist >= 0 and current_hist < 0:
            base_strength = min(abs(current_hist) / hist_abs_max, 1.0)
            cross_strength = abs(current_hist) / hist_abs_max
            direction_factor = -cross_strength * (0.8 + trend_strength * 0.2)
            direction = max(-1.0, direction_factor)

            volatility_boost = min(0.2, volatility_ratio * 0.1)
            strength = min(1.0, base_strength + volatility_boost) * mtf_confidence

            return SignalResult(
                signal_type="MACD_zero_cross_bearish_mtf",
                strength=strength,
                direction=direction,
                description=f"MACD histogram zero line cross down MTF (Hist: {current_hist:.6f}, MTF: {mtf_confidence:.2f})",
                metadata={
                    "histogram_value": current_hist,
                    "volatility_ratio": volatility_ratio,
                    "trend_strength": trend_strength,
                    "cross_strength": cross_strength,
                    "mtf_confidence": mtf_confidence,
                    "regime_adjusted": True,
                },
                validity_period=4,
                risk_level="medium",
            )

        # Histogram momentum signals
        hist_change = current_hist - previous_hist
        if abs(hist_change) > histogram_threshold:
            if hist_change > 0 and current_hist > 0:
                base_strength = max(0.5, min(abs(hist_change) / hist_std, 0.5) + 0.1)
                momentum_factor = abs(hist_change) / hist_std
                direction_factor = momentum_factor * (0.6 + trend_strength * 0.4)
                direction = min(0.8, direction_factor)

                volatility_boost = min(0.1, volatility_ratio * 0.05)
                strength = min(0.7, base_strength + volatility_boost) * mtf_confidence

                return SignalResult(
                    signal_type="MACD_bullish_momentum_mtf",
                    strength=strength,
                    direction=direction,
                    description=f"MACD bullish momentum MTF (Change: {hist_change:.6f}, MTF: {mtf_confidence:.2f})",
                    metadata={
                        "histogram_change": hist_change,
                        "volatility_ratio": volatility_ratio,
                        "trend_strength": trend_strength,
                        "momentum_factor": momentum_factor,
                        "mtf_confidence": mtf_confidence,
                        "regime_adjusted": True,
                    },
                    validity_period=3,
                    risk_level="low",
                )

            if hist_change < 0 and current_hist < 0:
                base_strength = max(0.5, min(abs(hist_change) / hist_std, 0.5) + 0.1)
                momentum_factor = abs(hist_change) / hist_std
                direction_factor = -momentum_factor * (0.6 + trend_strength * 0.4)
                direction = max(-0.8, direction_factor)

                volatility_boost = min(0.1, volatility_ratio * 0.05)
                strength = min(0.7, base_strength + volatility_boost) * mtf_confidence

                return SignalResult(
                    signal_type="MACD_bearish_momentum_mtf",
                    strength=strength,
                    direction=direction,
                    description=f"MACD bearish momentum MTF (Change: {hist_change:.6f}, MTF: {mtf_confidence:.2f})",
                    metadata={
                        "histogram_change": hist_change,
                        "volatility_ratio": volatility_ratio,
                        "trend_strength": trend_strength,
                        "momentum_factor": momentum_factor,
                        "mtf_confidence": mtf_confidence,
                        "regime_adjusted": True,
                    },
                    validity_period=3,
                    risk_level="low",
                )

        convergence_signal = self._check_convergence(
            analysis_data, macd_hist, local_index, volatility_ratio, trend_strength
        )
        if convergence_signal:
            return convergence_signal

        return None

    def calculate(self, data: pd.DataFrame) -> dict[str, pd.Series]:
        """Calculate MACD line, signal line, and histogram."""
        close = data["close"].astype(float)
        ema_fast = close.ewm(span=self.fast_period, adjust=False).mean()
        ema_slow = close.ewm(span=self.slow_period, adjust=False).mean()

        macd_line = ema_fast - ema_slow
        signal_line = macd_line.ewm(span=self.signal_period, adjust=False).mean()
        histogram = macd_line - signal_line

        return {
            "macd": macd_line.fillna(0.0),
            "signal": signal_line.fillna(0.0),
            "histogram": histogram.fillna(0.0),
        }

    def _calculate_macd_manual(self, data: pd.DataFrame) -> pd.Series:
        """
        Manual MACD calculation as fallback.
        TaLibが失敗した場合の手動MACD計算
        """
        return self.calculate(data)["histogram"]

    def _check_convergence(
        self,
        data: pd.DataFrame,
        macd_hist: pd.Series,
        index: int,
        volatility_ratio: float = 1.0,
        trend_strength: float = 0.5,
    ) -> SignalResult | None:
        """
        Check for MACD convergence/divergence patterns.
        MACD収束/発散パターンのチェック
        """
        if len(macd_hist) < 10:
            return None

        start_idx = max(0, index - 9)
        recent_prices = data["close"].iloc[start_idx : index + 1]
        recent_hist = macd_hist.iloc[start_idx : index + 1]

        if len(recent_prices) < 2 or len(recent_hist) < 2:
            return None

        # Bullish convergence (price down, MACD up)
        price_down = recent_prices.iloc[-1] < recent_prices.iloc[0]
        hist_up = recent_hist.iloc[-1] > recent_hist.iloc[0]

        if price_down and hist_up:
            base_strength = 0.5
            volatility_boost = min(0.2, volatility_ratio * 0.1)
            strength = min(0.8, base_strength + volatility_boost)

            base_direction = 0.6
            trend_amplification = trend_strength * 0.4
            direction = min(1.0, base_direction + trend_amplification)

            return SignalResult(
                signal_type="MACD_bullish_convergence",
                strength=strength,
                direction=direction,
                description="MACD bullish convergence detected",
                confidence=min(0.8, strength * 1.0),
                metadata={
                    "volatility_ratio": volatility_ratio,
                    "trend_strength": trend_strength,
                    "pattern_type": "bullish_convergence",
                },
            )

        # Bearish divergence (price up, MACD down)
        price_up = recent_prices.iloc[-1] > recent_prices.iloc[0]
        hist_down = recent_hist.iloc[-1] < recent_hist.iloc[0]

        if price_up and hist_down:
            base_strength = 0.5
            volatility_boost = min(0.2, volatility_ratio * 0.1)
            strength = min(0.8, base_strength + volatility_boost)

            base_direction = -0.6
            trend_amplification = trend_strength * 0.4
            direction = max(-1.0, base_direction - trend_amplification)

            return SignalResult(
                signal_type="MACD_bearish_divergence",
                strength=strength,
                direction=direction,
                description="MACD bearish divergence detected",
                confidence=min(0.8, strength * 1.0),
                metadata={
                    "volatility_ratio": volatility_ratio,
                    "trend_strength": trend_strength,
                    "pattern_type": "bearish_divergence",
                },
            )

        return None

    def _analyze_multi_timeframe_macd_alignment(
        self,
        current_hist: float,
        previous_hist: float,
        multi_timeframe_data: MultiTimeframeData | None,
    ) -> float:
        """Analyze multi-timeframe MACD alignment for enhanced signal confidence."""
        hist_change = current_hist - previous_hist
        return self.calculate_mtf_confidence(
            multi_timeframe_data,
            momentum_key="higher_timeframe_momentum",
            momentum_delta=hist_change,
            regime_boost_clusters=(2,),
        )

    def _adjust_thresholds_for_regime(
        self,
        multi_timeframe_data: MultiTimeframeData | None,
        pattern_type: str = "general",
    ) -> MACDRegimeThresholds:
        """
        Adjust MACD thresholds based on market regime.

        Args:
            multi_timeframe_data: Multi-timeframe data containing regime info

        Returns:
            Adjusted thresholds dictionary
        """
        del pattern_type  # Reserved for future pattern-specific tuning.

        base_thresholds: MACDRegimeThresholds = {
            "histogram_threshold": self.histogram_threshold,
        }

        if not self.regime_aware or not multi_timeframe_data:
            return base_thresholds

        regime_cluster = self._extract_regime_cluster(multi_timeframe_data, default=1)

        if regime_cluster == 0:
            return {
                "histogram_threshold": max(0.0, self.histogram_threshold * 0.8),
            }
        if regime_cluster == 2:
            return {
                "histogram_threshold": min(0.1, self.histogram_threshold * 1.5),
            }

        return base_thresholds
