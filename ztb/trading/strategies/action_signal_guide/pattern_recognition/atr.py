"""
ATR (Average True Range) Pattern Recognizer
既存のATR特徴量クラスを使用したパターン認識
"""

from typing import TypedDict

import pandas as pd

from ztb.features.generators.technical.volatility.atr import compute_atr
from ztb.trading.strategies.action_signal_guide.pattern_recognition.base import (
    IndicatorPatternRecognizer,
    MultiTimeframeData,
    SignalResult,
)

class ATRRegimeThresholds(TypedDict):
    """Regime-adjusted ATR thresholds."""

    volatility_threshold: float
    low_volatility_threshold: float

class ATRPatternRecognizer(IndicatorPatternRecognizer):
    """
    ATR-based pattern recognition using existing ATR feature class.
    既存のATR特徴量クラスを使用したパターン認識
    """

    def __init__(self, config: dict[str, object] | None = None):
        super().__init__(config)
        self.atr_period = int(self.config.get("atr_period", 14))
        self.volatility_threshold = float(self.config.get("volatility_threshold", 1.0))
        self.trend_strength_period = int(self.config.get("trend_strength_period", 5))

    def recognize(
        self,
        data: pd.DataFrame,
        index: int = -1,
        multi_timeframe_data: MultiTimeframeData | None = None,
    ) -> SignalResult | None:
        """
        Recognize ATR-based patterns with multi-timeframe support.
        ATRベースのパターン認識（複数時間軸対応）
        """
        min_periods = self.atr_period + self.trend_strength_period
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
            min_window=160,
            max_window=1000,
        )

        atr_values = compute_atr(analysis_data, period=self.atr_period)
        if atr_values.empty or atr_values.isna().all():
            return None

        current_atr = float(atr_values.iloc[local_index])
        if not pd.notna(current_atr):
            return None

        avg_window = atr_values.iloc[max(0, local_index - 19) : local_index + 1].dropna()
        avg_atr = float(avg_window.mean()) if not avg_window.empty else current_atr
        if not pd.notna(avg_atr) or avg_atr <= 0:
            avg_atr = max(abs(current_atr), 1e-9)

        mtf_confidence = self._analyze_multi_timeframe_volatility(
            current_atr, avg_atr, multi_timeframe_data
        )

        regime_adjusted_thresholds = (
            self._adjust_thresholds_for_regime(multi_timeframe_data)
            if self.regime_aware
            else {
                "volatility_threshold": self.volatility_threshold,
                "low_volatility_threshold": 0.8,
            }
        )

        volatility_ratio = self.safe_ratio(current_atr, avg_atr, default=1.0)

        # Volatility breakout signals with multi-timeframe confirmation
        if current_atr > avg_atr * regime_adjusted_thresholds["volatility_threshold"]:
            breakout_signal = self._analyze_breakout_mtf(
                analysis_data,
                current_atr,
                avg_atr,
                local_index,
                mtf_confidence,
                regime_adjusted_thresholds,
            )
            if breakout_signal:
                return breakout_signal

        # Trend strength analysis using ATR with multi-timeframe
        trend_signal = self._analyze_trend_strength_mtf(
            analysis_data, atr_values, local_index, mtf_confidence
        )
        if trend_signal:
            return trend_signal

        # Low volatility consolidation with regime context
        if (
            current_atr
            < avg_atr * regime_adjusted_thresholds["low_volatility_threshold"]
        ):
            low_vol_gap = self.safe_ratio(
                avg_atr * regime_adjusted_thresholds["low_volatility_threshold"] - current_atr,
                avg_atr,
                default=0.0,
            )
            strength = max(0.1, low_vol_gap) * mtf_confidence
            return SignalResult(
                signal_type="ATR_low_volatility_mtf",
                strength=strength,
                direction=0.0,
                description=f"ATR low volatility consolidation (ATR: {current_atr:.6f}, MTF: {mtf_confidence:.2f})",
                metadata={
                    "atr": current_atr,
                    "avg_atr": avg_atr,
                    "volatility_ratio": volatility_ratio,
                    "mtf_confidence": mtf_confidence,
                    "regime_adjusted": True,
                },
                validity_period=2,
                risk_level="low",
            )

        return None

    def _analyze_breakout(
        self, data: pd.DataFrame, current_atr: float, avg_atr: float, index: int
    ) -> SignalResult | None:
        """
        Analyze potential breakout during high volatility.
        高ボラティリティ時のブレイクアウト分析
        """
        base_thresholds: ATRRegimeThresholds = {
            "volatility_threshold": self.volatility_threshold,
            "low_volatility_threshold": 0.8,
        }
        return self._analyze_breakout_mtf(
            data,
            current_atr,
            avg_atr,
            index,
            mtf_confidence=1.0,
            regime_adjusted_thresholds=base_thresholds,
        )

    def _analyze_trend_strength(
        self, data: pd.DataFrame, atr_values: pd.Series, index: int
    ) -> SignalResult | None:
        """
        Analyze trend strength using ATR changes.
        ATR変化によるトレンド強度分析
        """
        return self._analyze_trend_strength_mtf(
            data,
            atr_values,
            index,
            mtf_confidence=1.0,
        )

    def _analyze_multi_timeframe_volatility(
        self,
        current_atr: float,
        avg_atr: float,
        multi_timeframe_data: MultiTimeframeData | None,
    ) -> float:
        """Analyze multi-timeframe volatility alignment for enhanced confidence."""
        volatility_delta = self.safe_ratio(current_atr, avg_atr, default=1.0) - 1.0
        return self.calculate_mtf_confidence(
            multi_timeframe_data,
            momentum_key="higher_timeframe_volatility",
            momentum_delta=volatility_delta,
            high_threshold=1.2,
            low_threshold=0.8,
            regime_boost_clusters=(2,),
        )

    def _adjust_thresholds_for_regime(
        self,
        multi_timeframe_data: MultiTimeframeData | None,
        pattern_type: str = "general",
    ) -> ATRRegimeThresholds:
        """
        Adjust ATR thresholds based on market regime.

        Args:
            multi_timeframe_data: Multi-timeframe data containing regime info

        Returns:
            Adjusted thresholds dictionary
        """
        del pattern_type  # Reserved for future pattern-specific tuning.

        base_thresholds: ATRRegimeThresholds = {
            "volatility_threshold": self.volatility_threshold,
            "low_volatility_threshold": 0.8,
        }

        if not self.regime_aware or not multi_timeframe_data:
            return base_thresholds

        regime_cluster = self._extract_regime_cluster(multi_timeframe_data, default=1)

        if regime_cluster == 0:
            return {
                "volatility_threshold": max(0.8, self.volatility_threshold * 0.9),
                "low_volatility_threshold": 0.75,
            }
        if regime_cluster == 2:
            return {
                "volatility_threshold": min(1.5, self.volatility_threshold * 1.3),
                "low_volatility_threshold": 0.9,
            }

        return base_thresholds

    def _analyze_breakout_mtf(
        self,
        data: pd.DataFrame,
        current_atr: float,
        avg_atr: float,
        index: int,
        mtf_confidence: float,
        regime_adjusted_thresholds: ATRRegimeThresholds,
    ) -> SignalResult | None:
        """
        Analyze potential breakout during high volatility with multi-timeframe confirmation.
        高ボラティリティ時のブレイクアウト分析（複数時間軸対応）
        """
        start_idx = max(0, index - 4)
        recent_prices = data["close"].iloc[start_idx : index + 1]

        if len(recent_prices) < 2:
            return None

        price_change = self.safe_ratio(
            float(recent_prices.iloc[-1] - recent_prices.iloc[0]),
            float(recent_prices.iloc[0]),
            default=0.0,
        )

        volatility_ratio = self.safe_ratio(current_atr, avg_atr, default=1.0)
        threshold = max(regime_adjusted_thresholds["volatility_threshold"], 1e-9)
        strength = min(volatility_ratio / threshold, 1.0) * mtf_confidence

        if abs(price_change) > 0.005:
            if price_change > 0:
                return SignalResult(
                    signal_type="ATR_bullish_breakout_mtf",
                    strength=strength,
                    direction=1.0,
                    description=f"ATR bullish breakout MTF (Vol: {volatility_ratio:.2f}, Price: +{price_change:.2%}, MTF: {mtf_confidence:.2f})",
                    metadata={
                        "atr": current_atr,
                        "avg_atr": avg_atr,
                        "volatility_ratio": volatility_ratio,
                        "price_change": price_change,
                        "mtf_confidence": mtf_confidence,
                        "regime_adjusted": True,
                    },
                    validity_period=3,
                    risk_level="medium",
                )

            return SignalResult(
                signal_type="ATR_bearish_breakout_mtf",
                strength=strength,
                direction=-1.0,
                description=f"ATR bearish breakout MTF (Vol: {volatility_ratio:.2f}, Price: {price_change:.2%}, MTF: {mtf_confidence:.2f})",
                metadata={
                    "atr": current_atr,
                    "avg_atr": avg_atr,
                    "volatility_ratio": volatility_ratio,
                    "price_change": price_change,
                    "mtf_confidence": mtf_confidence,
                    "regime_adjusted": True,
                },
                validity_period=3,
                risk_level="medium",
            )

        return SignalResult(
            signal_type="ATR_high_volatility_mtf",
            strength=strength * 0.5,
            direction=0.0,
            description=f"ATR high volatility MTF, awaiting direction (Vol: {volatility_ratio:.2f}, MTF: {mtf_confidence:.2f})",
            metadata={
                "atr": current_atr,
                "avg_atr": avg_atr,
                "volatility_ratio": volatility_ratio,
                "mtf_confidence": mtf_confidence,
                "regime_adjusted": True,
            },
            validity_period=2,
            risk_level="low",
        )

    def _analyze_trend_strength_mtf(
        self,
        data: pd.DataFrame,
        atr_values: pd.Series,
        index: int,
        mtf_confidence: float,
    ) -> SignalResult | None:
        """
        Analyze trend strength using ATR changes with multi-timeframe support.
        ATR変化によるトレンド強度分析（複数時間軸対応）
        """
        if len(atr_values) < self.trend_strength_period + 5:
            return None

        start_idx = max(0, index - self.trend_strength_period + 1)
        recent_atr = atr_values.iloc[start_idx : index + 1]
        recent_prices = data["close"].iloc[start_idx : index + 1]

        if len(recent_atr) < 2 or len(recent_prices) < 2:
            return None

        atr_trend = self.safe_ratio(
            float(recent_atr.iloc[-1] - recent_atr.iloc[0]),
            float(recent_atr.iloc[0]),
            default=0.0,
        )
        price_trend = self.safe_ratio(
            float(recent_prices.iloc[-1] - recent_prices.iloc[0]),
            float(recent_prices.iloc[0]),
            default=0.0,
        )

        # Strong trend with increasing ATR (healthy trend) with MTF confirmation
        if abs(price_trend) > 0.01 and atr_trend > 0.05:
            if price_trend > 0:
                strength = min(abs(price_trend) * 10, 0.6) * mtf_confidence
                return SignalResult(
                    signal_type="ATR_strong_bullish_trend_mtf",
                    strength=strength,
                    direction=1.0,
                    description=f"ATR strong bullish trend MTF (Price: +{price_trend:.2%}, ATR: +{atr_trend:.2%}, MTF: {mtf_confidence:.2f})",
                    metadata={
                        "atr_trend": atr_trend,
                        "price_trend": price_trend,
                        "mtf_confidence": mtf_confidence,
                        "regime_adjusted": True,
                    },
                    validity_period=4,
                    risk_level="medium",
                )

            strength = min(abs(price_trend) * 10, 0.6) * mtf_confidence
            return SignalResult(
                signal_type="ATR_strong_bearish_trend_mtf",
                strength=strength,
                direction=-1.0,
                description=f"ATR strong bearish trend MTF (Price: {price_trend:.2%}, ATR: +{atr_trend:.2%}, MTF: {mtf_confidence:.2f})",
                metadata={
                    "atr_trend": atr_trend,
                    "price_trend": price_trend,
                    "mtf_confidence": mtf_confidence,
                    "regime_adjusted": True,
                },
                validity_period=4,
                risk_level="medium",
            )

        # Weak trend with decreasing ATR (potential reversal) with MTF context
        if abs(price_trend) < 0.005 and atr_trend < -0.05:
            strength = 0.3 * mtf_confidence
            return SignalResult(
                signal_type="ATR_weakening_trend_mtf",
                strength=strength,
                direction=0.0,
                description=f"ATR weakening trend MTF (Price: {price_trend:.2%}, ATR: {atr_trend:.2%}, MTF: {mtf_confidence:.2f})",
                metadata={
                    "atr_trend": atr_trend,
                    "price_trend": price_trend,
                    "mtf_confidence": mtf_confidence,
                    "regime_adjusted": True,
                },
                validity_period=3,
                risk_level="low",
            )

        return None
