"""
Ensemble Signal Methods for Phase 3

多ソースシグナル統合による高度なアンサンブル手法
信頼度計算と動的ウェイト最適化を実装
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Protocol, TypedDict

import numpy as np
import pandas as pd

from ztb.trading.signal.common.utilities import clamp_value, normalize_weights

# Remove circular import - will import locally when needed
# from ztb.trading.signal.quality_scorer import SignalQualityScorer
from ztb.utils.logging_utils import get_logger

logger = get_logger(__name__)

SourceScores = dict[str, float]
SourceConfidences = dict[str, float]
SourceWeights = dict[str, float]

class MarketData(TypedDict, total=False):
    """Input payload for ensemble scoring."""

    df: pd.DataFrame
    continuous_action: float
    portfolio: Mapping[str, float]

class SignalReliability(TypedDict, total=False):
    """Structured reliability payload returned by the ensemble generator."""

    ensemble_score: float
    ensemble_confidence: float
    source_scores: SourceScores
    source_confidences: SourceConfidences
    source_reliabilities: SourceConfidences
    agreement_level: float
    overall_reliability: float
    signal_strength: str
    recommendation: str
    error: str

class SignalScorer(Protocol):
    """Protocol used to type heterogeneous scorer instances."""

    name: str

    def calculate_score(self, market_data: MarketData) -> float:
        ...

    def get_confidence(self, market_data: MarketData) -> float:
        ...

def _to_float(value: object, default: float = 0.0) -> float:
    """Convert object to float with safe fallback."""
    try:
        return float(value)
    except (TypeError, ValueError):
        return default

def _to_int(value: object, default: int = 0) -> int:
    """Convert object to int with safe fallback."""
    try:
        return int(value)
    except (TypeError, ValueError):
        return default

def _to_bool(value: object, default: bool = False) -> bool:
    """Convert object to bool with safe fallback."""
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"true", "1", "yes", "on"}:
            return True
        if normalized in {"false", "0", "no", "off"}:
            return False
    return default

class BaseSignalScorer:
    """Base class for signal scorers"""

    def __init__(self, config: Mapping[str, object] | None = None):
        self.config: dict[str, object] = dict(config) if config is not None else {}
        # Keep class name lowercased (avoid truncating 'Scorer') so tests
        # expecting 'basesignalscorer' still pass
        self.name = self.__class__.__name__.lower()

    @staticmethod
    def _get_dataframe(
        market_data: MarketData, min_rows: int = 1
    ) -> pd.DataFrame | None:
        """Extract market dataframe and validate minimum row count."""
        df = market_data.get("df")
        if isinstance(df, pd.DataFrame) and len(df) >= min_rows:
            return df
        return None

    @staticmethod
    def _bounded_score(score: float) -> float:
        """Clamp score to the expected [0, 100] range."""
        return float(clamp_value(float(score), 0.0, 100.0))

    @staticmethod
    def _bounded_confidence(confidence: float, lower: float, upper: float) -> float:
        """Clamp confidence to configured bounds."""
        return float(clamp_value(float(confidence), lower, upper))

    @staticmethod
    def _get_float(values: Mapping[str, object], key: str, default: float) -> float:
        """Read float-like mapping value with fallback."""
        return _to_float(values.get(key, default), default)

    def calculate_score(self, market_data: MarketData) -> float:
        """Calculate signal score (0-100)"""
        raise NotImplementedError

    def get_confidence(self, market_data: MarketData) -> float:
        """Get confidence level (0-1)"""
        return 0.5

class TechnicalSignalScorer(BaseSignalScorer):
    """Technical analysis based signal scorer"""

    def __init__(self, config: Mapping[str, object] | None = None):
        super().__init__(config)
        # Use TechnicalIndicators directly instead of SignalQualityScorer to avoid circular import
        from ztb.trading.signal.technical_indicators import TechnicalIndicators

        self.technical_indicators = TechnicalIndicators()

    def calculate_score(self, market_data: MarketData) -> float:
        """Calculate technical signal score using TechnicalIndicators"""
        try:
            df = self._get_dataframe(market_data, min_rows=1)
            if df is None:
                return 50.0

            # Get technical signals
            tech_signals = self.technical_indicators.get_technical_signals(df)

            # Simple scoring based on key indicators
            score = 50.0  # Neutral starting point

            # RSI scoring (0-100 scale)
            rsi = self._get_float(tech_signals, "rsi", 50.0)
            if rsi < 30:
                score += 20  # Oversold - bullish
            elif rsi > 70:
                score -= 20  # Overbought - bearish

            # MACD scoring
            macd_line = self._get_float(tech_signals, "macd", 0.0)
            signal_line = self._get_float(tech_signals, "macd_signal", 0.0)
            if macd_line > signal_line:
                score += 15  # Bullish crossover
            elif macd_line < signal_line:
                score -= 15  # Bearish crossover

            # Bollinger Band position
            bb_position = self._get_float(tech_signals, "bb_position", 0.5)
            if bb_position < 0.2:
                score += 10  # Near lower band - bullish
            elif bb_position > 0.8:
                score -= 10  # Near upper band - bearish

            return self._bounded_score(score)

        except Exception as e:
            logger.warning(f"Error in technical signal scoring: {e}")
            return 50.0

    def get_confidence(self, market_data: MarketData) -> float:
        """Get technical signal confidence"""
        try:
            df = self._get_dataframe(market_data, min_rows=20)
            if df is None:
                return 0.3

            # Confidence based on data quality and market conditions
            volatility = df["close"].pct_change().std()
            if pd.isna(volatility):
                return 0.3

            # Higher confidence with sufficient data and reasonable volatility
            confidence = 1.0 - float(volatility) * 10.0
            return self._bounded_confidence(confidence, lower=0.1, upper=0.9)

        except Exception:
            return 0.3

class PatternRecognitionScorer(BaseSignalScorer):
    """Pattern recognition based signal scorer"""

    _centered_x_cache: dict[int, np.ndarray] = {}

    def __init__(self, config: Mapping[str, object] | None = None):
        super().__init__(config)
        self.min_periods = _to_int(self.config.get("min_periods", 20), 20)

    @classmethod
    def _get_centered_x(cls, size: int) -> np.ndarray:
        """Return centered x values cached by window size."""
        centered_x = cls._centered_x_cache.get(size)
        if centered_x is None:
            centered_x = np.arange(size, dtype=np.float64)
            centered_x -= centered_x.mean()
            cls._centered_x_cache[size] = centered_x
        return centered_x

    @classmethod
    def _calculate_trend_slope(cls, prices: np.ndarray) -> float:
        """Fast O(n) trend slope approximation using centered regression."""
        if prices.size < 2:
            return 0.0
        centered_x = cls._get_centered_x(prices.size)
        centered_y = prices.astype(np.float64, copy=False) - float(np.mean(prices))
        denominator = float(np.dot(centered_x, centered_x))
        if denominator <= 0.0:
            return 0.0
        return float(np.dot(centered_x, centered_y) / denominator)

    def calculate_score(self, market_data: MarketData) -> float:
        """Calculate pattern-based signal score"""
        try:
            df = self._get_dataframe(market_data, min_rows=self.min_periods)
            if df is None:
                return 50.0

            # Simple pattern recognition: trend continuation vs reversal
            recent_prices = df["close"].tail(10).to_numpy(dtype=np.float64, copy=False)
            if recent_prices.size < 6:
                return 50.0

            # Calculate trend strength
            trend_slope = self._calculate_trend_slope(recent_prices)

            # Look for reversal patterns using prior candles (exclude current bar).
            reference_window = recent_prices[-6:-1]
            recent_high = float(np.max(reference_window))
            recent_low = float(np.min(reference_window))
            current_price = float(recent_prices[-1])

            # Reversal pattern detection
            if trend_slope > 0 and current_price < recent_low:
                # Potential reversal in uptrend
                return 30.0  # Bearish signal
            elif trend_slope < 0 and current_price > recent_high:
                # Potential reversal in downtrend
                return 70.0  # Bullish signal
            else:
                # Trend continuation: scale slope with tanh to guarantee 0-100 range
                # Use a moderate scale factor so typical slopes remain meaningful
                score = 50.0 + np.tanh(trend_slope * 2.0) * 50.0
                return self._bounded_score(float(score))

        except Exception as e:
            logger.warning(f"Error in pattern recognition scoring: {e}")
            return 50.0

    def get_confidence(self, market_data: MarketData) -> float:
        """Get pattern recognition confidence"""
        try:
            df = self._get_dataframe(market_data, min_rows=self.min_periods)
            if df is None:
                return 0.2

            # Confidence based on pattern clarity
            recent_volatility = df["close"].tail(10).pct_change().std()
            if pd.isna(recent_volatility):
                return 0.2
            confidence = 1.0 - float(recent_volatility) * 5.0
            return self._bounded_confidence(confidence, lower=0.1, upper=0.8)

        except Exception:
            return 0.2

class SentimentSignalScorer(BaseSignalScorer):
    """Sentiment-based signal scorer"""

    def __init__(self, config: Mapping[str, object] | None = None):
        super().__init__(config)
        self.sentiment_window = _to_int(self.config.get("sentiment_window", 24), 24)

    def calculate_score(self, market_data: MarketData) -> float:
        """Calculate sentiment-based signal score"""
        try:
            # For now, use price-based sentiment proxy
            # In production, this would integrate with social media, news, etc.
            df = self._get_dataframe(market_data, min_rows=10)
            if df is None:
                return 50.0

            # Price momentum as sentiment proxy
            short_momentum = df["close"].pct_change(5).iloc[-1]
            long_momentum = df["close"].pct_change(20).iloc[-1]
            if pd.isna(short_momentum) or pd.isna(long_momentum):
                return 50.0

            # Sentiment score based on momentum divergence
            if short_momentum > 0 and long_momentum > 0:
                return 65.0  # Positive sentiment
            elif short_momentum < 0 and long_momentum < 0:
                return 35.0  # Negative sentiment
            else:
                return 50.0  # Mixed sentiment

        except Exception as e:
            logger.warning(f"Error in sentiment scoring: {e}")
            return 50.0

    def get_confidence(self, market_data: MarketData) -> float:
        """Get sentiment confidence"""
        # Sentiment data is typically less reliable
        return 0.4

class VolumeProfileScorer(BaseSignalScorer):
    """Volume profile based signal scorer"""

    def __init__(self, config: Mapping[str, object] | None = None):
        super().__init__(config)
        self.volume_window = _to_int(self.config.get("volume_window", 20), 20)

    def calculate_score(self, market_data: MarketData) -> float:
        """Calculate volume-based signal score"""
        try:
            df = self._get_dataframe(market_data, min_rows=self.volume_window)
            if df is None or "volume" not in df.columns:
                return 50.0

            # Volume analysis
            recent_volume = df["volume"].tail(self.volume_window)
            avg_volume = float(recent_volume.mean())
            current_volume = float(recent_volume.iloc[-1])
            if avg_volume <= 0.0 or pd.isna(avg_volume):
                return 50.0

            # Price-volume relationship
            price_change = df["close"].pct_change().iloc[-1]
            if pd.isna(price_change):
                return 50.0
            volume_ratio = current_volume / avg_volume if avg_volume > 0 else 1.0

            # Volume confirmation
            if price_change > 0 and volume_ratio > 1.2:
                return 65.0  # Bullish volume confirmation
            elif price_change < 0 and volume_ratio > 1.2:
                return 35.0  # Bearish volume confirmation
            elif volume_ratio < 0.8:
                return 50.0  # Low volume, uncertain signal
            else:
                return 50.0  # Neutral

        except Exception as e:
            logger.warning(f"Error in volume profile scoring: {e}")
            return 50.0

    def get_confidence(self, market_data: MarketData) -> float:
        """Get volume profile confidence"""
        try:
            df = self._get_dataframe(market_data, min_rows=20)
            if df is None or "volume" not in df.columns:
                return 0.1

            # Confidence based on volume data quality
            recent_volume = df["volume"].tail(20)
            avg_volume = float(recent_volume.mean())
            if avg_volume <= 0.0:
                return 0.2
            volume_variation = float(recent_volume.std() / avg_volume)
            if np.isnan(volume_variation) or np.isinf(volume_variation):
                return 0.2
            return self._bounded_confidence(volume_variation * 2.0, lower=0.2, upper=0.9)

        except Exception:
            return 0.2

class EnsembleSignalGenerator:
    """アンサンブルシグナル生成器"""

    def __init__(self, config: Mapping[str, object] | None = None):
        self.config: dict[str, object] = self._get_default_config()
        if config:
            self.config.update(dict(config))

        available_sources: dict[str, SignalScorer] = {
            "technical": TechnicalSignalScorer(self._get_source_config("technical")),
            "pattern": PatternRecognitionScorer(self._get_source_config("pattern")),
            "sentiment": SentimentSignalScorer(self._get_source_config("sentiment")),
            "volume": VolumeProfileScorer(self._get_source_config("volume")),
        }

        configured_sources = self._get_enabled_source_names(
            self.config.get("signal_sources"), available_sources
        )
        self.signal_sources: dict[str, SignalScorer] = {
            name: available_sources[name] for name in configured_sources
        }

        self.ensemble_weights = self._resolve_ensemble_weights(configured_sources)
        self.enable_dynamic_weights = _to_bool(
            self.config.get("enable_dynamic_weights", True), True
        )

    @staticmethod
    def _get_default_config() -> dict[str, object]:
        """Get default configuration"""
        return {
            "ensemble_weights": {
                "technical": 0.4,
                "pattern": 0.3,
                "sentiment": 0.2,
                "volume": 0.1,
            },
            "enable_dynamic_weights": True,
            "min_confidence_threshold": 0.3,
            "technical": {},
            "pattern": {"min_periods": 20},
            "sentiment": {"sentiment_window": 24},
            "volume": {"volume_window": 20},
        }

    def _get_source_config(self, key: str) -> dict[str, object]:
        """Return per-source config dict with safe fallback."""
        config_value = self.config.get(key)
        if isinstance(config_value, Mapping):
            return {str(k): v for k, v in config_value.items() if isinstance(k, str)}
        return {}

    @staticmethod
    def _get_enabled_source_names(
        configured: object, available_sources: Mapping[str, SignalScorer]
    ) -> list[str]:
        """Filter configured source names and fallback to all available sources."""
        available = list(available_sources.keys())
        if isinstance(configured, list):
            selected = [
                source_name
                for source_name in configured
                if isinstance(source_name, str) and source_name in available_sources
            ]
            if selected:
                return selected
        return available

    def _resolve_ensemble_weights(self, source_names: list[str]) -> SourceWeights:
        """Resolve and normalize base ensemble weights for enabled sources."""
        configured_weights = self.config.get("ensemble_weights")
        weights: SourceWeights = {}

        if isinstance(configured_weights, Mapping):
            for source_name in source_names:
                weights[source_name] = max(
                    0.0, _to_float(configured_weights.get(source_name, 0.0), 0.0)
                )

        if sum(weights.values()) <= 0.0:
            if not source_names:
                return {}
            even_weight = 1.0 / len(source_names)
            return {source_name: even_weight for source_name in source_names}

        return normalize_weights(weights)

    def _collect_scores_and_confidences(
        self, market_data: MarketData
    ) -> tuple[SourceScores, SourceConfidences]:
        """Compute source scores/confidences once for reuse by callers."""
        scores: SourceScores = {}
        confidences: SourceConfidences = {}
        for source_name, scorer in self.signal_sources.items():
            score = scorer.calculate_score(market_data)
            confidence = scorer.get_confidence(market_data)
            scores[source_name] = float(clamp_value(_to_float(score, 50.0), 0.0, 100.0))
            confidences[source_name] = float(
                clamp_value(_to_float(confidence, 0.3), 0.0, 1.0)
            )
        return scores, confidences

    def _calculate_weighted_score(
        self, scores: SourceScores, weights: SourceWeights
    ) -> float:
        """Calculate weighted ensemble score with safe fallback behavior."""
        if not scores:
            return 50.0

        ensemble_score = 0.0
        total_weight = 0.0
        for source_name, score in scores.items():
            weight = float(weights.get(source_name, 0.0))
            ensemble_score += score * weight
            total_weight += weight

        if total_weight > 0.0:
            ensemble_score /= total_weight
        else:
            ensemble_score = float(np.mean(list(scores.values())))

        return float(clamp_value(ensemble_score, 0.0, 100.0))

    @staticmethod
    def _calculate_final_confidence(
        confidences: SourceConfidences, scores: SourceScores
    ) -> float:
        """Combine confidence with score divergence penalty."""
        if not confidences:
            return 0.3

        avg_confidence = float(np.mean(list(confidences.values())))
        score_std = float(np.std(list(scores.values()))) if scores else 0.0
        divergence_penalty = min(0.3, score_std / 25.0)  # Max 30% penalty
        final_confidence = avg_confidence * (1.0 - divergence_penalty)
        return float(clamp_value(final_confidence, 0.0, 1.0))

    @staticmethod
    def _calculate_agreement_level(scores: SourceScores) -> float:
        """Convert score dispersion into [0, 1] agreement level."""
        if len(scores) <= 1:
            return 0.5
        agreement = 1.0 - (float(np.std(list(scores.values()))) / 50.0)
        return float(clamp_value(agreement, 0.0, 1.0))

    def _resolve_weights(self, confidences: SourceConfidences) -> SourceWeights:
        """Resolve active weights based on static or dynamic mode."""
        if self.enable_dynamic_weights:
            return self._adjust_weights_dynamically(confidences)
        return self.ensemble_weights.copy()

    def generate_ensemble_signal(
        self, market_data: MarketData
    ) -> tuple[float, float]:
        """
        Generate ensemble signal with confidence

        Args:
            market_data: Market data dictionary

        Returns:
            tuple of (ensemble_score, confidence)
        """
        try:
            scores, confidences = self._collect_scores_and_confidences(market_data)
            adjusted_weights = self._resolve_weights(confidences)
            ensemble_score = self._calculate_weighted_score(scores, adjusted_weights)
            final_confidence = self._calculate_final_confidence(confidences, scores)

            logger.debug(
                f"Ensemble scores: {scores}, Weights: {adjusted_weights}, "
                f"Final: {ensemble_score:.1f}, Confidence: {final_confidence:.2f}"
            )

            return ensemble_score, final_confidence

        except Exception as e:
            logger.error(f"Error in ensemble signal generation: {e}")
            return 50.0, 0.3

    def _adjust_weights_dynamically(
        self, confidences: SourceConfidences
    ) -> SourceWeights:
        """
        Adjust weights dynamically based on source confidences

        Higher confidence sources get more weight
        """
        if not self.ensemble_weights:
            return {}

        adjusted_weights: SourceWeights = {}
        total_confidence = sum(confidences.values())

        if total_confidence > 0.0:
            for source_name, base_weight in self.ensemble_weights.items():
                confidence = float(confidences.get(source_name, 0.0))
                confidence_multiplier = 1.0 + (confidence - 0.5) * 0.5
                adjusted_weights[source_name] = max(
                    0.0, float(base_weight) * confidence_multiplier
                )
            return normalize_weights(adjusted_weights)

        return self.ensemble_weights.copy()

    def get_signal_reliability(self, market_data: MarketData) -> SignalReliability:
        """
        Get detailed signal reliability information

        Returns:
            Dictionary with reliability metrics
        """
        try:
            scores, confidences = self._collect_scores_and_confidences(market_data)
            adjusted_weights = self._resolve_weights(confidences)
            ensemble_score = self._calculate_weighted_score(scores, adjusted_weights)
            ensemble_confidence = self._calculate_final_confidence(confidences, scores)
            agreement = self._calculate_agreement_level(scores)

            reliability = float(clamp_value(ensemble_confidence * agreement, 0.0, 1.0))
            source_reliabilities = {source: float(value) for source, value in confidences.items()}
            return {
                "ensemble_score": ensemble_score,
                "ensemble_confidence": ensemble_confidence,
                "source_scores": scores,
                "source_confidences": confidences,
                "source_reliabilities": source_reliabilities,
                "agreement_level": agreement,
                "overall_reliability": reliability,
                "signal_strength": self._calculate_signal_strength(
                    ensemble_score, ensemble_confidence
                ),
                "recommendation": self._get_trading_recommendation(
                    ensemble_score, ensemble_confidence
                ),
            }

        except Exception as e:
            logger.error(f"Error getting signal reliability: {e}")
            return {"ensemble_score": 50.0, "ensemble_confidence": 0.3, "error": str(e)}

    def _calculate_signal_strength(self, score: float, confidence: float) -> str:
        """Calculate signal strength category"""
        strength_score = abs(score - 50.0) * confidence

        if strength_score > 20:
            return "strong"
        elif strength_score > 10:
            return "moderate"
        else:
            return "weak"

    def _get_trading_recommendation(self, score: float, confidence: float) -> str:
        """Get trading recommendation"""
        if confidence < 0.4:
            return "hold_low_confidence"

        if score > 65:
            return "buy"
        elif score < 35:
            return "sell"
        else:
            return "hold"
