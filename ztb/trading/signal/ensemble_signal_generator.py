"""
Ensemble Signal Methods for Phase 3

多ソースシグナル統合による高度なアンサンブル手法
信頼度計算と動的ウェイト最適化を実装
"""

from typing import Any, Dict, Optional, Tuple

import numpy as np

# Remove circular import - will import locally when needed
# from ztb.trading.signal.quality_scorer import SignalQualityScorer
from ztb.utils.logging_utils import get_logger

logger = get_logger(__name__)


class BaseSignalScorer:
    """Base class for signal scorers"""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        # Keep class name lowercased (avoid truncating 'Scorer') so tests
        # expecting 'basesignalscorer' still pass
        self.name = self.__class__.__name__.lower()

    def calculate_score(self, market_data: Dict[str, Any]) -> float:
        """Calculate signal score (0-100)"""
        raise NotImplementedError

    def get_confidence(self, market_data: Dict[str, Any]) -> float:
        """Get confidence level (0-1)"""
        return 0.5


class TechnicalSignalScorer(BaseSignalScorer):
    """Technical analysis based signal scorer"""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        # Use TechnicalIndicators directly instead of SignalQualityScorer to avoid circular import
        from ztb.trading.signal.technical_indicators import TechnicalIndicators

        self.technical_indicators = TechnicalIndicators()

    def calculate_score(self, market_data: Dict[str, Any]) -> float:
        """Calculate technical signal score using TechnicalIndicators"""
        try:
            df = market_data.get("df")
            continuous_action = market_data.get("continuous_action", 0.0)
            portfolio = market_data.get("portfolio", {})

            if df is None or len(df) == 0:
                return 50.0

            # Get technical signals
            tech_signals = self.technical_indicators.get_technical_signals(df)

            # Simple scoring based on key indicators
            score = 50.0  # Neutral starting point

            # RSI scoring (0-100 scale)
            rsi = tech_signals.get("rsi", 50.0)
            if rsi < 30:
                score += 20  # Oversold - bullish
            elif rsi > 70:
                score -= 20  # Overbought - bearish

            # MACD scoring
            macd_line = tech_signals.get("macd", 0.0)
            signal_line = tech_signals.get("macd_signal", 0.0)
            if macd_line > signal_line:
                score += 15  # Bullish crossover
            elif macd_line < signal_line:
                score -= 15  # Bearish crossover

            # Bollinger Band position
            bb_position = tech_signals.get("bb_position", 0.5)
            if bb_position < 0.2:
                score += 10  # Near lower band - bullish
            elif bb_position > 0.8:
                score -= 10  # Near upper band - bearish

            # Ensure score is within bounds
            score = max(0, min(100, score))

            return score

        except Exception as e:
            logger.warning(f"Error in technical signal scoring: {e}")
            return 50.0

    def get_confidence(self, market_data: Dict[str, Any]) -> float:
        """Get technical signal confidence"""
        try:
            df = market_data.get("df")
            if df is None or len(df) < 20:
                return 0.3

            # Confidence based on data quality and market conditions
            volatility = df["close"].pct_change().std()
            volume_avg = df["volume"].mean() if "volume" in df.columns else 1000

            # Higher confidence with sufficient data and reasonable volatility
            confidence = min(0.9, max(0.1, 1.0 - volatility * 10))
            return confidence

        except Exception:
            return 0.3


class PatternRecognitionScorer(BaseSignalScorer):
    """Pattern recognition based signal scorer"""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        self.min_periods = self.config.get("min_periods", 20)

    def calculate_score(self, market_data: Dict[str, Any]) -> float:
        """Calculate pattern-based signal score"""
        try:
            df = market_data.get("df")
            if df is None or len(df) < self.min_periods:
                return 50.0

            # Simple pattern recognition: trend continuation vs reversal
            recent_prices = df["close"].tail(10).values
            if len(recent_prices) < 5:
                return 50.0

            # Calculate trend strength
            trend_slope = np.polyfit(range(len(recent_prices)), recent_prices, 1)[0]

            # Look for reversal patterns
            recent_high = max(recent_prices[-5:])
            recent_low = min(recent_prices[-5:])
            current_price = recent_prices[-1]

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
                # Clamp to 0-100
                return float(max(0.0, min(100.0, score)))

        except Exception as e:
            logger.warning(f"Error in pattern recognition scoring: {e}")
            return 50.0

    def get_confidence(self, market_data: Dict[str, Any]) -> float:
        """Get pattern recognition confidence"""
        try:
            df = market_data.get("df")
            if df is None or len(df) < self.min_periods:
                return 0.2

            # Confidence based on pattern clarity
            recent_volatility = df["close"].tail(10).pct_change().std()
            return min(0.8, max(0.1, 1.0 - recent_volatility * 5))

        except Exception:
            return 0.2


class SentimentSignalScorer(BaseSignalScorer):
    """Sentiment-based signal scorer"""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        self.sentiment_window = self.config.get("sentiment_window", 24)  # hours

    def calculate_score(self, market_data: Dict[str, Any]) -> float:
        """Calculate sentiment-based signal score"""
        try:
            # For now, use price-based sentiment proxy
            # In production, this would integrate with social media, news, etc.
            df = market_data.get("df")
            if df is None or len(df) < 10:
                return 50.0

            # Price momentum as sentiment proxy
            short_momentum = df["close"].pct_change(5).iloc[-1]
            long_momentum = df["close"].pct_change(20).iloc[-1]

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

    def get_confidence(self, market_data: Dict[str, Any]) -> float:
        """Get sentiment confidence"""
        # Sentiment data is typically less reliable
        return 0.4


class VolumeProfileScorer(BaseSignalScorer):
    """Volume profile based signal scorer"""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        self.volume_window = self.config.get("volume_window", 20)

    def calculate_score(self, market_data: Dict[str, Any]) -> float:
        """Calculate volume-based signal score"""
        try:
            df = market_data.get("df")
            if df is None or "volume" not in df.columns or len(df) < self.volume_window:
                return 50.0

            # Volume analysis
            recent_volume = df["volume"].tail(self.volume_window)
            avg_volume = recent_volume.mean()
            current_volume = recent_volume.iloc[-1]

            # Price-volume relationship
            price_change = df["close"].pct_change().iloc[-1]
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

    def get_confidence(self, market_data: Dict[str, Any]) -> float:
        """Get volume profile confidence"""
        try:
            df = market_data.get("df")
            if df is None or "volume" not in df.columns:
                return 0.1

            # Confidence based on volume data quality
            volume_variation = (
                df["volume"].tail(20).std() / df["volume"].tail(20).mean()
            )
            return min(0.9, max(0.2, volume_variation * 2))

        except Exception:
            return 0.2


class EnsembleSignalGenerator:
    """アンサンブルシグナル生成器"""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or self._get_default_config()

        # Initialize signal sources
        self.signal_sources = {
            "technical": TechnicalSignalScorer(self.config.get("technical", {})),
            "pattern": PatternRecognitionScorer(self.config.get("pattern", {})),
            "sentiment": SentimentSignalScorer(self.config.get("sentiment", {})),
            "volume": VolumeProfileScorer(self.config.get("volume", {})),
        }

        # Ensemble weights
        self.ensemble_weights = self.config.get(
            "ensemble_weights",
            {"technical": 0.4, "pattern": 0.3, "sentiment": 0.2, "volume": 0.1},
        )

        # Dynamic weight adjustment
        self.enable_dynamic_weights = self.config.get("enable_dynamic_weights", True)

    def _get_default_config(self) -> Dict[str, Any]:
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

    def generate_ensemble_signal(
        self, market_data: Dict[str, Any]
    ) -> Tuple[float, float]:
        """
        Generate ensemble signal with confidence

        Args:
            market_data: Market data dictionary

        Returns:
            Tuple of (ensemble_score, confidence)
        """
        try:
            # Get individual scores and confidences
            scores = {}
            confidences = {}

            for source_name, scorer in self.signal_sources.items():
                scores[source_name] = scorer.calculate_score(market_data)
                confidences[source_name] = scorer.get_confidence(market_data)

            # Dynamic weight adjustment based on confidence
            if self.enable_dynamic_weights:
                adjusted_weights = self._adjust_weights_dynamically(confidences)
            else:
                adjusted_weights = self.ensemble_weights.copy()

            # Calculate weighted ensemble score
            ensemble_score = 0.0
            total_weight = 0.0

            for source_name, score in scores.items():
                weight = adjusted_weights.get(source_name, 0.0)
                ensemble_score += float(score) * weight
                total_weight += weight

            if total_weight > 0:
                ensemble_score /= total_weight

            # Ensure ensemble score is within 0-100
            ensemble_score = float(max(0.0, min(100.0, ensemble_score)))

            # Calculate overall confidence
            avg_confidence = np.mean(list(confidences.values()))
            score_std = np.std(list(scores.values()))

            # Reduce confidence if scores are highly divergent
            divergence_penalty = min(0.3, score_std / 25)  # Max 30% penalty
            final_confidence = avg_confidence * (1.0 - divergence_penalty)

            logger.debug(
                f"Ensemble scores: {scores}, Weights: {adjusted_weights}, "
                f"Final: {ensemble_score:.1f}, Confidence: {final_confidence:.2f}"
            )

            return ensemble_score, final_confidence

        except Exception as e:
            logger.error(f"Error in ensemble signal generation: {e}")
            return 50.0, 0.3

    def _adjust_weights_dynamically(
        self, confidences: Dict[str, float]
    ) -> Dict[str, float]:
        """
        Adjust weights dynamically based on source confidences

        Higher confidence sources get more weight
        """
        adjusted_weights = {}
        total_confidence = sum(confidences.values())

        if total_confidence > 0:
            for source_name, confidence in confidences.items():
                # Boost weight for high confidence sources
                base_weight = self.ensemble_weights.get(source_name, 0.0)
                confidence_multiplier = (
                    1.0 + (confidence - 0.5) * 0.5
                )  # ±25% adjustment
                adjusted_weights[source_name] = base_weight * confidence_multiplier
        else:
            # Fallback to base weights
            adjusted_weights = self.ensemble_weights.copy()

        # Normalize weights
        total_adjusted = sum(adjusted_weights.values())
        if total_adjusted > 0:
            for source_name in adjusted_weights:
                adjusted_weights[source_name] /= total_adjusted

        return adjusted_weights

    def get_signal_reliability(self, market_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Get detailed signal reliability information

        Returns:
            Dictionary with reliability metrics
        """
        try:
            scores = {}
            confidences = {}

            for source_name, scorer in self.signal_sources.items():
                scores[source_name] = scorer.calculate_score(market_data)
                confidences[source_name] = scorer.get_confidence(market_data)

            ensemble_score, ensemble_confidence = self.generate_ensemble_signal(
                market_data
            )

            # Calculate agreement level
            score_values = list(scores.values())
            if len(score_values) > 1:
                agreement = 1.0 - (
                    np.std(score_values) / 50.0
                )  # Lower std = higher agreement
                agreement = max(0.0, min(1.0, agreement))
            else:
                agreement = 0.5

            reliability = ensemble_confidence * agreement
            source_reliabilities = {k: float(v) for k, v in confidences.items()}
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
