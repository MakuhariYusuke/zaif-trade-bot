"""
Signal Quality Scorer

Enhanced signal quality scorer with regime adaptation capabilities.
Uses modular indicator system and configurable scoring logic.
"""

from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd

from ztb.trading.environment.components.threshold_manager import ThresholdManager
from ztb.trading.signal.common.base_classes import (
    BaseSignalProcessor,
    SignalContext,
    SignalResult,
)
from ztb.trading.signal.common.utilities import (
    normalize_weights,
    update_weights_with_dynamic_adjustment,
    validate_market_data,
)
from ztb.trading.signal.quality.indicators.macd import MACDIndicator
from ztb.trading.signal.quality.indicators.rsi import RSIIndicator
from ztb.utils.logging_utils import get_logger

logger = get_logger(__name__)


class SignalQualityScorer(BaseSignalProcessor):
    """
    Enhanced signal quality scorer with regime adaptation

    Provides deterministic scoring (0-100) based on technical indicators
    with configurable weights and thresholds.
    """

    def __init__(
        self,
        config: Optional[Dict[str, Any]] = None,
        threshold_manager: Optional[ThresholdManager] = None,
    ):
        super().__init__(config)

        # Initialize indicators
        self.indicators = self._initialize_indicators()

        # Initialize weights and thresholds from config
        # Normalize weights on init
        self.weights = normalize_weights(self.config.get("weights", {}))
        self.thresholds = self.config.get(
            "thresholds",
            {"strong_buy": 80, "buy": 65, "hold": 50, "sell": 35, "strong_sell": 20},
        )

        # Thresholds for signal generation
        self.buy_threshold = self.config.get("buy_threshold", 75)
        self.sell_threshold = self.config.get("sell_threshold", 25)
        self.hold_threshold = self.config.get("hold_threshold", 45)
        # Optional ThresholdManager for dynamic thresholds
        self.threshold_manager = threshold_manager

    def _get_default_config(self) -> Dict[str, Any]:
        """Get default configuration"""
        return {
            "weights": {
                "rsi": 0.20,  # Reduced RSI weight to prevent excessive SELL signals
                "macd": 0.20,  # Increased MACD weight for better trend confirmation
                "bollinger": 0.15,  # Reduced Bollinger weight
                "atr": 0.15,  # Increased ATR weight for volatility awareness
                "trend": 0.15,  # Increased Trend weight for better trend following
                "momentum": 0.10,  # New momentum indicator
                "stochastic": 0.05,  # New stochastic indicator
            },
            "thresholds": {
                "strong_buy": 80,
                "buy": 65,
                "hold": 50,
                "sell": 35,
                "strong_sell": 20,
            },
            "buy_threshold": 75,  # Lower threshold for BUY signals
            "sell_threshold": 25,  # Higher threshold for SELL signals
            "hold_threshold": 45,
            "indicators": {
                "rsi": {"periods": 14},
                "macd": {"fast_period": 12, "slow_period": 26, "signal_period": 9},
            },
        }

    def _initialize_indicators(self) -> Dict[str, Any]:
        """Initialize technical indicators"""
        indicator_configs = self.config.get("indicators", {})

        return {
            "rsi": RSIIndicator(indicator_configs.get("rsi", {"periods": 14})),
            "macd": MACDIndicator(
                indicator_configs.get(
                    "macd", {"fast_period": 12, "slow_period": 26, "signal_period": 9}
                )
            ),
        }

    def calculate_score(
        self, indicators: Dict[str, float], context: Optional[Dict[str, Any]] = None
    ) -> float:
        """
        Calculate signal quality score from indicators

        Args:
            indicators: Dictionary of indicator values
            context: Optional context information

        Returns:
            Quality score (0-100)
        """
        try:
            # Calculate component scores
            component_scores = self._calculate_component_scores(indicators, context)

            # Combine scores with weights
            total_score = self._combine_scores(component_scores)

            # Apply context adjustments
            if context:
                total_score = self._apply_context_adjustments(total_score, context)

            return max(0.0, min(100.0, total_score))

        except Exception as e:
            logger.error(f"Error calculating signal score: {e}")
            return 50.0  # Neutral fallback

    def _calculate_component_scores(
        self, indicators: Dict[str, float], context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, float]:
        """Calculate individual component scores"""
        scores = {}

        # RSI score (0-100, higher when oversold for BUY, oversold for SELL)
        rsi_value = indicators.get("rsi", 50.0)
        if rsi_value < 30:
            scores["rsi"] = 80.0  # Oversold - potential BUY
        elif rsi_value > 70:
            scores["rsi"] = 20.0  # Overbought - potential SELL
        else:
            scores["rsi"] = 50.0  # Neutral

        # MACD score
        macd_line = indicators.get("macd_line", 0.0)
        macd_signal = indicators.get("macd_signal", 0.0)
        macd_histogram = indicators.get("macd_histogram", 0.0)

        # MACD score based on trend
        if macd_line > macd_signal and macd_histogram > 0:
            scores["macd"] = 75.0  # Bullish
        elif macd_line < macd_signal and macd_histogram < 0:
            scores["macd"] = 25.0  # Bearish
        else:
            scores["macd"] = 50.0  # Neutral

        # Bollinger Bands score
        bb_position = indicators.get("bollinger_position", 0.5)
        if bb_position < 0.2:
            scores["bollinger"] = 80.0  # Near lower band - potential BUY
        elif bb_position > 0.8:
            scores["bollinger"] = 20.0  # Near upper band - potential SELL
        else:
            scores["bollinger"] = 50.0  # Middle range

        # ATR score (volatility)
        atr = indicators.get("atr", 1.0)
        if atr < 0.5:
            scores["atr"] = 30.0  # Low volatility
        elif atr > 2.0:
            scores["atr"] = 70.0  # High volatility
        else:
            scores["atr"] = 50.0  # Normal volatility

        # Trend score (simplified)
        trend_strength = indicators.get("trend_strength", 0.0)
        if trend_strength > 1.0:
            scores["trend"] = 75.0  # Strong trend
        elif trend_strength < -1.0:
            scores["trend"] = 25.0  # Strong downtrend
        else:
            scores["trend"] = 50.0  # Weak trend

        # Momentum score
        momentum = indicators.get("momentum", 0.0)
        scores["momentum"] = min(100.0, max(0.0, 50.0 + momentum * 25.0))

        # Stochastic score (placeholder)
        scores["stochastic"] = 50.0

        return scores

        # Bollinger Bands score
        bb_position = indicators.get("bollinger_position", 0.5)
        if bb_position < 0.2:
            scores["bollinger"] = 80.0  # Near lower band - potential BUY
        elif bb_position > 0.8:
            scores["bollinger"] = 20.0  # Near upper band - potential SELL
        else:
            scores["bollinger"] = 50.0  # Middle range

        # ATR score (volatility)

        # ATR score (volatility)
        atr = indicators.get("atr", 1.0)
        if atr < 0.5:
            scores["atr"] = 30.0  # Low volatility
        elif atr > 2.0:
            scores["atr"] = 70.0  # High volatility
        else:
            scores["atr"] = 50.0  # Normal volatility

        # Trend score (simplified)
        trend_strength = indicators.get("trend_strength", 0.0)
        if trend_strength > 1.0:
            scores["trend"] = 75.0  # Strong trend
        elif trend_strength < -1.0:
            scores["trend"] = 25.0  # Strong downtrend
        else:
            scores["trend"] = 50.0  # Weak trend

        # Momentum score
        momentum = indicators.get("momentum", 0.0)
        scores["momentum"] = min(100.0, max(0.0, 50.0 + momentum * 25.0))

        # Stochastic score (placeholder)
        scores["stochastic"] = 50.0

        return scores

    def _combine_scores(self, component_scores: Dict[str, float]) -> float:
        """Combine component scores with weights"""
        total_score = 0.0
        total_weight = 0.0

        for component, score in component_scores.items():
            weight = self.weights.get(component, 0.0)
            total_score += score * weight
            total_weight += weight

        return total_score / total_weight if total_weight > 0 else 50.0

    def _apply_context_adjustments(
        self, score: float, context: Dict[str, Any]
    ) -> float:
        """Apply position and market context adjustments"""
        # Position adjustments
        position_ratio = context.get("position_ratio", 0.5)
        if position_ratio > 0.8:
            # Overexposed - reduce buy signals, increase sell signals
            if score > 50:
                score = score * 0.8
            else:
                score = score * 1.2
        elif position_ratio < 0.2:
            # Underexposed - increase buy signals, reduce sell signals
            if score > 50:
                score = score * 1.2
            else:
                score = score * 0.8

        # Market regime adjustments (placeholder for Phase 2)
        regime = context.get("market_regime")
        if regime:
            score = self._apply_regime_adjustments(score, regime)

        return max(0, min(100, score))

    def _apply_regime_adjustments(self, score: float, regime: str) -> float:
        """Apply market regime specific adjustments (Phase 2)"""
        # Placeholder for regime-specific adjustments
        # Will be implemented in Phase 2 with MarketRegimeClassifier integration
        return score

    def process_signal(self, context: SignalContext) -> SignalResult:
        """
        Process signal using the new interface

        Args:
            context: Signal processing context

        Returns:
            Signal processing result
        """
        if not self.validate_input(context):
            return SignalResult(
                discrete_action=0,
                quality_score=50.0,
                confidence=0.5,
                metadata={"error": "Invalid input context"},
            )

        try:
            # Calculate indicators
            market_data = context.market_data
            indicators = self._calculate_indicators(market_data)

            # Calculate score
            quality_score = self.calculate_score(
                indicators,
                {
                    "position_ratio": self._calculate_position_ratio(context),
                    "market_regime": getattr(context, "market_regime", None),
                },
            )

            # Determine dynamic thresholds and weight adjustments when available
            if self.threshold_manager is not None:
                try:
                    # Use market data from context for adaptive thresholds
                    df = context.market_data
                    adaptive_thresholds = (
                        self.threshold_manager.calculate_adaptive_signal_thresholds(df)
                    )
                    confidence_threshold = adaptive_thresholds.get(
                        "confidence_threshold", self.buy_threshold / 100.0
                    )
                    from ztb.trading.signal.common.utilities import (
                        confidence_to_score_thresholds,
                    )

                    buy_t, sell_t = confidence_to_score_thresholds(
                        confidence_threshold,
                        default_buy=self.buy_threshold,
                        default_sell=self.sell_threshold,
                        min_gap=self.config.get("hold_gap", 10.0),
                    )
                    self.thresholds["buy"] = max(0, min(100, buy_t))
                    self.thresholds["sell"] = max(0, min(100, sell_t))
                    # Update weights based on regime adjustments if present
                    regime = self.threshold_manager.detect_market_regime(df)
                    adj = self.threshold_manager.get_regime_adjustments(regime)
                    # Apply same strength_multiplier across weights as a quick adaptation
                    weight_adjustments = {
                        k: adj.get("strength_multiplier", 1.0)
                        for k in self.weights.keys()
                    }
                    self.weights = update_weights_with_dynamic_adjustment(
                        self.weights, weight_adjustments
                    )
                except Exception as e:
                    logger.warning(
                        f"ThresholdManager failed during process_signal: {e}"
                    )

            # Determine action and confidence
            discrete_action, confidence = self.apply_thresholds(quality_score)

            return SignalResult(
                discrete_action=discrete_action,
                quality_score=quality_score,
                confidence=confidence,
                metadata={
                    "indicators": indicators,
                    "thresholds": {
                        "buy": self.buy_threshold,
                        "sell": self.sell_threshold,
                        "hold": self.hold_threshold,
                    },
                },
            )

        except Exception as e:
            logger.error(f"Error processing signal: {e}")
            return SignalResult(
                discrete_action=0,
                quality_score=50.0,
                confidence=0.5,
                metadata={"error": str(e)},
            )

    def _calculate_indicators(self, data: pd.DataFrame) -> Dict[str, float]:
        """Calculate all technical indicators"""
        indicators = {}

        # Use modular indicators
        for name, indicator in self.indicators.items():
            result = indicator.calculate(data)
            indicators.update(result)

        # Calculate additional indicators not covered by modular system
        indicators.update(self._calculate_additional_indicators(data))

        return indicators

    def _calculate_additional_indicators(self, data: pd.DataFrame) -> Dict[str, float]:
        """Calculate additional indicators"""
        # Bollinger Bands position
        if len(data) >= 20:
            sma = data["close"].rolling(20).mean()
            std = data["close"].rolling(20).std()
            upper_band = sma + (std * 2)
            lower_band = sma - (std * 2)
            bb_position = (data["close"].iloc[-1] - lower_band.iloc[-1]) / (
                upper_band.iloc[-1] - lower_band.iloc[-1]
            )
            bb_position = max(0, min(1, bb_position))
        else:
            bb_position = 0.5

        # Trend strength (simplified)
        if len(data) >= 10:
            recent_prices = data["close"].tail(10).values
            x = np.arange(len(recent_prices))
            slope = np.polyfit(x, recent_prices, 1)[0]
            trend_strength = slope * 100  # Scale for interpretation
        else:
            trend_strength = 0.0

        # Momentum (ROC)
        if len(data) >= 10:
            momentum = (data["close"].iloc[-1] - data["close"].iloc[-10]) / data[
                "close"
            ].iloc[-10]
        else:
            momentum = 0.0

        # Volatility
        if len(data) >= 20:
            returns = data["close"].pct_change().fillna(0)
            volatility = returns.rolling(20).std().iloc[-1]
        else:
            volatility = 0.05

        return {
            "bollinger_position": bb_position,
            "trend_strength": trend_strength,
            "momentum": momentum,
            "volatility": volatility,
        }

    def _calculate_position_ratio(self, context: SignalContext) -> float:
        """Calculate current position ratio"""
        portfolio = context.portfolio_state
        btc_balance = portfolio.get("btc_balance", 0.0)
        jpy_balance = portfolio.get("jpy_balance", 0.0)
        current_price = portfolio.get("current_price", 0.0)

        if current_price <= 0:
            return 0.5

        btc_value = btc_balance * current_price
        total_value = btc_value + jpy_balance

        return btc_value / total_value if total_value > 0 else 0.5

    # Legacy interface for backward compatibility
    def calculate_signal_quality(
        self, df: pd.DataFrame, continuous_action: float, portfolio: Dict[str, Any]
    ) -> Tuple[int, float]:
        """
        Legacy interface for backward compatibility
        """
        context = SignalContext(
            market_data=df,
            position_context={},
            portfolio_state=portfolio,
            timestamp=df.index[-1] if not df.empty else pd.Timestamp.now(),
        )

        result = self.process_signal(context)

        # Blend with continuous action (legacy behavior)
        action_score = (continuous_action + 1) * 50
        final_score = result.quality_score * 0.8 + action_score * 0.2

        # Recalculate action with blended score
        discrete_action, _ = self.apply_thresholds(final_score)

        return discrete_action, final_score

    def apply_thresholds(self, score: float) -> Tuple[int, float]:
        """
        Apply thresholds to convert score to discrete action

        Returns:
            Tuple of (discrete_action, confidence)
        """
        strong_buy_threshold = self.thresholds.get("strong_buy", 80)
        buy_threshold = self.thresholds.get("buy", 65)
        sell_threshold = self.thresholds.get("sell", 35)
        strong_sell_threshold = self.thresholds.get("strong_sell", 20)

        if score >= strong_buy_threshold:
            return 2, min(
                1.0, (score - strong_buy_threshold) / (100 - strong_buy_threshold)
            )
        elif score >= buy_threshold:
            return 1, min(
                1.0, (score - buy_threshold) / (strong_buy_threshold - buy_threshold)
            )
        elif score <= strong_sell_threshold:
            return -2, min(1.0, (strong_sell_threshold - score) / strong_sell_threshold)
        elif score <= sell_threshold:
            return -1, min(
                1.0, (sell_threshold - score) / (sell_threshold - strong_sell_threshold)
            )
        else:
            # HOLD zone
            distance_to_buy = abs(score - buy_threshold)
            distance_to_sell = abs(score - sell_threshold)
            confidence = 1.0 - min(distance_to_buy, distance_to_sell) / (
                buy_threshold - sell_threshold
            )
            return 0, max(0.0, confidence)

    def process_signal(self, context: SignalContext) -> SignalResult:
        """
        Process signal quality scoring

        Args:
            context: Signal processing context

        Returns:
            SignalResult with quality score and confidence
        """
        try:
            # Validate market data
            if not validate_market_data(context.market_data):
                return SignalResult(
                    discrete_action=0,
                    quality_score=50.0,
                    confidence=0.5,
                    metadata={"error": "Invalid market data"},
                )

            # Calculate indicators
            indicator_values = {}
            for name, indicator in self.indicators.items():
                try:
                    result = indicator.calculate(context.market_data)
                    indicator_values.update(result)
                except Exception as e:
                    logger.warning(f"Failed to calculate {name} indicator: {e}")
                    continue

            # Calculate quality score
            quality_score = self.calculate_score(
                indicator_values, context.position_context
            )

            # Determine action and confidence
            discrete_action, confidence = self.apply_thresholds(quality_score)

            return SignalResult(
                discrete_action=discrete_action,
                quality_score=quality_score,
                confidence=confidence,
                metadata={
                    "indicator_values": indicator_values,
                    "weights_used": self.weights,
                    "thresholds_used": self.thresholds,
                },
            )
        except Exception as e:
            logger.error(f"Error in signal quality scoring: {e}")
            return SignalResult(
                discrete_action=0,
                quality_score=50.0,
                confidence=0.5,
                metadata={"error": str(e)},
            )
