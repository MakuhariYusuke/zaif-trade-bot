"""
Signal Quality Scorer

テクニカル指標と市場条件に基づく決定論的信号品質スコアリング
確率的アプローチから決定論的アプローチへの移行

Features:
- Deterministic scoring (0-100) based on technical indicators
- Market condition awareness
- Position context consideration
- Signal frequency optimization for scalping (20-50 signals/day)
"""

from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd

from ztb.trading.environment.components.threshold_manager import ThresholdManager
from ztb.trading.signal.common.metrics import calculate_trend_metrics
from ztb.trading.signal.common.utilities import (
    normalize_weights,
    score_to_discrete_action,
)
from ztb.trading.signal.constants import (
    CONTINUOUS_TO_SCORE_SCALE,
    DEFAULT_BUY_THRESHOLD,
    DEFAULT_HOLD_THRESHOLD,
    DEFAULT_SELL_THRESHOLD,
    HIGH_SCORE_IS_BUY,
)
from ztb.trading.signal.ensemble_signal_generator import EnsembleSignalGenerator
from ztb.trading.signal.quality.indicators.base import (
    AdaptiveIndicator,
    CompositeIndicator,
)
from ztb.trading.signal.quality.indicators.macd import MACDIndicator
from ztb.trading.signal.quality.indicators.rsi import RSIIndicator
from ztb.trading.signal.technical_indicators import TechnicalIndicators
from ztb.utils.logging_utils import get_logger

logger = get_logger(__name__)


class SignalQualityScorer:
    """
    Deterministic signal quality scorer using technical indicators

    Replaces probabilistic signal guidance with score-based approach
    for higher frequency and accuracy
    """

    def __init__(
        self,
        config: Optional[Dict[str, Any]] = None,
        threshold_manager: Optional[ThresholdManager] = None,
    ):
        """
        Initialize signal quality scorer

        Args:
            config: Configuration dictionary
        """
        self.config = config or self._get_default_config()
        self.logger = get_logger(__name__)
        self.technical_indicators = TechnicalIndicators()

        # Initialize advanced indicators for Phase 2 integration
        self._initialize_advanced_indicators()

        # Initialize ensemble signal generator for Phase 3 integration
        self.ensemble_generator = EnsembleSignalGenerator(
            self.config.get("ensemble", {})
        )

        # Scoring weights - Phase 1 improved balance for higher win rate
        raw_weights = self.config.get(
            "weights",
            {
                "rsi": 0.22,  # Balanced RSI weight for momentum signals
                "macd": 0.22,  # Balanced MACD weight for trend confirmation
                "bollinger": 0.18,  # Balanced Bollinger weight for range signals
                "atr": 0.13,  # Balanced ATR weight for volatility context
                "trend": 0.13,  # Balanced Trend weight for directional bias
                "momentum": 0.07,  # Reduced momentum weight for price acceleration
                "stochastic": 0.05,  # Reduced stochastic weight for overbought/oversold levels
            },
        )
        # Normalize weights using common utility
        self.weights = normalize_weights(raw_weights)

        # Thresholds for signal generation (adjusted for 20-50 signals/day target)
        self.buy_threshold = self.config.get("buy_threshold", DEFAULT_BUY_THRESHOLD)
        self.sell_threshold = self.config.get("sell_threshold", DEFAULT_SELL_THRESHOLD)
        # Optionally use a ThresholdManager to provide dynamic thresholds at runtime
        self.threshold_manager = threshold_manager
        self.hold_threshold = self.config.get("hold_threshold", DEFAULT_HOLD_THRESHOLD)

        # Phase 3 ensemble settings
        self.enable_ensemble = self.config.get("enable_ensemble", False)
        self.ensemble_weight = self.config.get(
            "ensemble_weight", 0.3
        )  # How much ensemble influences final score

    def _initialize_advanced_indicators(self):
        """Initialize advanced indicators for Phase 2 integration"""
        # Base indicators for composite and adaptive functionality
        self.rsi_indicator = RSIIndicator(
            {"periods": self.config.get("rsi_period", 14)}
        )
        self.macd_indicator = MACDIndicator(
            {
                "fast_period": self.config.get("macd_fast", 12),
                "slow_period": self.config.get("macd_slow", 26),
                "signal_period": self.config.get("macd_signal", 9),
            }
        )

        # Composite indicator combining RSI and MACD
        self.composite_indicator = CompositeIndicator(
            indicators=[self.rsi_indicator, self.macd_indicator],
            weights={"rsi": 0.6, "macd": 0.4},
        )

        # Adaptive indicator for market regime adaptation
        self.adaptive_rsi = AdaptiveIndicator(
            base_indicator=self.rsi_indicator,
            config={
                "adaptive_params": {
                    "trending": {"periods": 21},  # Longer period for trending markets
                    "ranging": {"periods": 9},  # Shorter period for ranging markets
                    "volatile": {"periods": 14},  # Standard period for volatile markets
                }
            },
        )

        self.adaptive_macd = AdaptiveIndicator(
            base_indicator=self.macd_indicator,
            config={
                "adaptive_params": {
                    "trending": {
                        "fast_period": 8,
                        "slow_period": 21,
                        "signal_period": 5,
                    },  # Faster for trends
                    "ranging": {
                        "fast_period": 12,
                        "slow_period": 26,
                        "signal_period": 9,
                    },  # Standard for ranging
                    "volatile": {
                        "fast_period": 5,
                        "slow_period": 13,
                        "signal_period": 4,
                    },  # Very fast for volatility
                }
            },
        )

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
            "buy_threshold": 75,  # Lower threshold for BUY signals (more realistic)
            "sell_threshold": 25,  # Higher threshold for SELL signals (more realistic)
            "hold_threshold": 45,
            "rsi_period": 14,
            "macd_fast": 12,
            "macd_slow": 26,
            "macd_signal": 9,
            "bollinger_period": 20,
            "bollinger_std": 2.0,
            "atr_period": 14,
            "trend_window": 10,
        }

    def calculate_signal_quality(
        self, df: pd.DataFrame, continuous_action: float, portfolio: Dict[str, Any]
    ) -> Tuple[int, float]:
        """
        Calculate signal quality score and determine action

        Args:
            df: Market data DataFrame with OHLCV
            continuous_action: Raw continuous action from model (-1 to 1)
            portfolio: Current portfolio state

        Returns:
            Tuple of (discrete_action, quality_score)
        """
        try:
            # Quick fallback for very short data windows where technical indicators
            # cannot be computed reliably: rely primarily on continuous action.
            min_points_for_indicators = 10
            if df is None or len(df) < min_points_for_indicators:
                # Use fallback thresholding but also return a reasonable confidence score
                fallback_action = self._fallback_action(continuous_action)
                # Map fallback action to a confidence score: BUY->80, SELL->20, HOLD->50
                fallback_score = (
                    80.0
                    if fallback_action == 1
                    else (20.0 if fallback_action == -1 else 50.0)
                )
                return fallback_action, fallback_score
            # Determine market regime for adaptive indicators
            market_regime = self._determine_market_regime(df)

            # Get technical indicators (legacy method)
            tech_signals = self.technical_indicators.get_technical_signals(df)

            # Get enhanced signals using Phase 2 indicators
            enhanced_signals = self._get_enhanced_signals(df, market_regime)

            # Get RSI for potential force SELL logic (but don't force it)
            rsi = enhanced_signals.get("rsi", tech_signals.get("rsi", 50.0))

            # Calculate component scores using enhanced signals
            rsi_score = self._calculate_rsi_score(enhanced_signals)
            macd_score = self._calculate_macd_score_enhanced(enhanced_signals)
            bollinger_score = self._calculate_bollinger_score(enhanced_signals, df)
            atr_score = self._calculate_atr_score(enhanced_signals)
            trend_score = self._calculate_trend_score_enhanced(df)
            momentum_score = self._calculate_momentum_score(enhanced_signals)
            stochastic_score = self._calculate_stochastic_score(enhanced_signals)

            # Combine scores with weights
            total_score = (
                rsi_score * self.weights["rsi"]
                + macd_score * self.weights["macd"]
                + bollinger_score * self.weights["bollinger"]
                + atr_score * self.weights["atr"]
                + trend_score * self.weights["trend"]
                + momentum_score * self.weights["momentum"]
                + stochastic_score * self.weights["stochastic"]
            )

            # Apply position context adjustments
            total_score = self._apply_position_adjustments(total_score, portfolio)

            # Apply continuous action influence
            final_score = self._blend_continuous_action(total_score, continuous_action)

            # Phase 3: Apply ensemble signal integration if enabled
            if self.enable_ensemble:
                final_score = self._apply_ensemble_integration(
                    final_score, df, continuous_action, portfolio
                )

            # Determine numeric thresholds from threshold_manager if available
            if self.threshold_manager is not None:
                try:
                    adaptive_thresholds = (
                        self.threshold_manager.calculate_adaptive_signal_thresholds(df)
                    )
                    confidence_threshold = adaptive_thresholds.get(
                        "confidence_threshold", self.buy_threshold / 100.0
                    )
                    # Map using centralized helper (clamp + min_gap) into 0-100 score thresholds
                    from ztb.trading.signal.common.utilities import (
                        confidence_to_score_thresholds,
                    )

                    (
                        dynamic_buy_threshold,
                        dynamic_sell_threshold,
                    ) = confidence_to_score_thresholds(
                        confidence_threshold,
                        default_buy=self.buy_threshold,
                        default_sell=self.sell_threshold,
                        min_gap=self.config.get("hold_gap", 10.0),
                    )
                    discrete_action = score_to_discrete_action(
                        final_score,
                        buy_threshold=dynamic_buy_threshold,
                        sell_threshold=dynamic_sell_threshold,
                        high_score_is_buy=HIGH_SCORE_IS_BUY,
                    )
                except Exception as e:
                    self.logger.warning(
                        f"ThresholdManager failed to provide adaptive thresholds: {e}"
                    )
                    discrete_action = self._score_to_action(final_score)
            else:
                # Determine discrete action using configured static thresholds
                discrete_action = self._score_to_action(final_score)

            logger.debug(
                f"Signal scores - RSI: {rsi:.1f}, MACD: {macd_score:.1f}, "
                f"BB: {bollinger_score:.1f}, ATR: {atr_score:.1f}, "
                f"Trend: {trend_score:.1f}, Momentum: {momentum_score:.1f}, "
                f"Stoch: {stochastic_score:.1f}, Total: {final_score:.1f}, "
                f"Regime: {market_regime}, Action: {discrete_action}"
            )

            return discrete_action, final_score

        except Exception as e:
            logger.error(f"Error calculating signal quality: {e}")
            # Fallback to continuous action conversion
            return self._fallback_action(continuous_action), 50.0

    def _calculate_rsi_score(self, tech_signals: Dict[str, float]) -> float:
        """Calculate improved RSI-based score (0-100) with enhanced zoning for better signal balance"""
        rsi = tech_signals.get("rsi", 50.0)

        if rsi <= 25:
            # 極端なオーバーソールド - 強力BUYシグナル
            score = 90 + (25 - rsi) * 0.4  # 90-100 range
            return max(90, min(100, score))
        elif rsi <= 35:
            # 通常オーバーソールド - 中程度BUYシグナル
            score = 70 + (35 - rsi) * 1.0  # 70-80 range
            return max(70, min(80, score))
        elif rsi >= 75:
            # 極端なオーバーバウト - 強力SELLシグナル
            score = 10 - (rsi - 75) * 0.4  # 0-10 range
            return max(0, min(10, score))
        elif rsi >= 65:
            # 通常オーバーバウト - 中程度SELLシグナル
            score = 30 - (rsi - 65) * 1.0  # 20-30 range
            return max(20, min(30, score))
        else:
            # 中間ゾーン - トレンド依存のニュートラルシグナル
            score = 40 + (rsi - 50) * 0.6  # 25-55 range
            return max(25, min(55, score))

    def _calculate_macd_score(self, tech_signals: Dict[str, float]) -> float:
        """Calculate MACD-based score (0-100)"""
        macd_line = tech_signals.get("macd_line", 0.0)
        signal_line = tech_signals.get("macd_signal", 0.0)
        histogram = tech_signals.get("macd_histogram", 0.0)

        # MACD crossover signals - More balanced
        if macd_line > signal_line and histogram > 0:
            # Bullish crossover
            score = 60 + min(histogram * 50, 25)  # 60-85 range
        elif macd_line < signal_line and histogram < 0:
            # Bearish crossover
            score = 40 - min(abs(histogram) * 50, 25)  # 15-40 range
        else:
            # No clear signal
            score = 50

        return max(0, min(100, score))

    def _calculate_macd_score_enhanced(self, tech_signals: Dict[str, Any]) -> float:
        """Calculate enhanced MACD-based score using Phase 2 features (0-100)"""
        macd_line = tech_signals.get("macd_line", 0.0)
        signal_line = tech_signals.get("signal_line", 0.0)  # Enhanced: use signal_line
        histogram = tech_signals.get("histogram", 0.0)  # Enhanced: use histogram
        histogram_prev = tech_signals.get("histogram_prev", 0.0)

        # Enhanced MACD scoring with histogram momentum
        base_score = 50.0

        # Histogram momentum analysis
        if histogram > 0 and histogram_prev > 0:
            # Sustained bullish momentum
            if histogram > histogram_prev:
                base_score += 20  # Increasing bullish momentum
            else:
                base_score += 10  # Sustained bullish momentum
        elif histogram < 0 and histogram_prev < 0:
            # Sustained bearish momentum
            if histogram < histogram_prev:
                base_score -= 20  # Increasing bearish momentum
            else:
                base_score -= 10  # Sustained bearish momentum

        # Signal line crossover with histogram confirmation
        if macd_line > signal_line:
            if histogram > 0:
                base_score += 15  # Confirmed bullish crossover
            else:
                base_score += 5  # Weak bullish signal
        elif macd_line < signal_line:
            if histogram < 0:
                base_score -= 15  # Confirmed bearish crossover
            else:
                base_score -= 5  # Weak bearish signal

        # Histogram zero-crossing signals
        if histogram > 0 and histogram_prev <= 0:
            base_score += 10  # Bullish zero crossing
        elif histogram < 0 and histogram_prev >= 0:
            base_score -= 10  # Bearish zero crossing

        return max(0, min(100, base_score))

    def _calculate_bollinger_score(
        self, tech_signals: Dict[str, float], df: pd.DataFrame
    ) -> float:
        """Calculate Bollinger Bands-based score (0-100)"""
        bb_position = tech_signals.get("bb_position", 0.5)

        if len(df) == 0:
            return 50.0

        current_price = df["close"].iloc[-1]

        # Bollinger Band position scoring - Fixed BUY/SELL logic
        if bb_position <= 0.1:
            # Near lower band - potential bounce (SELL signal - mean reversion)
            return 25
        elif bb_position >= 0.9:
            # Near upper band - potential reversal (BUY signal - mean reversion)
            return 75
        elif bb_position <= 0.3:
            # Lower half - moderately bearish (SELL bias)
            return 40
        elif bb_position >= 0.7:
            # Upper half - moderately bullish (BUY bias)
            return 60
        else:
            # Middle - neutral
            return 50

    def _calculate_atr_score(self, tech_signals: Dict[str, float]) -> float:
        """Calculate contextual ATR-based score (0-100) with enhanced market volatility awareness"""
        atr = tech_signals.get("atr", 0.0)

        if atr <= 0:
            return 50.0

        # Get market volatility context
        avg_atr = tech_signals.get("avg_atr", atr)  # Historical average ATR
        market_volatility = atr / avg_atr if avg_atr > 0 else 1.0

        # Enhanced contextual scoring based on market conditions
        if market_volatility > 0.8:  # 高ボラティリティ市場
            # 高ATR = 好機（トレンド形成中）
            score = min(market_volatility * 60, 100)
            return max(50, min(100, score))
        elif market_volatility < 0.3:  # 低ボラティリティ市場
            # 高ATR = 注意（ノイズの可能性）
            score = 50 + (market_volatility - 1) * 20
            return max(0, min(80, score))
        else:  # 通常市場
            # バランスの取れたスコアリング
            score = 50 + (market_volatility - 1) * 25
            return max(0, min(100, score))

    def _calculate_momentum_score(self, tech_signals: Dict[str, float]) -> float:
        """Calculate momentum-based score (0-100)"""
        momentum = tech_signals.get("momentum", 0.0)

        # Momentum scoring: positive momentum = bullish, negative = bearish - More balanced
        if momentum > 1.5:  # Strong positive momentum
            score = 65 + min(momentum * 1.5, 20)  # 65-85
        elif momentum > 0.3:  # Moderate positive momentum
            score = 55 + momentum * 10  # 55-65
        elif momentum < -1.5:  # Strong negative momentum
            score = 35 - min(abs(momentum) * 1.5, 20)  # 15-35
        elif momentum < -0.3:  # Moderate negative momentum
            score = 45 + momentum * 10  # 35-45
        else:  # Neutral momentum
            score = 45 + momentum * 5  # 40-50

        return max(0, min(100, score))

    def _calculate_stochastic_score(self, tech_signals: Dict[str, float]) -> float:
        """Calculate stochastic-based score (0-100)"""
        stoch_k = tech_signals.get("stoch_k", 50.0)
        stoch_d = tech_signals.get("stoch_d", 50.0)

        # Use %K and %D crossover signals - More balanced
        if stoch_k > stoch_d and stoch_k > 70:
            # Bullish crossover in overbought zone
            score = 70 + (stoch_k - 70) * 0.5  # 70-85
        elif stoch_k > stoch_d:
            # Bullish crossover
            score = 55 + (stoch_k - 50) * 0.3  # 55-70
        elif stoch_k < stoch_d and stoch_k < 30:
            # Bearish crossover in oversold zone
            score = 30 - (30 - stoch_k) * 0.5  # 15-30
        elif stoch_k < stoch_d:
            # Bearish crossover
            score = 45 - (50 - stoch_k) * 0.3  # 30-45
        else:
            # No clear signal
            score = 45 + (stoch_k - 50) * 0.2  # 35-55

        return max(0, min(100, score))

    def _calculate_trend_score(self, df: pd.DataFrame) -> float:
        """Calculate trend-based score (0-100)"""
        trend_window = self.config.get(
            "trend_window", 10
        )  # Default to 10 if not specified

        if len(df) < trend_window:
            return 50.0

        recent_prices = df["close"].tail(trend_window).values
        if len(recent_prices) < 2:
            return 50.0

        # Simple linear trend
        x = np.arange(len(recent_prices))
        slope = np.polyfit(x, recent_prices, 1)[0]

        # Normalize slope to score
        # Positive slope = bullish, negative slope = bearish
        if slope > 0:
            score = 50 + min(slope * 1000, 50)  # 50-100
        else:
            score = 50 + max(slope * 1000, -50)  # 0-50

        return max(0, min(100, score))

    def _calculate_trend_score_enhanced(self, df: pd.DataFrame) -> float:
        """Calculate enhanced trend-based score using Phase 2 metrics (0-100)"""
        try:
            # Use enhanced trend metrics from Phase 2
            trend_metrics = calculate_trend_metrics(df, window=20)

            bull_strength = trend_metrics.get("bull_strength", 0.0)
            bear_strength = trend_metrics.get("bear_strength", 0.0)
            trend_strength = trend_metrics.get("trend_strength", 0.0)
            r_squared = trend_metrics.get("r_squared", 0.0)

            # Enhanced scoring using bull/bear strength
            if bull_strength > bear_strength:
                # Bullish trend dominant
                base_score = 50 + (bull_strength * 50)  # 50-100 range
                # Boost score based on trend consistency
                consistency_bonus = r_squared * 10
                base_score += consistency_bonus
            elif bear_strength > bull_strength:
                # Bearish trend dominant
                base_score = 50 - (bear_strength * 50)  # 0-50 range
                # Reduce score based on trend consistency
                consistency_penalty = r_squared * 10
                base_score -= consistency_penalty
            else:
                # Neutral trend
                base_score = 50.0
                # Add small trend strength bonus
                base_score += (trend_strength - 0.5) * 10

            return max(0, min(100, base_score))

        except Exception as e:
            logger.warning(f"Error in enhanced trend calculation: {e}")
            # Fallback to basic trend calculation
            return self._calculate_trend_score(df)

    def _apply_position_adjustments(
        self, score: float, portfolio: Dict[str, Any]
    ) -> float:
        """Apply position context adjustments to score"""
        # Get position information
        btc_balance = portfolio.get("btc_balance", 0.0)
        jpy_balance = portfolio.get("jpy_balance", 0.0)
        current_price = portfolio.get("current_price", 0.0)

        if current_price <= 0:
            return score

        # Calculate position ratio
        btc_value = btc_balance * current_price
        total_value = btc_value + jpy_balance
        position_ratio = btc_value / total_value if total_value > 0 else 0.0

        # Adjust score based on position
        if position_ratio > 0.8:
            # Overexposed - reduce buy signals, increase sell signals
            if score > 50:
                score = score * 0.8  # Reduce bullish signals
            else:
                score = score * 1.2  # Increase bearish signals
        elif position_ratio < 0.2:
            # Underexposed - increase buy signals, reduce sell signals
            if score > 50:
                score = score * 1.2  # Increase bullish signals
            else:
                score = score * 0.8  # Reduce bearish signals

        return max(0, min(100, score))

    def _blend_continuous_action(
        self, quality_score: float, continuous_action: float
    ) -> float:
        """Blend quality score with continuous action"""
        # Convert continuous action (-1, 1) to score (0, 100)
        action_score = (continuous_action + 1) * CONTINUOUS_TO_SCORE_SCALE

        # Blend with quality score (80% quality, 20% action for more deterministic behavior)
        blended_score = quality_score * 0.8 + action_score * 0.2

        return max(0, min(100, blended_score))

    def _apply_ensemble_integration(
        self,
        base_score: float,
        df: pd.DataFrame,
        continuous_action: float,
        portfolio: Dict[str, Any],
    ) -> float:
        """
        Phase 3: Apply ensemble signal integration to enhance base score.

        Args:
            base_score: Base signal quality score from Phase 2
            df: Market data DataFrame
            continuous_action: Continuous action value
            portfolio: Portfolio information

        Returns:
            Enhanced score with ensemble integration
        """
        try:
            # Generate ensemble signal
            market_data = {
                "df": df,
                "continuous_action": continuous_action,
                "portfolio": portfolio,
            }
            (
                ensemble_score,
                ensemble_confidence,
            ) = self.ensemble_generator.generate_ensemble_signal(market_data)

            # Calculate ensemble weight based on confidence
            ensemble_weight = self.ensemble_weight * ensemble_confidence

            # Get ensemble action score (normalized to -1 to 1 range)
            ensemble_action = (
                ensemble_score - 50
            ) / 50  # Convert 0-100 to -1 to 1 range

            # Blend ensemble signal with base score
            # Higher confidence = more weight to ensemble signal
            blended_score = (
                base_score * (1 - ensemble_weight)
                + ensemble_action * ensemble_weight * 100
            )  # Scale to score range

            # Apply reliability adjustment
            reliability_factor = 0.8 + (ensemble_confidence * 0.4)  # 0.8 to 1.2 range
            final_score = blended_score * reliability_factor

            self.logger.debug(
                f"Ensemble integration - Base: {base_score:.2f}, "
                f"Ensemble: {ensemble_score:.2f}, Confidence: {ensemble_confidence:.3f}, "
                f"Final: {final_score:.2f}"
            )

            return final_score

        except Exception as e:
            self.logger.warning(f"Ensemble integration failed: {e}")
            return base_score  # Fallback to base score

    def _score_to_action(self, score: float) -> int:
        """Convert score to discrete action"""
        # Use common utility with parity support
        return score_to_discrete_action(
            score,
            buy_threshold=self.buy_threshold,
            sell_threshold=self.sell_threshold,
            high_score_is_buy=HIGH_SCORE_IS_BUY,
        )

    def _fallback_action(self, continuous_action: float) -> int:
        """Fallback action conversion from continuous action"""
        if continuous_action > 0.3:  # Higher threshold for fallback BUY
            return 1
        elif continuous_action < -0.3:  # Lower threshold for fallback SELL
            return -1
        else:
            return 0

    def _determine_market_regime(self, df: pd.DataFrame) -> str:
        """
        Determine current market regime using trend metrics

        Returns:
            Market regime: 'trending', 'ranging', 'volatile'
        """
        try:
            # Use trend metrics to determine regime
            trend_metrics = calculate_trend_metrics(df, window=20)

            from ztb.features.generators.technical.volatility.return_std import (
                compute_return_stddev,
            )
            from ztb.trading.constants import TRADING_DAYS_PER_YEAR

            vol_series = compute_return_stddev(df, period=len(df))
            last_val = vol_series.iloc[-1]
            volatility = (
                float(last_val) * np.sqrt(TRADING_DAYS_PER_YEAR)
                if not pd.isna(last_val)
                else 0.0
            )
            trend_strength = trend_metrics.get("trend_strength", 0.0)
            r_squared = trend_metrics.get("r_squared", 0.0)

            # Determine regime based on trend strength and volatility
            if trend_strength > 0.7 and r_squared > 0.6:
                return "trending"  # Strong trend with high consistency
            elif volatility > 0.05:  # High volatility (>5% annualized)
                return "volatile"
            else:
                return "ranging"  # Default ranging market

        except Exception as e:
            logger.warning(f"Error determining market regime: {e}")
            return "ranging"  # Default fallback

    def _get_enhanced_signals(
        self, df: pd.DataFrame, market_regime: str
    ) -> Dict[str, Any]:
        """
        Get enhanced signals using Phase 2 advanced indicators

        Args:
            df: Market data DataFrame
            market_regime: Current market regime

        Returns:
            Enhanced signal dictionary
        """
        enhanced_signals = {}

        try:
            # Get adaptive RSI and MACD signals
            adaptive_rsi_result = self.adaptive_rsi.calculate_adaptive(
                df, market_regime
            )
            adaptive_macd_result = self.adaptive_macd.calculate_adaptive(
                df, market_regime
            )

            # Merge results
            enhanced_signals.update(adaptive_rsi_result)
            enhanced_signals.update(adaptive_macd_result)

            # Get composite signal
            composite_result = self.composite_indicator.calculate(df)
            enhanced_signals.update(composite_result)

            # Add trend metrics (bull_strength, bear_strength)
            trend_metrics = calculate_trend_metrics(df, window=20)
            enhanced_signals.update(trend_metrics)

            # Add market regime info
            enhanced_signals["market_regime"] = market_regime

        except Exception as e:
            logger.warning(f"Error getting enhanced signals: {e}")
            # Fallback to basic technical indicators
            enhanced_signals = self.technical_indicators.get_technical_signals(df)

        return enhanced_signals
