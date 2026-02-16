#!/usr/bin/env python3
"""
Advanced Signal Guidance System
Type-safe, high-performance signal guidance for SAC action conversion
"""

from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Literal, Optional, Union

import numpy as np
import pandas as pd

from ztb.utils.logging_utils import get_logger

logger = get_logger(__name__)

from ztb.trading.signal.common.utilities import score_to_discrete_action
from ztb.trading.signal.constants import DEFAULT_FALLBACK_THRESHOLD, HIGH_SCORE_IS_BUY
from ztb.trading.signal.multi_timeframe_analyzer import (
    ConvergenceAnalysis,
    MultiTimeframeAnalyzer,
    Timeframe,
)
from ztb.trading.signal.quality_scorer import SignalQualityScorer
from ztb.trading.signal.trend_convergence_calculator import TrendConvergenceCalculator


class MarketTrend(Enum):
    """Market trend enumeration"""

    BULLISH = "bullish"
    BEARISH = "bearish"
    NEUTRAL = "neutral"


class SignalType(Enum):
    """Signal type enumeration"""

    BUY = 1
    SELL = -1
    HOLD = 0


@dataclass
class MarketContext:
    """Structured market context data"""

    price_trend: List[float] = field(default_factory=list)
    volume_trend: List[float] = field(default_factory=list)
    trend_window: int = 10
    current_trend: MarketTrend = MarketTrend.NEUTRAL


@dataclass
class PositionContext:
    """Structured position context data"""

    has_position: bool
    position_ratio: float
    is_overexposed: bool
    is_underexposed: bool
    can_buy: bool
    btc_balance: float
    jpy_balance: float
    total_value: float


@dataclass
class SignalContext:
    """Structured signal context data"""

    recent_bias: Literal["buy", "sell", "neutral"]
    signal_streak: int
    last_signal: Optional[SignalType]
    sell_signal_ratio: float


@dataclass
class GuidanceConfig:
    """Configuration for signal guidance system"""

    guidance_level: Literal["conservative", "adaptive", "aggressive"] = "adaptive"
    max_history: int = 20
    base_threshold: float = 0.33
    aggressive_threshold: float = 0.15
    conservative_threshold: float = 0.5
    trend_window: int = 10
    sell_injection_base_probability: float = 0.15
    sell_injection_bearish_multiplier: float = 1.5
    sell_injection_overexposed_multiplier: float = 1.8
    sell_injection_no_recent_sell_multiplier: float = 2.0
    sell_injection_streak_penalty: float = 0.3
    sell_injection_max_probability: float = 0.5
    trend_sell_probability: float = 0.4
    signal_sell_probability: float = 0.25
    signal_sell_recent_threshold: float = 0.1


class SignalGuidanceSystem:
    """
    Enhanced signal guidance system for SAC action conversion

    Features:
    - Type-safe implementation with structured data classes
    - Deterministic signal generation with probabilistic enhancements
    - Market trend analysis and position-aware guidance
    - Configurable thresholds and behavior
    """

    def __init__(
        self,
        config: Optional[GuidanceConfig] = None,
        threshold_manager: Optional[object] = None,
    ):
        self.config = config or GuidanceConfig()
        self.signal_history: deque[SignalType] = deque(maxlen=self.config.max_history)
        self.market_context = MarketContext()
        # Initialize quality scorer for deterministic scoring
        self.quality_scorer = SignalQualityScorer(threshold_manager=threshold_manager)
        # Initialize multi-timeframe analyzer for Phase 2 enhancement
        self.multi_timeframe_analyzer = MultiTimeframeAnalyzer()
        # Initialize trend convergence calculator for Phase 2 enhancement
        self.convergence_calculator = TrendConvergenceCalculator()
        # Keep market data history for technical indicator calculations
        self.market_data_history: deque[pd.Series] = deque(maxlen=100)
        self.max_history_size = 100  # Keep last 100 data points for technical analysis

    def update_market_context(self, row: pd.Series, portfolio: Dict[str, Any]) -> None:
        """Update market context for guidance decisions"""
        # Track price trend
        if len(self.market_context.price_trend) >= self.config.trend_window:
            self.market_context.price_trend.pop(0)
        close_price = row.get("close", row.get("price", 0))
        self.market_context.price_trend.append(float(close_price))

        # Track volume trend
        if "volume" in row.index:
            if len(self.market_context.volume_trend) >= self.config.trend_window:
                self.market_context.volume_trend.pop(0)
            volume_val = row["volume"]
            # Safely convert to float
            try:
                if hasattr(volume_val, "iloc"):
                    volume_val = volume_val.iloc[0] if len(volume_val) > 0 else 0
                volume_float = float(volume_val)  # type: ignore
                self.market_context.volume_trend.append(volume_float)
            except (ValueError, TypeError):
                self.market_context.volume_trend.append(0.0)

        # Update current trend
        self.market_context.current_trend = self._analyze_market_trend()

        # Update multi-timeframe data for Phase 2 enhancement
        self._update_multi_timeframe_data(row)

    def get_market_trend(self) -> MarketTrend:
        """Get current market trend"""
        return self.market_context.current_trend

    def get_position_context(self, portfolio: Dict[str, Any]) -> PositionContext:
        """Analyze current position context"""
        btc_balance = float(portfolio["btc_balance"])
        jpy_balance = float(portfolio["jpy_balance"])
        total_value = float(portfolio["portfolio_value"])
        current_price = float(portfolio.get("current_price", 0))

        # Position size relative to total portfolio
        position_ratio = (
            (btc_balance * current_price) / total_value if total_value > 0 else 0.0
        )

        return PositionContext(
            has_position=btc_balance > 0.0001,
            position_ratio=position_ratio,
            is_overexposed=position_ratio > 0.8,
            is_underexposed=position_ratio < 0.1,
            can_buy=jpy_balance > 1000,
            btc_balance=btc_balance,
            jpy_balance=jpy_balance,
            total_value=total_value,
        )

    def get_signal_context(self) -> SignalContext:
        """Analyze recent signal patterns"""
        if len(self.signal_history) < 3:
            return SignalContext(
                recent_bias="neutral",
                signal_streak=0,
                last_signal=None,
                sell_signal_ratio=0.0,
            )

        recent_signals = self.signal_history[-3:]

        # Check for bias
        buy_count = sum(1 for s in recent_signals if s == SignalType.BUY)
        sell_count = sum(1 for s in recent_signals if s == SignalType.SELL)
        hold_count = sum(1 for s in recent_signals if s == SignalType.HOLD)

        if buy_count > sell_count and buy_count > hold_count:
            bias = "buy"
        elif sell_count > buy_count and sell_count > hold_count:
            bias = "sell"
        else:
            bias = "neutral"

        # Check for streaks
        last_signal = recent_signals[-1]
        streak = 0
        for s in reversed(recent_signals):
            if s == last_signal:
                streak += 1
            else:
                break

        # Calculate sell signal ratio
        total_recent = (
            len(self.signal_history[-10:])
            if len(self.signal_history) >= 10
            else len(self.signal_history)
        )
        sell_ratio = (
            sum(1 for s in self.signal_history[-10:] if s == SignalType.SELL)
            / total_recent
            if total_recent > 0
            else 0.0
        )

        return SignalContext(
            recent_bias=bias,
            signal_streak=streak,
            last_signal=last_signal,
            sell_signal_ratio=sell_ratio,
        )

    def apply_guidance(
        self, continuous_action: float, row: pd.Series, portfolio: Dict[str, Any]
    ) -> int:
        """Apply intelligent signal guidance using deterministic quality scoring with Phase 2 multi-timeframe enhancement"""
        try:
            # Update context for backward compatibility
            self.update_market_context(row, portfolio)

            # Add current market data to history for technical analysis
            self.market_data_history.append(row.copy())

            # Create DataFrame from recent market data for technical analysis
            market_df = self._create_market_dataframe(row, portfolio)

            # Phase 2: Get multi-timeframe convergence analysis
            convergence_analysis = self.multi_timeframe_analyzer.analyze_convergence()

            # Get trend analyses for convergence calculation
            trend_analyses = {}
            for timeframe in self.multi_timeframe_analyzer.timeframes.keys():
                analysis = self.multi_timeframe_analyzer.analyze_timeframe_trend(
                    timeframe
                )
                if analysis:
                    trend_analyses[timeframe] = analysis

            convergence_report = self.convergence_calculator.get_convergence_report(
                trend_analyses
            )

            # Use quality scorer for deterministic signal generation
            guided_action, quality_score = self.quality_scorer.calculate_signal_quality(
                market_df, continuous_action, portfolio
            )

            # Phase 2: Apply convergence-based enhancement to quality score
            enhanced_score = self._apply_convergence_enhancement(
                quality_score, convergence_analysis, convergence_report
            )

            # Convert enhanced score to action
            guided_action = self._convert_score_to_action(enhanced_score)

            # Apply minimal position-based safety checks (deterministic)
            guided_action = self._apply_position_safety(guided_action, portfolio)

            # Record signal
            signal_type = SignalType(guided_action)
            self.signal_history.append(signal_type)

            return guided_action

        except Exception as e:
            # Fallback to simple threshold conversion on error
            logger.warning(f"Quality scoring failed, using fallback: {e}")
            return self._fallback_conversion(continuous_action)

    def _analyze_market_trend(self) -> MarketTrend:
        """Analyze current market trend"""
        if len(self.market_context.price_trend) < 5:
            return MarketTrend.NEUTRAL

        # Simple trend analysis
        recent_prices = self.market_context.price_trend[-5:]
        if recent_prices[-1] > recent_prices[0] * 1.002:  # 0.2% up
            return MarketTrend.BULLISH
        elif recent_prices[-1] < recent_prices[0] * 0.998:  # 0.2% down
            return MarketTrend.BEARISH
        else:
            return MarketTrend.NEUTRAL

    def _get_adaptive_threshold(
        self,
        market_trend: MarketTrend,
        position_ctx: PositionContext,
        signal_ctx: SignalContext,
    ) -> float:
        """Get adaptive threshold based on market conditions"""
        if self.config.guidance_level == "conservative":
            return self.config.conservative_threshold
        elif self.config.guidance_level == "aggressive":
            return self.config.aggressive_threshold

        # Adaptive logic
        base_threshold = self.config.base_threshold

        # Adjust based on market trend
        if market_trend == MarketTrend.BULLISH:
            base_threshold *= 0.8  # More sensitive to BUY signals
        elif market_trend == MarketTrend.BEARISH:
            base_threshold *= 0.9  # Slightly more sensitive to SELL signals

        # Adjust based on position
        if position_ctx.is_overexposed:
            base_threshold *= 1.2  # Less sensitive to BUY, more HOLD
        elif position_ctx.is_underexposed:
            base_threshold *= 0.9  # More sensitive to BUY

        # Adjust based on signal patterns
        if signal_ctx.signal_streak >= 3:
            base_threshold *= 1.1  # Encourage diversification after streaks

        return min(base_threshold, 0.8)  # Cap at reasonable maximum

    def _apply_market_guidance(
        self,
        continuous_action: float,
        threshold: float,
        market_trend: MarketTrend,
        position_ctx: PositionContext,
        signal_ctx: SignalContext,
    ) -> int:
        """Apply market-aware guidance to action conversion"""
        # Base conversion
        if continuous_action > threshold:
            base_action = SignalType.BUY.value
        elif continuous_action < -threshold:
            base_action = SignalType.SELL.value
        else:
            base_action = SignalType.HOLD.value

        # Apply probabilistic SELL signal injection for diversity
        if base_action == SignalType.HOLD.value and position_ctx.has_position:
            # Inject SELL signals probabilistically to ensure trading diversity
            sell_probability = self._calculate_sell_probability(
                market_trend, position_ctx, signal_ctx
            )
            if np.random.random() < sell_probability:
                base_action = SignalType.SELL.value

        # Apply guidance rules
        guided_action = self._apply_position_guidance(base_action, position_ctx)
        guided_action = self._apply_trend_guidance(
            guided_action, market_trend, position_ctx
        )
        guided_action = self._apply_signal_guidance(guided_action, signal_ctx)

        return guided_action

    def _calculate_sell_probability(
        self,
        market_trend: MarketTrend,
        position_ctx: PositionContext,
        signal_ctx: SignalContext,
    ) -> float:
        """Calculate probability of injecting SELL signal for diversity"""
        base_probability = self.config.sell_injection_base_probability

        # Increase probability in bearish markets
        if market_trend == MarketTrend.BEARISH:
            base_probability *= self.config.sell_injection_bearish_multiplier

        # Increase probability when overexposed
        if position_ctx.is_overexposed:
            base_probability *= self.config.sell_injection_overexposed_multiplier

        # Increase probability when no recent SELL signals
        recent_signals = (
            len(self.signal_history[-5:])
            if len(self.signal_history) >= 5
            else len(self.signal_history)
        )
        sell_count = (
            sum(1 for s in self.signal_history[-5:] if s == SignalType.SELL)
            if recent_signals > 0
            else 0
        )
        if sell_count == 0:
            base_probability *= self.config.sell_injection_no_recent_sell_multiplier

        # Decrease probability if recent SELL streak
        if signal_ctx.last_signal == SignalType.SELL and signal_ctx.signal_streak >= 2:
            base_probability *= self.config.sell_injection_streak_penalty

        return min(base_probability, self.config.sell_injection_max_probability)

    def _apply_position_guidance(
        self, action: int, position_ctx: PositionContext
    ) -> int:
        """Apply position-based guidance"""
        # Prevent overexposure
        if action == SignalType.BUY.value and position_ctx.is_overexposed:
            return SignalType.HOLD.value

        # Prevent selling when no position
        if action == SignalType.SELL.value and not position_ctx.has_position:
            return SignalType.HOLD.value

        # Prevent buying when no funds
        if action == SignalType.BUY.value and not position_ctx.can_buy:
            return SignalType.HOLD.value

        return action

    def _apply_trend_guidance(
        self, action: int, market_trend: MarketTrend, position_ctx: PositionContext
    ) -> int:
        """Apply market trend-based guidance"""
        # In bullish markets, be more open to BUY signals
        if market_trend == MarketTrend.BULLISH:
            if (
                action == SignalType.HOLD.value
                and position_ctx.is_underexposed
                and position_ctx.can_buy
            ):
                # Consider BUY instead of HOLD when underexposed in bullish market
                if np.random.random() < 0.3:
                    return SignalType.BUY.value

        # In bearish markets, be more open to SELL signals
        elif market_trend == MarketTrend.BEARISH:
            if action == SignalType.HOLD.value and position_ctx.has_position:
                # Consider SELL instead of HOLD when having position in bearish market
                if np.random.random() < self.config.trend_sell_probability:
                    return SignalType.SELL.value
            elif action == SignalType.BUY.value and not position_ctx.is_underexposed:
                # Consider HOLD instead of BUY when not underexposed in bearish market
                if np.random.random() < 0.4:
                    return SignalType.HOLD.value

        return action

    def _apply_signal_guidance(self, action: int, signal_ctx: SignalContext) -> int:
        """Apply signal pattern-based guidance"""
        # Prevent excessive streaks
        if signal_ctx.signal_streak >= 4:
            streak_signal = signal_ctx.last_signal
            if streak_signal is not None:
                # If we've been BUYING too much, encourage diversification
                if streak_signal == SignalType.BUY and action == SignalType.BUY.value:
                    if np.random.random() < 0.5:
                        return SignalType.HOLD.value

                # If we've been SELLING too much, encourage diversification
                elif (
                    streak_signal == SignalType.SELL and action == SignalType.SELL.value
                ):
                    if np.random.random() < 0.5:
                        return SignalType.HOLD.value

        # Balance bias and promote SELL signals when needed
        if signal_ctx.recent_bias == "buy" and signal_ctx.signal_streak >= 2:
            # If too many BUY signals recently, slightly favor SELL/HOLD
            if action == SignalType.BUY.value:
                if np.random.random() < 0.2:
                    return SignalType.HOLD.value

        elif signal_ctx.recent_bias == "sell" and signal_ctx.signal_streak >= 2:
            # If too many SELL signals recently, slightly favor BUY/HOLD
            if action == SignalType.SELL.value:
                if np.random.random() < 0.2:
                    return SignalType.HOLD.value

        # Promote SELL signals when there are too few recent sells
        if (
            signal_ctx.sell_signal_ratio < self.config.signal_sell_recent_threshold
            and action == SignalType.HOLD.value
        ):
            # Less than threshold sells, consider injecting SELL
            if np.random.random() < self.config.signal_sell_probability:
                return SignalType.SELL.value

        return action

    def _create_market_dataframe(
        self, row: pd.Series, portfolio: Dict[str, Any]
    ) -> pd.DataFrame:
        """Create market DataFrame from current row and historical context"""
        try:
            # Use market data history if available (preferred for technical analysis)
            if (
                len(self.market_data_history) >= 30
            ):  # Minimum for most technical indicators
                # Create DataFrame from recent history
                recent_data = self.market_data_history[
                    -50:
                ]  # Use last 50 points for analysis
                data = {
                    "open": [r.get("open", r.get("price", 0)) for r in recent_data],
                    "high": [r.get("high", r.get("price", 0)) for r in recent_data],
                    "low": [r.get("low", r.get("price", 0)) for r in recent_data],
                    "close": [r.get("close", r.get("price", 0)) for r in recent_data],
                    "volume": [r.get("volume", 1.0) for r in recent_data],
                }
                return pd.DataFrame(data)
            else:
                # Fallback: use recent price trend for technical analysis
                return self._create_fallback_dataframe(row, portfolio)
        except Exception as e:
            logger.warning(f"Error creating market DataFrame: {e}")
            return self._create_fallback_dataframe(row, portfolio)

    def _create_fallback_dataframe(
        self, row: pd.Series, portfolio: Dict[str, Any]
    ) -> pd.DataFrame:
        """Create fallback DataFrame when insufficient historical data is available"""
        # Use recent price trend for technical analysis
        recent_prices = (
            self.market_context.price_trend[-50:]
            if len(self.market_context.price_trend) >= 50
            else self.market_context.price_trend
        )

        if not recent_prices:
            # Fallback: create minimal DataFrame from current row
            current_price = float(row.get("close", row.get("price", 0)))
            high_price = float(row.get("high", current_price))
            low_price = float(row.get("low", current_price))
            volume = float(row.get("volume", 0))

            data = {
                "open": [current_price],
                "high": [high_price],
                "low": [low_price],
                "close": [current_price],
                "volume": [volume],
            }
        else:
            # Create DataFrame from historical context
            # Assume OHLCV data is available in row
            current_price = float(
                row.get(
                    "close", row.get("price", recent_prices[-1] if recent_prices else 0)
                )
            )
            high_price = float(row.get("high", current_price))
            low_price = float(row.get("low", current_price))
            volume = float(row.get("volume", 0))

            # Create time series from recent prices (simplified)
            n_points = min(len(recent_prices), 50)
            prices = recent_prices[-n_points:] + [current_price]

            data = {
                "open": prices[:-1] + [prices[-1]],  # Simplified
                "high": [max(p, current_price) for p in prices[:-1]] + [high_price],
                "low": [min(p, current_price) for p in prices[:-1]] + [low_price],
                "close": prices,
                "volume": [volume] * len(prices),  # Simplified
            }

        return pd.DataFrame(data)

    def _apply_position_safety(self, action: int, portfolio: Dict[str, Any]) -> int:
        """Apply basic position-based safety checks (deterministic)"""
        # Get position information
        btc_balance = portfolio.get("btc_balance", 0.0)
        jpy_balance = portfolio.get("jpy_balance", 0.0)
        current_price = portfolio.get("current_price", 0.0)

        if current_price <= 0:
            return action

        # Calculate position value
        btc_value = btc_balance * current_price
        total_value = btc_value + jpy_balance

        # Basic safety checks
        if action == SignalType.BUY.value:
            # Don't buy if no JPY balance
            if jpy_balance < current_price * 0.001:  # Minimum trade size
                return SignalType.HOLD.value
        elif action == SignalType.SELL.value:
            # Don't sell if no BTC balance
            if btc_balance < 0.001:  # Minimum BTC amount
                return SignalType.HOLD.value

        return action

    def _fallback_conversion(self, continuous_action: float) -> int:
        """Fallback action conversion from continuous action"""
        # Use a conservative default threshold for fallback conversions to preserve sensitivity
        # to smaller continuous actions while avoiding noisy conversions.
        fallback_threshold = DEFAULT_FALLBACK_THRESHOLD
        if continuous_action >= fallback_threshold:
            return SignalType.BUY.value
        elif continuous_action <= -fallback_threshold:
            return SignalType.SELL.value
        else:
            return SignalType.HOLD.value

    def _update_multi_timeframe_data(self, row: pd.Series) -> None:
        """
        Update multi-timeframe analyzer with current market data

        Phase 2: Multi-timeframe trend analysis enhancement
        """
        try:
            close_price = float(row.get("close", row.get("price", 0)))
            volume = float(row.get("volume", 1.0))

            # Update all timeframes with current data
            # Note: In production, this would receive actual timeframe-specific data
            # For now, we use the same data for all timeframes as a simplified approach
            for timeframe in [Timeframe.M1, Timeframe.M5, Timeframe.M15]:
                self.multi_timeframe_analyzer.update_timeframe_data(
                    timeframe=timeframe, price=close_price, volume=volume
                )

        except Exception as e:
            logger.warning(f"Failed to update multi-timeframe data: {e}")

    def _apply_convergence_enhancement(
        self,
        base_score: float,
        convergence_analysis: "ConvergenceAnalysis",
        convergence_report: Dict[str, Union[float, str]],
    ) -> float:
        """
        Apply Phase 2 convergence enhancement to base quality score

        Args:
            base_score: Base quality score from SignalQualityScorer (0-100)
            convergence_analysis: Multi-timeframe convergence analysis
            convergence_report: Detailed convergence metrics

        Returns:
            Enhanced score with convergence weighting
        """
        try:
            convergence_score = convergence_analysis.convergence_score
            recommendation = convergence_report.get(
                "recommendation", "weak_convergence"
            )

            # Base enhancement factor from convergence
            enhancement_factor = convergence_score / 100.0  # 0-1 scale

            # Apply recommendation-based adjustments
            if recommendation == "strong_convergence":
                # Boost score for strong convergence
                enhancement_factor *= 1.2
            elif recommendation == "moderate_convergence":
                # Moderate boost
                enhancement_factor *= 1.1
            elif recommendation == "weak_convergence":
                # Slight boost
                enhancement_factor *= 1.05
            elif recommendation == "divergence":
                # Reduce score for divergence
                enhancement_factor *= 0.9

            # Apply convergence weighting to base score
            # Weight convergence at 30% for Phase 2 enhancement
            convergence_weight = 0.3
            enhanced_score = (
                base_score * (1 - convergence_weight)
                + (base_score * enhancement_factor) * convergence_weight
            )

            # Ensure score stays within valid range
            enhanced_score = max(0.0, min(100.0, enhanced_score))

            logger.debug(
                f"Phase 2 enhancement: base_score={base_score:.1f}, "
                f"convergence={convergence_score:.1f}, "
                f"recommendation={recommendation}, "
                f"enhanced_score={enhanced_score:.1f}"
            )

            return enhanced_score

        except Exception as e:
            logger.warning(f"Convergence enhancement failed, using base score: {e}")
            return base_score

    def _convert_score_to_action(self, score: float) -> int:
        """
        Convert enhanced quality score to discrete action

        Phase 2: Enhanced score-to-action conversion with convergence weighting

        Args:
            score: Enhanced quality score (0-100)

        Returns:
            Discrete action: 1 (BUY), -1 (SELL), 0 (HOLD)
        """
        # Phase 2 thresholds with convergence enhancement
        # Higher convergence can lower thresholds for more responsive signals
        buy_threshold = 85.0  # BUY if score >= 85
        sell_threshold = 15.0  # SELL if score <= 15
        # Use shared utility to preserve parity support
        return score_to_discrete_action(
            score,
            buy_threshold=buy_threshold,
            sell_threshold=sell_threshold,
            high_score_is_buy=HIGH_SCORE_IS_BUY,
        )

    def get_multi_timeframe_analysis(self) -> Dict[str, Any]:
        """
        Get comprehensive multi-timeframe analysis for Phase 2

        Returns:
            Dictionary containing convergence analysis and timeframe details
        """
        try:
            # Get convergence analysis
            convergence = self.multi_timeframe_analyzer.analyze_convergence()

            # Get trend analyses for all timeframes
            trend_analyses = {}
            for timeframe in self.multi_timeframe_analyzer.timeframes.keys():
                analysis = self.multi_timeframe_analyzer.analyze_timeframe_trend(
                    timeframe
                )
                if analysis:
                    trend_analyses[timeframe.value] = {
                        "direction": analysis.direction.value,
                        "strength": analysis.strength,
                        "momentum": analysis.momentum,
                        "rsi": analysis.rsi,
                        "macd_signal": analysis.macd_signal,
                        "bollinger_position": analysis.bollinger_position,
                    }

            # Get convergence report
            trend_analyses_dict = {}
            for tf in self.multi_timeframe_analyzer.timeframes.keys():
                analysis = self.multi_timeframe_analyzer.analyze_timeframe_trend(tf)
                if analysis is not None:
                    trend_analyses_dict[tf] = analysis

            convergence_report = self.convergence_calculator.get_convergence_report(
                trend_analyses_dict
            )

            return {
                "phase": "Phase 2 - Multi-timeframe Analysis",
                "convergence": {
                    "score": convergence.convergence_score,
                    "dominant_trend": convergence.dominant_trend.value,
                    "timeframe_agreement": convergence.timeframe_agreement,
                    "short_term_bias": convergence.short_term_bias.value,
                    "medium_term_bias": convergence.medium_term_bias.value,
                },
                "convergence_report": convergence_report,
                "timeframe_analyses": trend_analyses,
                "data_points": {
                    tf.value: len(data.prices)
                    for tf, data in self.multi_timeframe_analyzer.timeframes.items()
                },
            }

        except Exception as e:
            logger.error(f"Failed to get multi-timeframe analysis: {e}")
            return {"error": str(e), "phase": "Phase 2 - Error"}

    def get_phase_2_status(self) -> Dict[str, Any]:
        """
        Get Phase 2 implementation status and metrics

        Returns:
            Dictionary with Phase 2 status information
        """
        analysis = self.get_multi_timeframe_analysis()

        return {
            "phase": "Phase 2 - Multi-timeframe Trend Detection",
            "status": "active" if "error" not in analysis else "error",
            "components": {
                "MultiTimeframeAnalyzer": "active",
                "TrendConvergenceCalculator": "active",
                "SignalGuidanceSystem": "enhanced",
            },
            "metrics": {
                "convergence_score": analysis.get("convergence", {}).get("score", 0),
                "timeframe_agreement": analysis.get("convergence", {}).get(
                    "timeframe_agreement", 0
                ),
                "data_points_m1": analysis.get("data_points", {}).get("1m", 0),
                "data_points_m5": analysis.get("data_points", {}).get("5m", 0),
                "data_points_m15": analysis.get("data_points", {}).get("15m", 0),
            },
            "recommendation": analysis.get("convergence_report", {}).get(
                "recommendation", "unknown"
            ),
        }
