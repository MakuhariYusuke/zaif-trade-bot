#!/usr/bin/env python3
"""
Multi-Timeframe Analyzer for Signal Quality Enhancement

Phase 2: マイクロトレンド検出システム
複数時間軸同時分析による精度向上

Features:
- 1分足、5分足、15分足のトレンド分析
- 時間軸間トレンド収束度計算
- 短期・中期トレンドの統合評価
"""

from typing import Optional, Any
from dataclasses import dataclass, field
from enum import Enum
import numpy as np
from collections import deque

from ztb.utils.logging_utils import get_logger
from ztb.trading.signal.technical_indicators import TechnicalIndicators

logger = get_logger(__name__)

class Timeframe(Enum):
    """Timeframe enumeration"""
    M1 = "1m"    # 1 minute
    M5 = "5m"    # 5 minutes
    M15 = "15m"  # 15 minutes

class TrendDirection(Enum):
    """Trend direction enumeration"""
    STRONG_BULLISH = "strong_bullish"
    BULLISH = "bullish"
    NEUTRAL = "neutral"
    BEARISH = "bearish"
    STRONG_BEARISH = "strong_bearish"

@dataclass
class TimeframeData:
    """Timeframe-specific market data"""
    timeframe: Timeframe
    prices: deque = field(default_factory=lambda: deque(maxlen=100))
    volumes: deque = field(default_factory=lambda: deque(maxlen=100))
    timestamps: deque = field(default_factory=lambda: deque(maxlen=100))

    def add_data(self, price: float, volume: float, timestamp: float | None = None) -> None:
        """Add new price/volume data"""
        self.prices.append(price)
        self.volumes.append(volume)
        if timestamp is not None:
            self.timestamps.append(timestamp)

    def get_price_array(self) -> np.ndarray:
        """Get price data as numpy array"""
        return np.array(list(self.prices))

    def get_volume_array(self) -> np.ndarray:
        """Get volume data as numpy array"""
        return np.array(list(self.volumes))

    def has_minimum_data(self, min_points: int = 20) -> bool:
        """Check if we have minimum data points"""
        return len(self.prices) >= min_points

@dataclass
class TrendAnalysis:
    """Trend analysis result for a timeframe"""
    direction: TrendDirection
    strength: float  # 0-100
    momentum: float  # -100 to 100
    rsi: float
    macd_signal: str  # "bullish", "bearish", "neutral"
    bollinger_position: str  # "upper", "middle", "lower"

@dataclass
class ConvergenceAnalysis:
    """Multi-timeframe convergence analysis"""
    convergence_score: float  # 0-100 (higher = more converged)
    dominant_trend: TrendDirection
    timeframe_agreement: float  # 0-1 (1.0 = all timeframes agree)
    short_term_bias: TrendDirection
    medium_term_bias: TrendDirection

class MultiTimeframeAnalyzer:
    """
    Multi-timeframe trend analyzer for enhanced signal quality

    Analyzes trends across multiple timeframes (1m, 5m, 15m) and calculates
    convergence scores to improve signal reliability.
    """

    def __init__(self) -> None:
        self.technical_indicators = TechnicalIndicators()
        self.timeframes = {
            Timeframe.M1: TimeframeData(Timeframe.M1),
            Timeframe.M5: TimeframeData(Timeframe.M5),
            Timeframe.M15: TimeframeData(Timeframe.M15)
        }

        # Minimum data points required for analysis
        self.min_data_points = {
            Timeframe.M1: 20,
            Timeframe.M5: 12,
            Timeframe.M15: 8
        }

        logger.info("MultiTimeframeAnalyzer initialized")

    def update_timeframe_data(self, timeframe: Timeframe, price: float,
                            volume: float, timestamp: float | None = None) -> None:
        """
        Update data for specific timeframe

        Args:
            timeframe: Target timeframe
            price: Current price
            volume: Current volume
            timestamp: Optional timestamp
        """
        if timeframe not in self.timeframes:
            logger.warning(f"Unknown timeframe: {timeframe}")
            return

        self.timeframes[timeframe].add_data(price, volume, timestamp)
        logger.debug(f"Updated {timeframe.value} data: price={price}, volume={volume}")

    def analyze_timeframe_trend(self, timeframe: Timeframe) -> TrendAnalysis | None:
        """
        Analyze trend for specific timeframe

        Args:
            timeframe: Target timeframe

        Returns:
            TrendAnalysis or None if insufficient data
        """
        data = self.timeframes[timeframe]

        if not data.has_minimum_data(self.min_data_points[timeframe]):
            logger.debug(f"Insufficient data for {timeframe.value} analysis")
            return None

        try:
            prices = data.get_price_array()

            # Calculate RSI
            rsi = self.technical_indicators.calculate_rsi(prices)

            # Calculate MACD components
            macd_line, signal_line, histogram = self.technical_indicators.calculate_macd(prices)

            # Determine MACD signal
            macd_signal = "neutral"
            if histogram > 0 and macd_line > signal_line:
                macd_signal = "bullish"
            elif histogram < 0 and macd_line < signal_line:
                macd_signal = "bearish"

            # Calculate Bollinger Bands position
            upper, middle, lower = self.technical_indicators.calculate_bollinger_bands(prices)
            current_price = prices[-1]

            bollinger_position = "middle"
            if current_price > upper:
                bollinger_position = "upper"
            elif current_price < lower:
                bollinger_position = "lower"

            # Calculate trend direction and strength
            direction, strength, momentum = self._calculate_trend_direction(prices, rsi, macd_signal)

            return TrendAnalysis(
                direction=direction,
                strength=strength,
                momentum=momentum,
                rsi=rsi,
                macd_signal=macd_signal,
                bollinger_position=bollinger_position
            )

        except Exception as e:
            logger.error(f"Error analyzing {timeframe.value} trend: {e}")
            return None

    def analyze_convergence(self) -> ConvergenceAnalysis:
        """
        Analyze convergence across all timeframes

        Returns:
            ConvergenceAnalysis with convergence metrics
        """
        trend_analyses = {}

        # Analyze each timeframe
        for timeframe in self.timeframes.keys():
            analysis = self.analyze_timeframe_trend(timeframe)
            if analysis:
                trend_analyses[timeframe] = analysis

        if not trend_analyses:
            # Return neutral analysis if no data available
            return ConvergenceAnalysis(
                convergence_score=50.0,
                dominant_trend=TrendDirection.NEUTRAL,
                timeframe_agreement=0.0,
                short_term_bias=TrendDirection.NEUTRAL,
                medium_term_bias=TrendDirection.NEUTRAL
            )

        # Calculate convergence metrics
        convergence_score = self._calculate_convergence_score(trend_analyses)
        dominant_trend = self._determine_dominant_trend(trend_analyses)
        timeframe_agreement = self._calculate_timeframe_agreement(trend_analyses)

        # Determine short-term (1m/5m) and medium-term (5m/15m) bias
        short_term_bias = self._calculate_short_term_bias(trend_analyses)
        medium_term_bias = self._calculate_medium_term_bias(trend_analyses)

        return ConvergenceAnalysis(
            convergence_score=convergence_score,
            dominant_trend=dominant_trend,
            timeframe_agreement=timeframe_agreement,
            short_term_bias=short_term_bias,
            medium_term_bias=medium_term_bias
        )

    def _calculate_trend_direction(self, prices: np.ndarray, rsi: float,
                                 macd_signal: str) -> tuple[TrendDirection, float, float]:
        """
        Calculate trend direction, strength, and momentum

        Args:
            prices: Price array
            rsi: RSI value
            macd_signal: MACD signal

        Returns:
            tuple of (direction, strength, momentum)
        """
        # Price momentum (recent price change)
        if len(prices) >= 5:
            recent_change = (prices[-1] - prices[-5]) / prices[-5] * 100
        else:
            recent_change = 0.0

        # RSI contribution
        rsi_score = 0.0
        if rsi > 70:
            rsi_score = -30.0  # Overbought
        elif rsi < 30:
            rsi_score = 30.0   # Oversold
        elif rsi > 60:
            rsi_score = -10.0  # Bullish
        elif rsi < 40:
            rsi_score = 10.0   # Bearish

        # MACD contribution
        macd_score = 0.0
        if macd_signal == "bullish":
            macd_score = 20.0
        elif macd_signal == "bearish":
            macd_score = -20.0

        # Combined momentum score
        momentum = recent_change + rsi_score + macd_score
        momentum = max(-100.0, min(100.0, momentum))

        # Determine direction and strength
        if momentum > 30:
            direction = TrendDirection.STRONG_BULLISH
            strength = min(100.0, abs(momentum))
        elif momentum > 10:
            direction = TrendDirection.BULLISH
            strength = min(100.0, abs(momentum))
        elif momentum < -30:
            direction = TrendDirection.STRONG_BEARISH
            strength = min(100.0, abs(momentum))
        elif momentum < -10:
            direction = TrendDirection.BEARISH
            strength = min(100.0, abs(momentum))
        else:
            direction = TrendDirection.NEUTRAL
            strength = 50.0

        return direction, strength, momentum

    def _calculate_convergence_score(self, trend_analyses: dict[Timeframe, TrendAnalysis]) -> float:
        """
        Calculate convergence score across timeframes

        Higher score = more converged (aligned) trends
        """
        if len(trend_analyses) < 2:
            return 50.0  # Neutral score for insufficient data

        # Convert directions to numeric values for comparison
        direction_values = {
            TrendDirection.STRONG_BULLISH: 2,
            TrendDirection.BULLISH: 1,
            TrendDirection.NEUTRAL: 0,
            TrendDirection.BEARISH: -1,
            TrendDirection.STRONG_BEARISH: -2
        }

        directions = [direction_values[analysis.direction] for analysis in trend_analyses.values()]

        # Calculate variance (lower variance = higher convergence)
        variance = float(np.var(directions))

        # Convert variance to convergence score (0-100)
        # Max variance for 3 timeframes with values [-2, 2] is ~8
        convergence_score = max(0.0, 100.0 - (variance * 12.5))

        return convergence_score

    def _determine_dominant_trend(self, trend_analyses: dict[Timeframe, TrendAnalysis]) -> TrendDirection:
        """
        Determine the dominant trend across timeframes
        """
        direction_counts: dict[TrendDirection, int] = {}

        for analysis in trend_analyses.values():
            direction_counts[analysis.direction] = direction_counts.get(analysis.direction, 0) + 1

        # Find most common direction
        dominant_direction = max(direction_counts.items(), key=lambda x: x[1])[0]

        return dominant_direction

    def _calculate_timeframe_agreement(self, trend_analyses: dict[Timeframe, TrendAnalysis]) -> float:
        """
        Calculate agreement ratio across timeframes (0-1)
        """
        if not trend_analyses:
            return 0.0

        dominant_trend = self._determine_dominant_trend(trend_analyses)
        agreement_count = sum(1 for analysis in trend_analyses.values()
                            if analysis.direction == dominant_trend)

        return agreement_count / len(trend_analyses)

    def _calculate_short_term_bias(self, trend_analyses: dict[Timeframe, TrendAnalysis]) -> TrendDirection:
        """
        Calculate short-term bias from 1m and 5m timeframes
        """
        short_term_frames = [Timeframe.M1, Timeframe.M5]
        short_term_analyses = {tf: analysis for tf, analysis in trend_analyses.items()
                             if tf in short_term_frames}

        if not short_term_analyses:
            return TrendDirection.NEUTRAL

        return self._determine_dominant_trend(short_term_analyses)

    def _calculate_medium_term_bias(self, trend_analyses: dict[Timeframe, TrendAnalysis]) -> TrendDirection:
        """
        Calculate medium-term bias from 5m and 15m timeframes
        """
        medium_term_frames = [Timeframe.M5, Timeframe.M15]
        medium_term_analyses = {tf: analysis for tf, analysis in trend_analyses.items()
                               if tf in medium_term_frames}

        if not medium_term_analyses:
            return TrendDirection.NEUTRAL

        return self._determine_dominant_trend(medium_term_analyses)

    def get_analysis_summary(self) -> dict[str, Any]:
        """
        Get comprehensive analysis summary

        Returns:
            Dictionary with all analysis results
        """
        convergence = self.analyze_convergence()

        timeframe_analyses = {}
        for timeframe in self.timeframes.keys():
            analysis = self.analyze_timeframe_trend(timeframe)
            if analysis:
                timeframe_analyses[timeframe.value] = {
                    "direction": analysis.direction.value,
                    "strength": analysis.strength,
                    "momentum": analysis.momentum,
                    "rsi": analysis.rsi,
                    "macd_signal": analysis.macd_signal,
                    "bollinger_position": analysis.bollinger_position
                }

        return {
            "convergence": {
                "score": convergence.convergence_score,
                "dominant_trend": convergence.dominant_trend.value,
                "timeframe_agreement": convergence.timeframe_agreement,
                "short_term_bias": convergence.short_term_bias.value,
                "medium_term_bias": convergence.medium_term_bias.value
            },
            "timeframe_analyses": timeframe_analyses,
            "data_points": {
                tf.value: len(data.prices) for tf, data in self.timeframes.items()
            }
        }
