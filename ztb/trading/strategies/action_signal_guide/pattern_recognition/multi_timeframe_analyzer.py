"""
MultiTimeframeAnalyzer Component.

Enhanced multi-timeframe analysis for Action Signal Guide.
Provides hierarchical time axis analysis and alignment scoring.
"""

from typing import Dict, List

import numpy as np
import pandas as pd

from ztb.trading.signal.common.utilities import calculate_volatility_from_prices
from ztb.utils.logging_utils import get_logger

from ..types import AnalysisResult, MultiTimeframeAnalysis, MultiTimeframeData


class MultiTimeframeAnalyzer:
    """
    Advanced multi-timeframe analysis system.

    Provides:
    - Hierarchical time axis analysis
    - Alignment scoring across timeframes
    - Cross-timeframe pattern validation
    - Adaptive timeframe selection
    """

    def __init__(self):
        self.logger = get_logger("ztb.trading.strategies.multi_timeframe_analyzer")

        # Timeframe hierarchy (from short to long term)
        self.timeframe_hierarchy = ["1m", "5m", "15m", "1h", "4h", "1d"]

        # Timeframe weights for alignment scoring
        self.timeframe_weights = {
            "1m": 0.1,  # Short-term noise
            "5m": 0.2,  # Very short-term trends
            "15m": 0.3,  # Short-term trends
            "1h": 0.25,  # Medium-term trends
            "4h": 0.1,  # Long-term trends
            "1d": 0.05,  # Very long-term trends
        }

        # Alignment scoring parameters
        self.alignment_threshold = 0.7
        self.consistency_bonus = 1.2
        self.conflict_penalty = 0.8

    def analyze_multi_timeframe_alignment(
        self, multi_timeframe_data: MultiTimeframeData, primary_timeframe: str = "15m"
    ) -> MultiTimeframeAnalysis:
        """
        Analyze alignment across multiple timeframes.

        Args:
            multi_timeframe_data: Data for different timeframes
            primary_timeframe: Primary timeframe for analysis

        Returns:
            Analysis results for each timeframe
        """
        analysis_results = {}

        for tf, tf_data in multi_timeframe_data.items():
            if "data" not in tf_data:
                continue

            df = tf_data["data"]
            if df.empty or len(df) < 10:
                continue

            # Analyze timeframe-specific patterns
            analysis = self._analyze_single_timeframe(df, tf)

            # Calculate alignment with primary timeframe
            if tf != primary_timeframe and primary_timeframe in multi_timeframe_data:
                alignment_score = self._calculate_alignment_score(
                    df,
                    multi_timeframe_data[primary_timeframe]["data"],
                    tf,
                    primary_timeframe,
                )
                analysis["alignment_score"] = alignment_score
                analysis["is_aligned"] = alignment_score > self.alignment_threshold
            else:
                analysis[
                    "alignment_score"
                ] = 1.0  # Primary timeframe aligns with itself
                analysis["is_aligned"] = True

            analysis_results[tf] = analysis

        return analysis_results

    def calculate_overall_alignment_score(
        self, analysis_results: MultiTimeframeAnalysis
    ) -> float:
        """
        Calculate overall alignment score across all timeframes.

        Args:
            analysis_results: Individual timeframe analyses

        Returns:
            Overall alignment score (0-1)
        """
        if not analysis_results:
            return 0.0

        weighted_scores = []
        total_weight = 0.0

        for tf, analysis in analysis_results.items():
            weight = self.timeframe_weights.get(tf, 0.1)
            alignment = analysis.get("alignment_score", 0.0)

            weighted_scores.append(alignment * weight)
            total_weight += weight

        if total_weight == 0:
            return 0.0

        return sum(weighted_scores) / total_weight

    def get_hierarchical_signal_strength(
        self, analysis_results: MultiTimeframeAnalysis, signal_type: str = "trend"
    ) -> float:
        """
        Calculate hierarchical signal strength based on timeframe consensus.

        Args:
            analysis_results: Individual timeframe analyses
            signal_type: Type of signal to analyze

        Returns:
            Hierarchical signal strength (0-1)
        """
        if not analysis_results:
            return 0.0

        # Group timeframes by hierarchy level
        short_term = ["1m", "5m"]
        medium_term = ["15m", "1h"]
        long_term = ["4h", "1d"]

        # Calculate consensus for each level
        short_consensus = self._calculate_level_consensus(
            analysis_results, short_term, signal_type
        )
        medium_consensus = self._calculate_level_consensus(
            analysis_results, medium_term, signal_type
        )
        long_consensus = self._calculate_level_consensus(
            analysis_results, long_term, signal_type
        )

        # Hierarchical weighting: long-term has higher weight
        hierarchical_strength = (
            short_consensus * 0.2 + medium_consensus * 0.3 + long_consensus * 0.5
        )

        return min(hierarchical_strength, 1.0)

    def _analyze_single_timeframe(
        self, df: pd.DataFrame, timeframe: str
    ) -> AnalysisResult:
        """Analyze patterns within a single timeframe."""
        analysis = {
            "timeframe": timeframe,
            "trend_direction": self._calculate_trend_direction(df),
            "trend_strength": self._calculate_trend_strength(df),
            "volatility": self._calculate_volatility(df),
            "momentum": self._calculate_momentum(df),
            "support_resistance": self._identify_support_resistance(df),
            "pattern_signals": self._detect_basic_patterns(df),
        }

        return analysis

    def _calculate_trend_direction(self, df: pd.DataFrame) -> int:
        """Calculate trend direction (-1, 0, 1)."""
        if "close" not in df.columns or len(df) < 5:
            return 0

        closes = df["close"].values
        recent = np.mean(closes[-5:])
        older = np.mean(closes[:-5]) if len(closes) > 5 else recent

        if recent > older * 1.001:
            return 1
        elif recent < older * 0.999:
            return -1
        else:
            return 0

    def _calculate_trend_strength(self, df: pd.DataFrame) -> float:
        """Calculate trend strength (0-1)."""
        if len(df) < 10:
            return 0.0

        # Use linear regression slope as trend strength
        closes = df["close"].values
        x = np.arange(len(closes))
        slope, _ = np.polyfit(x, closes, 1)

        # Normalize slope by average price
        avg_price = np.mean(closes)
        normalized_slope = slope / avg_price

        # Convert to 0-1 scale
        strength = min(abs(normalized_slope) * 100, 1.0)
        return strength

    def _calculate_volatility(self, df: pd.DataFrame) -> float:
        """Calculate volatility measure."""
        if "close" not in df.columns or len(df) < 5:
            return 0.0

        # Use central utility to compute volatility from price series and annualize to match prior behavior
        if "close" not in df.columns:
            return 0.0
        try:
            return float(
                calculate_volatility_from_prices(
                    df["close"], window=min(20, len(df)), annualize=True
                )
            )
        except Exception:
            # Fallback to previous implementation if anything goes wrong
            from ztb.features.generators.technical.volatility.return_std import (
                compute_return_stddev,
            )
            from ztb.trading.constants import TRADING_DAYS_PER_YEAR

            vol_series = compute_return_stddev(df, period=len(df))
            last_val = vol_series.iloc[-1]
            val = float(last_val) if not pd.isna(last_val) else 0.0
            return val * np.sqrt(TRADING_DAYS_PER_YEAR)

    def _calculate_momentum(self, df: pd.DataFrame) -> float:
        """Calculate momentum indicator."""
        if "close" not in df.columns or len(df) < 14:
            return 0.0

        # Simple momentum: rate of change
        recent = df["close"].iloc[-1]
        past = df["close"].iloc[-14]

        if past == 0:
            return 0.0

        momentum = (recent - past) / past
        return momentum

    def _identify_support_resistance(self, df: pd.DataFrame) -> Dict[str, float]:
        """Identify support and resistance levels."""
        if "high" not in df.columns or "low" not in df.columns:
            return {"support": 0.0, "resistance": 0.0}

        # Simple identification: recent highs/lows
        recent_high = df["high"].tail(20).max()
        recent_low = df["low"].tail(20).min()

        return {"support": recent_low, "resistance": recent_high}

    def _detect_basic_patterns(self, df: pd.DataFrame) -> List[str]:
        """Detect basic price patterns."""
        patterns = []

        if len(df) < 5:
            return patterns

        closes = df["close"].values

        # Simple pattern detection
        if self._is_higher_highs(closes):
            patterns.append("higher_highs")
        if self._is_lower_lows(closes):
            patterns.append("lower_lows")
        if self._is_consolidation(closes):
            patterns.append("consolidation")

        return patterns

    def _calculate_alignment_score(
        self, df1: pd.DataFrame, df2: pd.DataFrame, tf1: str, tf2: str
    ) -> float:
        """Calculate alignment score between two timeframes."""
        trend1 = self._calculate_trend_direction(df1)
        trend2 = self._calculate_trend_direction(df2)

        momentum1 = self._calculate_momentum(df1)
        momentum2 = self._calculate_momentum(df2)

        # Trend alignment
        trend_alignment = 1.0 if trend1 == trend2 else 0.0

        # Momentum alignment (within tolerance)
        momentum_diff = abs(momentum1 - momentum2)
        momentum_alignment = max(0.0, 1.0 - momentum_diff * 10)  # 10% tolerance

        # Weighted alignment score
        alignment = trend_alignment * 0.7 + momentum_alignment * 0.3

        return alignment

    def _calculate_level_consensus(
        self,
        analysis_results: MultiTimeframeAnalysis,
        timeframes: List[str],
        signal_type: str,
    ) -> float:
        """Calculate consensus within a timeframe level."""
        relevant_results = [
            analysis_results[tf] for tf in timeframes if tf in analysis_results
        ]

        if not relevant_results:
            return 0.0

        if signal_type == "trend":
            directions = [r.get("trend_direction", 0) for r in relevant_results]
            strengths = [r.get("trend_strength", 0.0) for r in relevant_results]

            # Consensus: majority direction with average strength
            if directions:
                majority_direction = 1 if sum(directions) > 0 else -1
                direction_consensus = sum(
                    1 for d in directions if d == majority_direction
                ) / len(directions)
                avg_strength = np.mean(strengths)

                return direction_consensus * avg_strength

        return 0.0

    def _is_higher_highs(self, closes: np.ndarray) -> bool:
        """Check for higher highs pattern."""
        if len(closes) < 4:
            return False
        return closes[-1] > closes[-3] and closes[-2] > closes[-4]

    def _is_lower_lows(self, closes: np.ndarray) -> bool:
        """Check for lower lows pattern."""
        if len(closes) < 4:
            return False
        return closes[-1] < closes[-3] and closes[-2] < closes[-4]

    def _is_consolidation(self, closes: np.ndarray) -> bool:
        """Check for consolidation pattern."""
        if len(closes) < 10:
            return False
        recent_range = closes[-10:].max() - closes[-10:].min()
        avg_price = closes[-10:].mean()
        return recent_range / avg_price < 0.02  # Less than 2% range
