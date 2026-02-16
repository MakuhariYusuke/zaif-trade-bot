"""
MultiTimeframeAnalyzer Component.

Enhanced multi-timeframe analysis for Action Signal Guide.
Provides hierarchical time axis analysis and alignment scoring.
"""

from collections import OrderedDict

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

    def __init__(self) -> None:
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
        self._regression_cache: OrderedDict[int, tuple[np.ndarray, float]] = OrderedDict()
        self._max_regression_cache_entries = 64

    @staticmethod
    def _extract_timeframe_data(payload: object) -> pd.DataFrame | None:
        if not isinstance(payload, dict):
            return None
        tf_df = payload.get("data")
        return tf_df if isinstance(tf_df, pd.DataFrame) else None

    def _get_regression_weights(self, length: int) -> tuple[np.ndarray, float]:
        cached = self._regression_cache.get(length)
        if cached is not None:
            self._regression_cache.move_to_end(length)
            return cached

        x = np.arange(length, dtype=np.float64)
        centered_x = x - float(np.mean(x))
        denominator = float(np.dot(centered_x, centered_x))
        if denominator <= 0:
            denominator = 1.0

        weights = (centered_x, denominator)
        self._regression_cache[length] = weights
        if len(self._regression_cache) > self._max_regression_cache_entries:
            self._regression_cache.popitem(last=False)
        return weights

    def _calculate_normalized_slope(self, closes: np.ndarray) -> float:
        """Compute normalized slope without repeated `np.polyfit` allocations."""
        if closes.size < 2:
            return 0.0

        values = np.asarray(closes, dtype=np.float64)
        mean_price = float(np.mean(values))
        if mean_price == 0.0:
            return 0.0

        centered_x, denominator = self._get_regression_weights(values.size)
        centered_y = values - float(np.mean(values))
        slope = float(np.dot(centered_x, centered_y) / denominator)
        return float(slope / mean_price)

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
        analysis_results: MultiTimeframeAnalysis = {}
        primary_payload = (
            multi_timeframe_data.get(primary_timeframe)
            if isinstance(multi_timeframe_data, dict)
            else None
        )
        primary_df = self._extract_timeframe_data(primary_payload)

        for tf, tf_data in multi_timeframe_data.items():
            df = self._extract_timeframe_data(tf_data)
            if df is None or df.empty or len(df) < 10:
                continue

            # Analyze timeframe-specific patterns
            analysis = self._analyze_single_timeframe(df, tf)

            # Calculate alignment with primary timeframe
            if (
                tf != primary_timeframe
                and primary_df is not None
                and not primary_df.empty
                and len(primary_df) >= 10
            ):
                alignment_score = self._calculate_alignment_score(
                    df,
                    primary_df,
                    tf,
                    primary_timeframe,
                )
                analysis["alignment_score"] = alignment_score
                analysis["is_aligned"] = alignment_score > self.alignment_threshold
            else:
                analysis["alignment_score"] = 1.0
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

        weighted_scores: list[float] = []
        total_weight = 0.0

        for tf, analysis in analysis_results.items():
            weight = self.timeframe_weights.get(tf, 0.1)
            alignment_raw = analysis.get("alignment_score", 0.0)
            alignment = (
                float(alignment_raw)
                if isinstance(alignment_raw, (int, float))
                else 0.0
            )

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

        closes = df["close"].to_numpy(dtype=np.float64, copy=False)
        recent_window = 5 if len(closes) >= 10 else max(2, len(closes) // 2)
        if recent_window <= 0 or len(closes) <= recent_window:
            return 0

        recent = float(np.mean(closes[-recent_window:]))
        older_slice = closes[-2 * recent_window : -recent_window]
        if older_slice.size == 0:
            older_slice = closes[:-recent_window]
        if older_slice.size == 0:
            return 0
        older = float(np.mean(older_slice))
        if older == 0.0:
            return 0

        relative_change = (recent - older) / abs(older)
        if relative_change > 0.001:
            return 1
        if relative_change < -0.001:
            return -1
        return 0

    def _calculate_trend_strength(self, df: pd.DataFrame) -> float:
        """Calculate trend strength (0-1)."""
        if len(df) < 10 or "close" not in df.columns:
            return 0.0

        closes = df["close"].to_numpy(dtype=np.float64, copy=False)
        normalized_slope = self._calculate_normalized_slope(closes)

        # Convert to 0-1 scale
        return float(min(abs(normalized_slope) * 100.0, 1.0))

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
        recent = float(df["close"].iloc[-1])
        past = float(df["close"].iloc[-14])

        if past == 0:
            return 0.0

        momentum = (recent - past) / past
        return float(momentum)

    def _identify_support_resistance(self, df: pd.DataFrame) -> dict[str, float]:
        """Identify support and resistance levels."""
        if "high" not in df.columns or "low" not in df.columns:
            return {"support": 0.0, "resistance": 0.0}

        # Simple identification: recent highs/lows
        recent_high = float(df["high"].tail(20).max())
        recent_low = float(df["low"].tail(20).min())

        return {"support": recent_low, "resistance": recent_high}

    def _detect_basic_patterns(self, df: pd.DataFrame) -> list[str]:
        """Detect basic price patterns."""
        patterns: list[str] = []

        if len(df) < 5:
            return patterns

        closes = df["close"].to_numpy(dtype=np.float64, copy=False)

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
        _ = (tf1, tf2)
        if "close" not in df1.columns or "close" not in df2.columns:
            return 0.0

        trend1 = self._calculate_trend_direction(df1)
        trend2 = self._calculate_trend_direction(df2)

        momentum1 = self._calculate_momentum(df1)
        momentum2 = self._calculate_momentum(df2)

        # Trend alignment
        if trend1 == 0 or trend2 == 0:
            trend_alignment = 0.5
        else:
            trend_alignment = 1.0 if trend1 == trend2 else 0.0

        # Momentum alignment (within tolerance)
        momentum_diff = abs(momentum1 - momentum2)
        momentum_alignment = max(0.0, 1.0 - momentum_diff * 10)  # 10% tolerance

        # Weighted alignment score
        alignment = trend_alignment * 0.7 + momentum_alignment * 0.3

        return float(max(0.0, min(1.0, alignment)))

    def _calculate_level_consensus(
        self,
        analysis_results: MultiTimeframeAnalysis,
        timeframes: list[str],
        signal_type: str,
    ) -> float:
        """Calculate consensus within a timeframe level."""
        relevant_results = [
            analysis_results[tf] for tf in timeframes if tf in analysis_results
        ]

        if not relevant_results:
            return 0.0

        if signal_type == "trend":
            directions: list[int] = []
            strengths: list[float] = []
            for result in relevant_results:
                direction_raw = result.get("trend_direction", 0)
                strength_raw = result.get("trend_strength", 0.0)
                directions.append(int(direction_raw) if isinstance(direction_raw, (int, float)) else 0)
                strengths.append(float(strength_raw) if isinstance(strength_raw, (int, float)) else 0.0)

            # Consensus: majority direction with average strength
            if directions:
                direction_sum = sum(directions)
                if direction_sum > 0:
                    majority_direction = 1
                elif direction_sum < 0:
                    majority_direction = -1
                else:
                    majority_direction = 0

                direction_consensus = sum(1 for d in directions if d == majority_direction) / len(directions)
                avg_strength = float(np.mean(strengths)) if strengths else 0.0

                return float(direction_consensus * max(0.0, min(1.0, avg_strength)))

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
        if avg_price == 0:
            return False
        return bool((recent_range / avg_price) < 0.02)  # Less than 2% range
