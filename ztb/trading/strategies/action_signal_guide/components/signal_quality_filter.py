"""
Signal Quality Filter for Action Signal Guide.

This component filters and ranks signals based on quality metrics:
- Signal strength and confidence
- Pattern reliability
- Market condition alignment
- Historical performance
- Risk-adjusted quality scores
"""

import time
from collections import defaultdict, deque
from collections.abc import Mapping
from dataclasses import dataclass
from typing import TypedDict

import numpy as np
import pandas as pd

from ztb.trading.signal.common.utilities import (
    calculate_volatility as calculate_volatility_util,
)
from ztb.utils.logging_utils import get_logger


@dataclass
class SignalQualityMetrics:
    """Quality metrics for a trading signal."""

    signal_id: str
    pattern_name: str
    strength: float
    confidence: float
    reliability: float
    market_alignment: float
    risk_adjusted_score: float
    timestamp: float
    metadata: dict[str, object]


class PatternQualityRecord(TypedDict):
    composite_score: float
    strength: float
    confidence: float
    reliability: float
    market_alignment: float
    risk_adjusted_score: float
    timestamp: float


@dataclass
class QualityThresholds:
    """Dynamic quality thresholds."""

    min_strength: float
    min_confidence: float
    min_reliability: float
    min_market_alignment: float
    max_signals_per_bar: int


class SignalQualityFilter:
    """
    Filters and ranks signals based on comprehensive quality metrics.

    This class implements:
    - Multi-dimensional signal quality assessment
    - Dynamic threshold adjustment
    - Risk-adjusted signal ranking
    - Quality-based signal pruning
    """

    def __init__(self, config: Mapping[str, object] | None = None):
        """
        Initialize signal quality filter.

        Args:
            config: Configuration dictionary
        """
        self.logger = get_logger("ztb.trading.strategies.signal_quality_filter")
        self.config: dict[str, object] = dict(config) if config else {}

        # Quality tracking
        self.signal_history: deque[SignalQualityMetrics] = deque(maxlen=5000)
        self.pattern_quality_stats: dict[str, deque[PatternQualityRecord]] = defaultdict(
            lambda: deque(maxlen=1000)
        )

        # Dynamic thresholds
        self.quality_thresholds = QualityThresholds(
            min_strength=self._clamp(
                self._to_float(self.config.get("min_strength"), 0.3), 0.0, 1.0
            ),
            min_confidence=self._clamp(
                self._to_float(self.config.get("min_confidence"), 0.4), 0.0, 1.0
            ),
            min_reliability=self._clamp(
                self._to_float(self.config.get("min_reliability"), 0.35), 0.0, 1.0
            ),
            min_market_alignment=self._clamp(
                self._to_float(self.config.get("min_market_alignment"), 0.3), 0.0, 1.0
            ),
            max_signals_per_bar=max(
                1, self._to_int(self.config.get("max_signals_per_bar"), 3)
            ),
        )

        # Quality weights for composite scoring
        self.quality_weights = {
            "strength": self._to_float(self.config.get("strength_weight"), 0.25),
            "confidence": self._to_float(self.config.get("confidence_weight"), 0.25),
            "reliability": self._to_float(self.config.get("reliability_weight"), 0.2),
            "market_alignment": self._to_float(
                self.config.get("market_alignment_weight"), 0.2
            ),
            "risk_adjustment": self._to_float(
                self.config.get("risk_adjustment_weight"), 0.1
            ),
        }

        # Market condition factors
        self.market_condition_factor = 1.0
        self.volatility_factor = 1.0

        # Adaptation parameters
        self.adaptation_window = max(
            10, self._to_int(self.config.get("adaptation_window"), 100)
        )
        self.quality_decay_factor = self._clamp(
            self._to_float(self.config.get("quality_decay_factor"), 0.98), 0.0, 1.0
        )

        self.logger.info("SignalQualityFilter initialized")

    def filter_signals(
        self, signals: list[object], market_data: pd.DataFrame, market_regime: object
    ) -> list[object]:
        """
        Filter and rank signals based on quality metrics.

        Args:
            signals: List of trading signals
            market_data: Current market data
            market_regime: Current market regime

        Returns:
            Filtered and ranked list of signals
        """
        if not signals:
            return []

        # Assess quality for each signal
        quality_signals: list[tuple[object, SignalQualityMetrics]] = []
        for signal in signals:
            quality_metrics = self._assess_signal_quality(
                signal, market_data, market_regime
            )
            if quality_metrics:
                quality_signals.append((signal, quality_metrics))

        if not quality_signals:
            return []

        # Update market condition factors
        self._update_market_factors(market_data, market_regime)

        # Apply quality filtering
        filtered_signals = self._apply_quality_filter(quality_signals)

        # Rank by composite quality score
        ranked_signals = self._rank_by_quality_score(filtered_signals)

        # Limit signals per bar
        final_signals = self._limit_signals_per_bar(ranked_signals)

        # Update quality statistics
        self._update_quality_statistics(final_signals)

        self.logger.debug(
            f"Filtered {len(signals)} signals to {len(final_signals)} high-quality signals"
        )
        return [signal for signal, _ in final_signals]

    def filter_by_quality(
        self, signals: list[object], market_data: pd.DataFrame
    ) -> list[object]:
        """
        Compatibility wrapper for legacy callers.

        Applies quality filtering without explicit regime context.
        """
        return self.filter_signals(signals, market_data, market_regime=None)

    def _assess_signal_quality(
        self, signal: object, market_data: pd.DataFrame, market_regime: object
    ) -> SignalQualityMetrics | None:
        """
        Assess comprehensive quality metrics for a signal.

        Args:
            signal: Trading signal object
            market_data: Current market data
            market_regime: Current market regime

        Returns:
            SignalQualityMetrics if signal passes basic checks, None otherwise
        """
        try:
            # Basic validation
            strength_raw = getattr(signal, "strength", None)
            confidence_raw = getattr(signal, "confidence", None)
            if strength_raw is None or confidence_raw is None:
                return None
            strength_value = self._to_float(strength_raw, float("nan"))
            confidence_value = self._to_float(confidence_raw, float("nan"))
            if np.isnan(strength_value) or np.isnan(confidence_value):
                return None

            pattern_name = self._resolve_pattern_name(signal)

            # Calculate individual quality components
            strength_score = self._calculate_strength_score(strength_value)
            confidence_score = self._calculate_confidence_score(confidence_value)
            reliability_score = self._calculate_reliability_score(pattern_name)
            market_alignment_score = self._calculate_market_alignment_score(
                signal, market_data, market_regime
            )

            # Calculate risk-adjusted score
            risk_adjusted_score = self._calculate_risk_adjusted_score(
                signal, market_data, strength_score, confidence_score
            )

            # Create quality metrics
            quality_metrics = SignalQualityMetrics(
                signal_id=self._resolve_signal_id(signal, pattern_name),
                pattern_name=pattern_name,
                strength=strength_score,
                confidence=confidence_score,
                reliability=reliability_score,
                market_alignment=market_alignment_score,
                risk_adjusted_score=risk_adjusted_score,
                timestamp=time.time(),
                metadata={
                    "original_strength": strength_value,
                    "original_confidence": confidence_value,
                    "market_regime": str(market_regime) if market_regime else "unknown",
                },
            )

            # DEBUG LOGGING
            self.logger.debug(
                f"DEBUG: Signal Quality - Pattern: {pattern_name}, Strength: {strength_score:.2f}, Confidence: {confidence_score:.2f}, Reliability: {reliability_score:.2f}, Alignment: {market_alignment_score:.2f}"
            )

            return quality_metrics

        except Exception as e:
            self.logger.warning(f"Error assessing signal quality: {e}")
            import traceback

            self.logger.warning(traceback.format_exc())
            return None

    def _calculate_strength_score(self, strength: float) -> float:
        """Calculate normalized strength score."""
        # Normalize to 0-1 scale with emphasis on higher strength
        if strength >= 0.8:
            return 1.0
        elif strength >= 0.6:
            return 0.8 + (strength - 0.6) * 2.5  # 0.8-1.0
        elif strength >= 0.4:
            return 0.5 + (strength - 0.4) * 3.0  # 0.5-0.8
        elif strength >= 0.2:
            return 0.2 + (strength - 0.2) * 1.5  # 0.2-0.5
        else:
            return max(0.1, strength * 2.0)  # 0.1-0.4

    def _calculate_confidence_score(self, confidence: float) -> float:
        """Calculate normalized confidence score."""
        # Confidence is typically already 0-1, but ensure proper scaling
        return max(0.0, min(1.0, confidence))

    def _calculate_reliability_score(self, pattern_name: str) -> float:
        """Calculate pattern reliability based on historical performance."""
        if pattern_name not in self.pattern_quality_stats:
            return 0.5  # Neutral score for new patterns

        history = list(self.pattern_quality_stats[pattern_name])
        if not history:
            return 0.5

        # Calculate reliability as average quality score with recency weighting
        recent_history = history[-min(20, len(history)) :]  # Last 20 signals

        if not recent_history:
            return 0.5

        # Weight recent signals more heavily
        weights = np.linspace(0.5, 1.0, len(recent_history))
        quality_scores = [h["composite_score"] for h in recent_history]

        weighted_avg = float(np.average(quality_scores, weights=weights))
        return max(0.1, min(1.0, weighted_avg))

    def _calculate_market_alignment_score(
        self, signal: object, market_data: pd.DataFrame, market_regime: object
    ) -> float:
        """Calculate how well signal aligns with current market conditions."""
        if not market_regime or len(market_data) < 10:
            return 0.5

        alignment_score = 0.5  # Base score

        try:
            # Get recent market data
            recent_data = market_data.tail(20)

            # Calculate trend alignment
            regime_bucket = self._regime_bucket(market_regime)

            # Trend regime alignment
            if regime_bucket == "TRENDING":
                # In trending markets, prefer directional signals
                if hasattr(signal, "direction"):
                    direction = getattr(signal, "direction", 0)
                    recent_trend = self._calculate_recent_trend(recent_data)
                    if (direction > 0 and recent_trend > 0) or (
                        direction < 0 and recent_trend < 0
                    ):
                        alignment_score += 0.2
                    elif (direction > 0 and recent_trend < 0) or (
                        direction < 0 and recent_trend > 0
                    ):
                        alignment_score -= 0.2

            # Ranging regime alignment
            elif regime_bucket == "RANGING":
                # In ranging markets, prefer mean-reversion signals
                pattern_type = getattr(
                    signal, "pattern_type", getattr(signal, "signal_type", "")
                )
                pattern_type_str = str(pattern_type)
                if (
                    "oscillator" in pattern_type_str.lower()
                    or "rsi" in pattern_type_str.lower()
                    or "stochastic" in pattern_type_str.lower()
                    or "cci" in pattern_type_str.lower()
                    or "williams" in pattern_type_str.lower()
                    or "mfi" in pattern_type_str.lower()
                    or "bollinger" in pattern_type_str.lower()
                    or "reversal" in pattern_type_str.lower()
                ):
                    alignment_score += 0.15

            # Volatility alignment
            volatility = self._calculate_volatility(recent_data)
            if volatility > 0.05:  # High volatility
                # Prefer stronger signals in volatile markets
                if hasattr(signal, "strength") and signal.strength > 0.7:
                    alignment_score += 0.1
            else:  # Low volatility
                # Can accept weaker signals in stable markets
                alignment_score += 0.05

        except Exception as e:
            self.logger.debug(f"Error calculating market alignment: {e}")

        return max(0.0, min(1.0, alignment_score))

    def _calculate_risk_adjusted_score(
        self,
        _signal: object,
        market_data: pd.DataFrame,
        strength_score: float,
        confidence_score: float,
    ) -> float:
        """Calculate risk-adjusted quality score."""
        try:
            # Get recent volatility as risk measure
            recent_data = market_data.tail(20)
            volatility = self._calculate_volatility(recent_data)

            # Risk adjustment factor (higher volatility = higher risk = lower score)
            risk_factor = max(0.5, 1.0 - volatility * 2.0)

            # Base score from strength and confidence
            base_score = (strength_score + confidence_score) / 2.0

            # Apply risk adjustment
            risk_adjusted = base_score * risk_factor

            # Apply market condition factor
            final_score = risk_adjusted * self.market_condition_factor

            return max(0.0, min(1.0, final_score))

        except Exception as e:
            self.logger.debug(f"Error calculating risk-adjusted score: {e}")
            return (strength_score + confidence_score) / 2.0

    def _calculate_recent_trend(self, data: pd.DataFrame) -> float:
        """Calculate recent price trend."""
        if len(data) < 5:
            return 0.0

        prices = data["close"].values
        recent_prices = prices[-min(10, len(prices)) :]

        # Simple linear trend
        x = np.arange(len(recent_prices))
        slope, _ = np.polyfit(x, recent_prices, 1)

        # Normalize by average price
        avg_price = np.mean(recent_prices)
        normalized_slope = slope / avg_price if avg_price > 0 else 0

        return normalized_slope

    def _calculate_volatility(self, data: pd.DataFrame) -> float:
        """Calculate price volatility."""
        # Preserve default behavior for very short input
        if len(data) < 5:
            return 0.02  # Default moderate volatility

        returns = data["close"].pct_change().dropna()
        if len(returns) < 3:
            return 0.02

        try:
            vol = calculate_volatility_util(
                returns, window=min(20, len(returns)), method="std"
            )
            return float(vol)
        except Exception:
            # Fallback to simple std() if central helper raises
            return float(returns.std())

    def _apply_quality_filter(
        self, quality_signals: list[tuple[object, SignalQualityMetrics]]
    ) -> list[tuple[object, SignalQualityMetrics]]:
        """Apply quality-based filtering to signals."""
        filtered = []

        for signal, metrics in quality_signals:
            # Check individual thresholds
            dropped_reason = None
            if metrics.strength < self.quality_thresholds.min_strength:
                dropped_reason = f"Strength {metrics.strength:.2f} < {self.quality_thresholds.min_strength}"
            elif metrics.confidence < self.quality_thresholds.min_confidence:
                dropped_reason = f"Confidence {metrics.confidence:.2f} < {self.quality_thresholds.min_confidence}"
            elif metrics.reliability < self.quality_thresholds.min_reliability:
                dropped_reason = f"Reliability {metrics.reliability:.2f} < {self.quality_thresholds.min_reliability}"
            elif (
                metrics.market_alignment < self.quality_thresholds.min_market_alignment
            ):
                dropped_reason = f"Alignment {metrics.market_alignment:.2f} < {self.quality_thresholds.min_market_alignment}"

            if dropped_reason:
                self.logger.debug(
                    f"DEBUG: Dropped signal {metrics.pattern_name}: {dropped_reason}"
                )
            else:
                filtered.append((signal, metrics))

        return filtered

    def _rank_by_quality_score(
        self, quality_signals: list[tuple[object, SignalQualityMetrics]]
    ) -> list[tuple[object, SignalQualityMetrics]]:
        """Rank signals by composite quality score."""

        # Sort by composite score descending
        ranked = sorted(
            quality_signals,
            key=lambda x: self._calculate_composite_score(x[1]),
            reverse=True,
        )

        return ranked

    def _limit_signals_per_bar(
        self, ranked_signals: list[tuple[object, SignalQualityMetrics]]
    ) -> list[tuple[object, SignalQualityMetrics]]:
        """Limit the number of signals per bar/timestamp."""
        if len(ranked_signals) <= self.quality_thresholds.max_signals_per_bar:
            return ranked_signals

        # Take top N signals
        return ranked_signals[: self.quality_thresholds.max_signals_per_bar]

    def _update_market_factors(self, market_data: pd.DataFrame, market_regime: object):
        """Update market condition factors for quality assessment."""
        try:
            # Update volatility factor
            volatility = self._calculate_volatility(market_data.tail(20))
            # Higher volatility = more lenient quality thresholds
            self.volatility_factor = max(0.8, min(1.2, 1.0 + volatility))

            # Update market condition factor based on regime
            if market_regime:
                regime_bucket = self._regime_bucket(market_regime)
                if regime_bucket == "VOLATILE":
                    self.market_condition_factor = 0.9  # Stricter in volatile markets
                elif regime_bucket == "TRENDING":
                    self.market_condition_factor = 1.0  # Normal in trending markets
                elif regime_bucket == "RANGING":
                    self.market_condition_factor = (
                        1.1  # More lenient in ranging markets
                    )
                else:
                    self.market_condition_factor = 1.0

        except Exception as e:
            self.logger.debug(f"Error updating market factors: {e}")

    def _update_quality_statistics(
        self, final_signals: list[tuple[object, SignalQualityMetrics]]
    ) -> None:
        """Update quality statistics for pattern performance tracking."""
        for _signal, metrics in final_signals:
            composite_score = self._calculate_composite_score(metrics)

            # Store in pattern history
            quality_record: PatternQualityRecord = {
                "composite_score": composite_score,
                "strength": metrics.strength,
                "confidence": metrics.confidence,
                "reliability": metrics.reliability,
                "market_alignment": metrics.market_alignment,
                "risk_adjusted_score": metrics.risk_adjusted_score,
                "timestamp": metrics.timestamp,
            }

            self.pattern_quality_stats[metrics.pattern_name].append(quality_record)

            # Store in general signal history
            self.signal_history.append(metrics)

    def adapt_thresholds(self) -> None:
        """Adapt quality thresholds based on recent performance."""
        if len(self.signal_history) < self.adaptation_window:
            return

        recent_signals = list(self.signal_history)[-self.adaptation_window :]

        # Calculate statistics from recent signals
        strengths = [s.strength for s in recent_signals]
        confidences = [s.confidence for s in recent_signals]
        reliabilities = [s.reliability for s in recent_signals]
        market_alignments = [s.market_alignment for s in recent_signals]

        if not strengths:
            return

        # Update thresholds based on percentile performance
        try:
            self.quality_thresholds.min_strength = max(
                0.2, float(np.percentile(strengths, 25))
            )  # 25th percentile as minimum
            self.quality_thresholds.min_confidence = max(
                0.3, float(np.percentile(confidences, 25))
            )
            self.quality_thresholds.min_reliability = max(
                0.25, float(np.percentile(reliabilities, 25))
            )
            self.quality_thresholds.min_market_alignment = max(
                0.2, float(np.percentile(market_alignments, 25))
            )

            # Adjust max signals based on signal density
            signal_density = len(recent_signals) / self.adaptation_window
            if signal_density > 0.1:  # High signal density
                self.quality_thresholds.max_signals_per_bar = max(
                    2, self.quality_thresholds.max_signals_per_bar - 1
                )
            elif signal_density < 0.02:  # Low signal density
                self.quality_thresholds.max_signals_per_bar = min(
                    5, self.quality_thresholds.max_signals_per_bar + 1
                )

            self.logger.debug(
                f"Adapted quality thresholds: strength={self.quality_thresholds.min_strength:.2f}, "
                f"confidence={self.quality_thresholds.min_confidence:.2f}"
            )

        except Exception as e:
            self.logger.warning(f"Error adapting thresholds: {e}")

    def update_thresholds(
        self,
        market_data: pd.DataFrame,
        recent_performance: Mapping[str, float] | None = None,
    ) -> None:
        """
        Compatibility wrapper for older adaptive pipeline.

        Args:
            market_data: Latest market data used for context update.
            recent_performance: Optional recent performance map by pattern.
        """
        self._update_market_factors(market_data, market_regime=None)
        self.adapt_thresholds()

        if recent_performance:
            avg_perf = float(np.mean(list(recent_performance.values())))
            if avg_perf < 0.3:
                self.quality_thresholds.min_strength = max(
                    0.2, self.quality_thresholds.min_strength - 0.05
                )
                self.quality_thresholds.min_confidence = max(
                    0.3, self.quality_thresholds.min_confidence - 0.05
                )
            elif avg_perf > 0.7:
                self.quality_thresholds.min_strength = min(
                    0.8, self.quality_thresholds.min_strength + 0.03
                )
                self.quality_thresholds.min_confidence = min(
                    0.9, self.quality_thresholds.min_confidence + 0.03
                )

    def get_quality_statistics(self) -> dict[str, object]:
        """Get comprehensive quality statistics."""
        pattern_quality_summary: dict[str, dict[str, float | int]] = {}
        stats: dict[str, object] = {
            "total_signals_processed": len(self.signal_history),
            "current_thresholds": {
                "min_strength": self.quality_thresholds.min_strength,
                "min_confidence": self.quality_thresholds.min_confidence,
                "min_reliability": self.quality_thresholds.min_reliability,
                "min_market_alignment": self.quality_thresholds.min_market_alignment,
                "max_signals_per_bar": self.quality_thresholds.max_signals_per_bar,
            },
            "pattern_quality_summary": pattern_quality_summary,
            "market_factors": {
                "market_condition_factor": self.market_condition_factor,
                "volatility_factor": self.volatility_factor,
            },
        }

        # Pattern-specific statistics
        for pattern_name, history in self.pattern_quality_stats.items():
            if history:
                recent_history = list(history)[-min(50, len(history)) :]
                pattern_quality_summary[pattern_name] = {
                    "avg_composite_score": float(
                        np.mean([h["composite_score"] for h in recent_history])
                    ),
                    "avg_strength": float(np.mean([h["strength"] for h in recent_history])),
                    "avg_confidence": float(
                        np.mean([h["confidence"] for h in recent_history])
                    ),
                    "signal_count": len(recent_history),
                }

        return stats

    def _calculate_composite_score(self, metrics: SignalQualityMetrics) -> float:
        """Calculate weighted composite quality score."""
        return (
            self.quality_weights["strength"] * metrics.strength
            + self.quality_weights["confidence"] * metrics.confidence
            + self.quality_weights["reliability"] * metrics.reliability
            + self.quality_weights["market_alignment"] * metrics.market_alignment
            + self.quality_weights["risk_adjustment"] * metrics.risk_adjusted_score
        )

    @staticmethod
    def _resolve_pattern_name(signal: object) -> str:
        pattern_name = getattr(signal, "pattern_name", None)
        if isinstance(pattern_name, str) and pattern_name:
            return pattern_name

        source_patterns = getattr(signal, "source_patterns", None)
        if (
            isinstance(source_patterns, (list, tuple))
            and len(source_patterns) > 0
            and isinstance(source_patterns[0], str)
        ):
            return source_patterns[0]

        signal_type = getattr(signal, "signal_type", None)
        if isinstance(signal_type, str) and signal_type:
            return signal_type

        return "unknown"

    @staticmethod
    def _resolve_signal_id(signal: object, pattern_name: str) -> str:
        signal_id = getattr(signal, "signal_id", None)
        if isinstance(signal_id, str) and signal_id:
            return signal_id
        return f"{pattern_name}_{time.time()}"

    @staticmethod
    def _to_float(value: object, default: float) -> float:
        try:
            return float(value)
        except (TypeError, ValueError):
            return default

    @staticmethod
    def _to_int(value: object, default: int) -> int:
        try:
            return int(value)
        except (TypeError, ValueError):
            return default

    @staticmethod
    def _clamp(value: float, min_value: float, max_value: float) -> float:
        return max(min_value, min(max_value, value))

    @staticmethod
    def _regime_bucket(market_regime: object) -> str:
        """Normalize various regime enum/string representations into coarse buckets."""
        regime_name = getattr(market_regime, "name", str(market_regime)).upper()
        if any(key in regime_name for key in ["VOLATILE", "VOLATILITY", "EXTREME"]):
            return "VOLATILE"
        if any(
            key in regime_name for key in ["TREND", "BULL", "BEAR", "BREAKOUT", "BREAKDOWN"]
        ):
            return "TRENDING"
        if any(key in regime_name for key in ["RANG", "SIDEWAYS", "CONSOLIDATION"]):
            return "RANGING"
        return "UNKNOWN"


class SignalQualityEvaluator:
    """Compatibility evaluator used by advanced SignalGenerator pipelines."""

    def __init__(self, config: Mapping[str, object] | None = None) -> None:
        cfg = dict(config) if config else {}
        self.weights = {
            "strength": float(cfg.get("strength_weight", 0.35)),
            "confidence": float(cfg.get("confidence_weight", 0.35)),
            "regime_alignment": float(cfg.get("regime_alignment_weight", 0.2)),
            "sac_alignment": float(cfg.get("sac_alignment_weight", 0.1)),
        }

    def evaluate_signal_quality(
        self,
        signal: object,
        market_data: pd.DataFrame,
        sac_decision: object = None,
        market_regime: object = None,
    ) -> dict[str, float]:
        """Evaluate signal quality components for ranking/filtering."""
        strength = self._clamp(self._to_float(getattr(signal, "strength", 0.0)), 0.0, 1.0)
        confidence = self._clamp(
            self._to_float(getattr(signal, "confidence", 0.0)), 0.0, 1.0
        )
        regime_alignment = self._calculate_regime_alignment(signal, market_regime)
        sac_alignment = self._calculate_sac_alignment(signal, sac_decision)

        volatility = 0.0
        if not market_data.empty and "close" in market_data.columns:
            returns = market_data["close"].pct_change().dropna()
            if not returns.empty:
                volatility = self._clamp(float(returns.std()), 0.0, 1.0)

        return {
            "strength": strength,
            "confidence": confidence,
            "regime_alignment": regime_alignment,
            "sac_alignment": sac_alignment,
            "volatility_penalty": self._clamp(volatility, 0.0, 0.3),
        }

    def get_overall_quality_score(self, quality_scores: Mapping[str, float]) -> float:
        """Aggregate component scores into a single quality score."""
        score = (
            self.weights["strength"] * quality_scores.get("strength", 0.0)
            + self.weights["confidence"] * quality_scores.get("confidence", 0.0)
            + self.weights["regime_alignment"] * quality_scores.get("regime_alignment", 0.0)
            + self.weights["sac_alignment"] * quality_scores.get("sac_alignment", 0.0)
        )
        score -= quality_scores.get("volatility_penalty", 0.0)
        return self._clamp(float(score), 0.0, 1.0)

    def _calculate_regime_alignment(self, signal: object, market_regime: object) -> float:
        if market_regime is None:
            return 0.5
        direction = self._to_float(getattr(signal, "direction", 0.0))
        regime_bucket = SignalQualityFilter._regime_bucket(market_regime)
        if regime_bucket == "TRENDING":
            return 0.8 if abs(direction) > 0.1 else 0.4
        if regime_bucket == "RANGING":
            return 0.7 if abs(direction) <= 0.5 else 0.5
        if regime_bucket == "VOLATILE":
            return 0.7 if abs(direction) > 0.3 else 0.45
        return 0.5

    def _calculate_sac_alignment(self, signal: object, sac_decision: object) -> float:
        if sac_decision is None:
            return 0.5
        signal_direction = self._to_float(getattr(signal, "direction", 0.0))
        sac_direction = self._to_float(getattr(sac_decision, "direction", 0.0))
        if sac_direction == 0.0:
            return 0.5
        return 0.8 if signal_direction * sac_direction > 0 else 0.2

    @staticmethod
    def _to_float(value: object, default: float = 0.0) -> float:
        try:
            return float(value)
        except (TypeError, ValueError):
            return default

    @staticmethod
    def _clamp(value: float, min_value: float = 0.0, max_value: float = 1.0) -> float:
        return max(min_value, min(max_value, value))
