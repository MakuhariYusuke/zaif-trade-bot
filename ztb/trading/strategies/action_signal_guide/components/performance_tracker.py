"""
PerformanceTracker Component.

This component is responsible for tracking performance metrics.
Follows Single Responsibility Principle by focusing only on performance tracking.
"""

import statistics
import time
from collections import defaultdict
from typing import TYPE_CHECKING, TypedDict

import pandas as pd

from ztb.metrics.metrics import skewness
from ztb.utils.logging_utils import get_logger

from .history_helpers import append_with_compaction
from .interfaces import IPerformanceTracker

if TYPE_CHECKING:
    from ..types import PerformanceStats
    from ..action_signal_guide import ActionSignal

class PortfolioState(TypedDict, total=False):
    value: float
    position: float

class CorrelationEntry(TypedDict):
    timestamp: float
    signal_action: str
    signal_confidence: float
    signal_reason: str
    sac_action: float | None
    market_regime: str
    portfolio_value: float
    position_size: float

class PerformanceTracker(IPerformanceTracker):
    """
    Tracks performance metrics for ActionSignalGuide operations.

    This class monitors:
    - Signal generation times
    - Pattern recognition performance
    - Cache hit rates
    - Memory usage
    - Error rates
    """

    def __init__(self, enable_detailed_tracking: bool = True):
        """
        Initialize PerformanceTracker.

        Args:
            enable_detailed_tracking: Whether to track detailed metrics
        """
        self.enable_detailed_tracking = enable_detailed_tracking
        self.logger = get_logger("ztb.trading.strategies.performance_tracker")

        # Timing metrics
        self.signal_generation_times: list[float] = []
        self.pattern_recognition_times: dict[str, list[float]] = defaultdict(list)
        self.cache_operation_times: list[float] = []

        # Performance counters
        self.total_signals_generated = 0
        self.total_patterns_recognized = 0
        self.cache_hits = 0
        self.cache_misses = 0
        self.errors = 0

        # Pattern-specific metrics
        self.pattern_success_rates: dict[str, dict[str, int]] = defaultdict(
            lambda: {"success": 0, "total": 0}
        )
        self.pattern_strengths: dict[str, list[float]] = defaultdict(list)
        self.pattern_confidences: dict[str, list[float]] = defaultdict(list)

        # Memory and resource metrics
        self.memory_usage_samples: list[float] = []
        self.start_time = time.time()

        # SAC correlation tracking
        self.max_history_size = 1000
        self._sac_history_high_water = self.max_history_size * 2
        self.sac_action_history: list[float] = []
        self.signal_sac_correlation_data: list[CorrelationEntry] = []
        self.regime_performance_correlation: dict[str, list[float]] = defaultdict(list)

    def record_pattern_recognition(
        self, pattern_type: str, duration: float, success: bool
    ) -> None:
        """
        Record pattern recognition performance.

        Args:
            pattern_type: Type of pattern recognized
            duration: Time taken for recognition
            success: Whether recognition was successful
        """
        if self.enable_detailed_tracking:
            append_with_compaction(
                self.pattern_recognition_times[pattern_type],
                duration,
                high_water=100,
                retain=50,
            )

        self.total_patterns_recognized += 1

        # Update success rates
        self.pattern_success_rates[pattern_type]["total"] += 1
        if success:
            self.pattern_success_rates[pattern_type]["success"] += 1

    def record_signal_generation(self, duration: float) -> None:
        """
        Record signal generation performance.

        Args:
            duration: Time taken for signal generation
        """
        if self.enable_detailed_tracking:
            append_with_compaction(
                self.signal_generation_times,
                duration,
                high_water=100,
                retain=50,
            )

        self.total_signals_generated += 1

    def record_pattern_signal(
        self, pattern_type: str, strength: float, confidence: float
    ) -> None:
        """
        Record pattern signal metrics.

        Args:
            pattern_type: Type of pattern
            strength: Signal strength
            confidence: Signal confidence
        """
        if self.enable_detailed_tracking:
            append_with_compaction(
                self.pattern_strengths[pattern_type],
                strength,
                high_water=100,
                retain=50,
            )
            append_with_compaction(
                self.pattern_confidences[pattern_type],
                confidence,
                high_water=100,
                retain=50,
            )

    def record_cache_operation(self, duration: float, hit: bool) -> None:
        """
        Record cache operation performance.

        Args:
            duration: Time taken for cache operation
            hit: Whether it was a cache hit
        """
        append_with_compaction(
            self.cache_operation_times,
            duration,
            high_water=500,
            retain=250,
        )

        if hit:
            self.cache_hits += 1
        else:
            self.cache_misses += 1

    def record_error(self, error_type: str, error_message: str) -> None:
        """
        Record error occurrence.

        Args:
            error_type: Type of error
            error_message: Error message
        """
        self.errors += 1
        self.logger.warning(
            f"Performance error recorded: {error_type} - {error_message}"
        )

    def record_memory_usage(self, memory_mb: float) -> None:
        """
        Record memory usage sample.

        Args:
            memory_mb: Memory usage in MB
        """
        append_with_compaction(
            self.memory_usage_samples,
            memory_mb,
            high_water=100,
            retain=50,
        )

    def track_sac_correlation(
        self,
        signal: "ActionSignal",
        sac_action: int | float | None = None,
        market_regime: str | None = None,
        portfolio_state: PortfolioState | None = None,
    ) -> None:
        """
        Track correlation between Action Signal Guide signals and SAC actions.

        Args:
            signal: Generated Action Signal Guide signal
            sac_action: Corresponding SAC action (-1 to 1)
            market_regime: Current market regime
            portfolio_state: Current portfolio state
        """
        if not self.enable_detailed_tracking:
            return

        signal_action = self._coerce_signal_action(getattr(signal, "action", "unknown"))
        signal_confidence = self._coerce_float(getattr(signal, "confidence", 0.0))
        normalized_sac_action = self._coerce_optional_float(sac_action)
        correlation_entry: CorrelationEntry = {
            "timestamp": time.time(),
            "signal_action": signal_action,
            "signal_confidence": signal_confidence,
            "signal_reason": str(getattr(signal, "reason", "")),
            "sac_action": normalized_sac_action,
            "market_regime": market_regime or "unknown",
            "portfolio_value": self._coerce_float(
                portfolio_state.get("value", 0.0) if portfolio_state else 0.0
            ),
            "position_size": self._coerce_float(
                portfolio_state.get("position", 0.0) if portfolio_state else 0.0
            ),
        }

        append_with_compaction(
            self.signal_sac_correlation_data,
            correlation_entry,
            high_water=self._sac_history_high_water,
            retain=self.max_history_size,
        )

        # Track regime-specific performance
        if market_regime:
            regime_key = f"{market_regime}_{signal_action}"
            if normalized_sac_action is not None:
                append_with_compaction(
                    self.sac_action_history,
                    normalized_sac_action,
                    high_water=1000,
                    retain=500,
                )
                # Calculate correlation strength (simplified)
                correlation_strength = (
                    abs(signal_confidence - abs(normalized_sac_action))
                    if signal_confidence
                    else 0.0
                )
                append_with_compaction(
                    self.regime_performance_correlation[regime_key],
                    correlation_strength,
                    high_water=500,
                    retain=250,
                )

    def get_performance_summary(self) -> dict[str, object]:
        """
        Get comprehensive performance summary.

        Returns:
            Dictionary with performance metrics
        """
        summary = {
            "total_signals_generated": self.total_signals_generated,
            "total_patterns_recognized": self.total_patterns_recognized,
            "total_errors": self.errors,
            "uptime_seconds": time.time() - self.start_time,
            "cache_hit_rate": self._calculate_cache_hit_rate(),
        }

        # Signal generation metrics
        if self.signal_generation_times:
            summary["signal_generation"] = {
                "avg_time": statistics.mean(self.signal_generation_times),
                "min_time": min(self.signal_generation_times),
                "max_time": max(self.signal_generation_times),
                "median_time": statistics.median(self.signal_generation_times),
                "samples": len(self.signal_generation_times),
            }

        # Pattern recognition metrics
        if self.enable_detailed_tracking and self.pattern_recognition_times:
            pattern_metrics = {}
            for pattern_type, times in self.pattern_recognition_times.items():
                if times:
                    pattern_metrics[pattern_type] = {
                        "avg_time": statistics.mean(times),
                        "success_rate": self._calculate_pattern_success_rate(
                            pattern_type
                        ),
                        "samples": len(times),
                    }
            summary["pattern_recognition"] = pattern_metrics

        # Cache metrics
        if self.cache_operation_times:
            summary["cache_operations"] = {
                "avg_time": statistics.mean(self.cache_operation_times),
                "total_operations": len(self.cache_operation_times),
                "hit_rate": self._calculate_cache_hit_rate(),
            }

        # Memory metrics
        if self.memory_usage_samples:
            summary["memory"] = {
                "avg_usage_mb": statistics.mean(self.memory_usage_samples),
                "max_usage_mb": max(self.memory_usage_samples),
                "current_usage_mb": self.memory_usage_samples[-1]
                if self.memory_usage_samples
                else 0,
            }

        return summary

    def get_pattern_performance(
        self, pattern_type: str | None = None
    ) -> dict[str, object]:
        """
        Get detailed pattern performance metrics.

        Args:
            pattern_type: Specific pattern type, or None for all

        Returns:
            Dictionary with pattern performance metrics
        """
        if pattern_type:
            return self._get_single_pattern_performance(pattern_type)
        else:
            return {
                pt: self._get_single_pattern_performance(pt)
                for pt in self.pattern_success_rates.keys()
            }

    def reset_metrics(self) -> None:
        """
        Reset all performance metrics.
        """
        self.signal_generation_times.clear()
        self.pattern_recognition_times.clear()
        self.cache_operation_times.clear()

        self.total_signals_generated = 0
        self.total_patterns_recognized = 0
        self.cache_hits = 0
        self.cache_misses = 0
        self.errors = 0

        self.pattern_success_rates.clear()
        self.pattern_strengths.clear()
        self.pattern_confidences.clear()
        self.memory_usage_samples.clear()

        self.start_time = time.time()
        self.logger.info("Performance metrics reset")

    def _calculate_cache_hit_rate(self) -> float:
        """
        Calculate cache hit rate.

        Returns:
            Cache hit rate as percentage
        """
        total = self.cache_hits + self.cache_misses
        return (self.cache_hits / total * 100) if total > 0 else 0.0

    def _calculate_pattern_success_rate(self, pattern_type: str) -> float:
        """
        Calculate success rate for a pattern type.

        Args:
            pattern_type: Pattern type

        Returns:
            Success rate as percentage
        """
        stats = self.pattern_success_rates.get(pattern_type, {"success": 0, "total": 0})
        return (stats["success"] / stats["total"] * 100) if stats["total"] > 0 else 0.0

    def _get_single_pattern_performance(self, pattern_type: str) -> dict[str, object]:
        """
        Get performance metrics for a single pattern type.

        Args:
            pattern_type: Pattern type

        Returns:
            Dictionary with pattern metrics
        """
        metrics = {
            "success_rate": self._calculate_pattern_success_rate(pattern_type),
            "total_attempts": self.pattern_success_rates[pattern_type]["total"],
            "successful_attempts": self.pattern_success_rates[pattern_type]["success"],
        }

        # Add timing metrics if available
        if pattern_type in self.pattern_recognition_times:
            times = self.pattern_recognition_times[pattern_type]
            if times:
                metrics["recognition_time"] = {
                    "avg": statistics.mean(times),
                    "min": min(times),
                    "max": max(times),
                }

        # Add signal quality metrics if available
        if (
            pattern_type in self.pattern_strengths
            and self.pattern_strengths[pattern_type]
        ):
            strengths = self.pattern_strengths[pattern_type]
            metrics["signal_strength"] = {
                "avg": statistics.mean(strengths),
                "min": min(strengths),
                "max": max(strengths),
            }

        if (
            pattern_type in self.pattern_confidences
            and self.pattern_confidences[pattern_type]
        ):
            confidences = self.pattern_confidences[pattern_type]
            metrics["signal_confidence"] = {
                "avg": statistics.mean(confidences),
                "min": min(confidences),
                "max": max(confidences),
            }

        return metrics

    def record_cache_hit(self) -> None:
        """Record cache hit."""
        self.cache_hits += 1

    def record_cache_miss(self) -> None:
        """Record cache miss."""
        self.cache_misses += 1

    def get_performance_stats(self) -> "PerformanceStats":
        """
        Get performance statistics.

        Returns:
            Dictionary of performance statistics
        """
        total_cache_operations = self.cache_hits + self.cache_misses
        cache_hit_rate = (
            (self.cache_hits / total_cache_operations)
            if total_cache_operations > 0
            else 0.0
        )

        stats = {
            "total_signals_generated": self.total_signals_generated,
            "total_patterns_recognized": self.total_patterns_recognized,
            "cache_hit_rate": cache_hit_rate,
            "cache_hits": self.cache_hits,
            "cache_misses": self.cache_misses,
            "errors": self.errors,
            "uptime_seconds": time.time() - self.start_time,
        }

        # Add timing statistics if available
        if self.signal_generation_times:
            stats["signal_generation_time"] = {
                "avg": statistics.mean(self.signal_generation_times),
                "min": min(self.signal_generation_times),
                "max": max(self.signal_generation_times),
                "count": len(self.signal_generation_times),
            }

        if self.cache_operation_times:
            stats["cache_operation_time"] = {
                "avg": statistics.mean(self.cache_operation_times),
                "min": min(self.cache_operation_times),
                "max": max(self.cache_operation_times),
                "count": len(self.cache_operation_times),
            }

        return stats

    def get_sac_correlation_analysis(self) -> dict[str, object]:
        """
        Get comprehensive SAC correlation analysis.

        Returns:
            Dictionary containing correlation analysis results
        """
        if not self.signal_sac_correlation_data:
            return {"error": "No SAC correlation data available"}

        df = pd.DataFrame(self.signal_sac_correlation_data)

        analysis = {
            "total_correlation_points": len(df),
            "correlation_period_days": (df["timestamp"].max() - df["timestamp"].min())
            / 86400,
            "signal_distribution": self._analyze_signal_distribution(df),
            "sac_action_distribution": self._analyze_sac_action_distribution(df),
            "regime_correlation": self._analyze_regime_correlation(df),
            "performance_correlation": self._analyze_performance_correlation(df),
            "temporal_patterns": self._analyze_temporal_correlation_patterns(df),
        }

        return analysis

    def _analyze_signal_distribution(
        self, df: pd.DataFrame
    ) -> dict[str, object]:
        """Analyze signal distribution in correlation data."""
        signal_counts = df["signal_action"].value_counts().to_dict()
        total_signals = len(df)

        return {
            "signal_counts": signal_counts,
            "signal_percentages": {
                k: v / total_signals for k, v in signal_counts.items()
            },
            "most_common_signal": max(signal_counts.items(), key=lambda x: x[1])[0]
            if signal_counts
            else None,
        }

    def _analyze_sac_action_distribution(
        self, df: pd.DataFrame
    ) -> dict[str, object]:
        """Analyze SAC action distribution."""
        sac_actions = df["sac_action"].dropna()

        if len(sac_actions) == 0:
            return {"error": "No SAC action data available"}

        return {
            "mean_sac_action": float(sac_actions.mean()),
            "std_sac_action": float(sac_actions.std()),
            "sac_action_range": [float(sac_actions.min()), float(sac_actions.max())],
            "sac_action_skewness": float(skewness(sac_actions)),
            "strong_buy_signals": int((sac_actions > 0.7).sum()),
            "strong_sell_signals": int((sac_actions < -0.7).sum()),
        }

    def _analyze_regime_correlation(
        self, df: pd.DataFrame
    ) -> dict[str, object]:
        """Analyze correlation by market regime."""
        regime_groups = df.groupby("market_regime")

        regime_analysis = {}
        for regime, group in regime_groups:
            valid_data = group.dropna(subset=["sac_action", "signal_confidence"])

            if len(valid_data) > 1:
                correlation = valid_data["signal_confidence"].corr(
                    valid_data["sac_action"]
                )
                regime_analysis[regime] = {
                    "correlation_coefficient": float(correlation)
                    if not pd.isna(correlation)
                    else 0.0,
                    "sample_size": len(valid_data),
                    "avg_signal_confidence": float(
                        valid_data["signal_confidence"].mean()
                    ),
                    "avg_sac_action": float(valid_data["sac_action"].mean()),
                }
            else:
                regime_analysis[regime] = {
                    "error": "Insufficient data for correlation analysis"
                }

        return regime_analysis

    def _analyze_performance_correlation(
        self, df: pd.DataFrame
    ) -> dict[str, object]:
        """Analyze correlation between signals and portfolio performance."""
        # Calculate forward returns (simplified approach)
        df_sorted = df.sort_values("timestamp").copy()

        # Calculate correlation between signal confidence and subsequent portfolio changes
        valid_data = df_sorted.dropna(subset=["signal_confidence", "portfolio_value"])

        if len(valid_data) < 2:
            return {"error": "Insufficient data for performance correlation"}

        # Simple correlation analysis (in practice, you'd want more sophisticated forward-looking analysis)
        confidence_correlation = valid_data["signal_confidence"].corr(
            valid_data["portfolio_value"]
        )

        return {
            "signal_confidence_portfolio_correlation": float(confidence_correlation)
            if not pd.isna(confidence_correlation)
            else 0.0,
            "sample_size": len(valid_data),
            "correlation_strength": self._interpret_correlation_strength(
                confidence_correlation
            ),
        }

    def _analyze_temporal_correlation_patterns(
        self, df: pd.DataFrame
    ) -> dict[str, object]:
        """Analyze temporal patterns in correlation data."""
        df_sorted = df.sort_values("timestamp").copy()
        df_sorted["hour"] = pd.to_datetime(df_sorted["timestamp"], unit="s").dt.hour
        df_sorted["day_of_week"] = pd.to_datetime(
            df_sorted["timestamp"], unit="s"
        ).dt.dayofweek

        hourly_patterns = (
            df_sorted.groupby("hour")
            .agg({"signal_confidence": "mean", "sac_action": "mean"})
            .to_dict()
        )
        signal_confidence_by_hour = hourly_patterns.get("signal_confidence", {})
        sac_action_by_hour = hourly_patterns.get("sac_action", {})

        if not signal_confidence_by_hour:
            return {
                "hourly_signal_confidence": signal_confidence_by_hour,
                "hourly_sac_action": sac_action_by_hour,
                "best_signal_hour": None,
                "best_sac_hour": None,
            }

        return {
            "hourly_signal_confidence": signal_confidence_by_hour,
            "hourly_sac_action": sac_action_by_hour,
            "best_signal_hour": max(
                signal_confidence_by_hour.items(), key=lambda x: x[1]
            )[0],
            "best_sac_hour": max(sac_action_by_hour.items(), key=lambda x: x[1])[0]
            if sac_action_by_hour
            else None,
        }

    @staticmethod
    def _coerce_signal_action(action: object) -> str:
        raw_action = getattr(action, "value", action)
        return str(raw_action)

    @staticmethod
    def _coerce_float(value: object, default: float = 0.0) -> float:
        try:
            return float(value)
        except (TypeError, ValueError):
            return default

    @classmethod
    def _coerce_optional_float(cls, value: object) -> float | None:
        if value is None:
            return None
        return cls._coerce_float(value)

    def _interpret_correlation_strength(self, correlation: float) -> str:
        """Interpret correlation coefficient strength."""
        if pd.isna(correlation):
            return "no_data"
        abs_corr = abs(correlation)
        if abs_corr >= 0.8:
            return "very_strong"
        elif abs_corr >= 0.6:
            return "strong"
        elif abs_corr >= 0.3:
            return "moderate"
        elif abs_corr >= 0.1:
            return "weak"
        else:
            return "very_weak"
