"""
Interfaces for ActionSignalGuide components.

This module defines interfaces for the ActionSignalGuide system components,
following SOLID principles for better maintainability and testability.
"""

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Optional

import pandas as pd

if TYPE_CHECKING:
    from ..action_signal_guide import ActionSignal
    from ..types import CacheStats, PatternStats, PerformanceStats


class ISignalGenerator(ABC):
    """Interface for signal generation."""

    @abstractmethod
    def generate_signal(
        self,
        data: pd.DataFrame,
        current_index: int,
        multi_timeframe_data: Optional[dict] = None,
    ) -> "ActionSignal":
        """
        Generate trading signal from market data.

        Args:
            data: OHLCV DataFrame
            current_index: Current bar index
            multi_timeframe_data: Optional multi-timeframe data

        Returns:
            Generated action signal
        """
        pass

    @abstractmethod
    def initialize_recognizers(self) -> None:
        """Initialize all pattern recognizers."""
        pass


class ICacheManager(ABC):
    """Interface for signal caching."""

    @abstractmethod
    def get_cached_signal(self, cache_key: str) -> Optional["ActionSignal"]:
        """
        Get cached signal if available and valid.

        Args:
            cache_key: Cache key for the signal

        Returns:
            Cached signal or None if not available/expired
        """
        pass

    @abstractmethod
    def cache_signal(self, cache_key: str, signal: "ActionSignal") -> None:
        """
        Cache a signal.

        Args:
            cache_key: Cache key for the signal
            signal: Signal to cache
        """
        pass

    @abstractmethod
    def clear_expired_cache(self) -> None:
        """Clear expired cache entries."""
        pass

    @abstractmethod
    def invalidate_cache(self, pattern: Optional[str] = None) -> None:
        """
        Invalidate cache entries.

        Args:
            pattern: Optional pattern to match keys for selective invalidation
        """
        pass

    @abstractmethod
    def get_cache_stats(self) -> "CacheStats":
        """
        Get cache statistics.

        Returns:
            Dictionary with cache statistics
        """
        pass


class IPerformanceTracker(ABC):
    """Interface for performance tracking."""

    @abstractmethod
    def record_signal_generation(self, processing_time: float) -> None:
        """
        Record signal generation performance.

        Args:
            processing_time: Time taken to generate signal
        """
        pass

    @abstractmethod
    def record_cache_hit(self) -> None:
        """Record cache hit."""
        pass

    @abstractmethod
    def record_cache_miss(self) -> None:
        """Record cache miss."""
        pass

    @abstractmethod
    def get_performance_stats(self) -> "PerformanceStats":
        """
        Get performance statistics.

        Returns:
            Dictionary of performance statistics
        """
        pass


class IPatternStatistics(ABC):
    """Interface for pattern statistics tracking."""

    @abstractmethod
    def record_pattern_signal(self, pattern_type: str, signal: "ActionSignal") -> None:
        """
        Record pattern signal statistics.

        Args:
            pattern_type: Type of pattern (e.g., 'candlestick', 'fibonacci')
            signal: Generated signal
        """
        pass

    @abstractmethod
    def get_pattern_statistics(
        self, pattern_type: Optional[str] = None
    ) -> "PatternStats":
        """
        Get pattern statistics.

        Args:
            pattern_type: Specific pattern type, or None for all

        Returns:
            Dictionary of pattern statistics by type
        """
        pass

    @abstractmethod
    def update_pattern_strength_stats(self, pattern_type: str, strength: float) -> None:
        """
        Update pattern strength statistics.

        Args:
            pattern_type: Type of pattern
            strength: Signal strength value
        """
        pass
