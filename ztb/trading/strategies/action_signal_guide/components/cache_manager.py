"""
CacheManager Component.

This component is responsible for managing cached data and computations.
Follows Single Responsibility Principle by focusing only on caching operations.
"""

import time
from collections import OrderedDict
from typing import TYPE_CHECKING, Any, List, Optional, Tuple, Union

import numpy as np

from ztb.utils.logging_utils import get_logger

from .interfaces import ICacheManager

if TYPE_CHECKING:
    from ..action_signal_guide import ActionSignal
    from ..types import CacheStats


class CacheManager(ICacheManager):
    """
    Manages cached data and computations for ActionSignalGuide.

    This class handles:
    - LRU cache for observations and signals
    - Cache size management
    - Cache hit/miss statistics
    - Memory usage optimization
    """

    def __init__(self, max_cache_size: int = 1000, cache_ttl: int = 300):
        """
        Initialize CacheManager.

        Args:
            max_cache_size: Maximum number of cached items
            cache_ttl: Time-to-live for cached items in seconds
        """
        self.max_cache_size = max_cache_size
        self.cache_ttl = cache_ttl
        self.logger = get_logger("ztb.trading.strategies.cache_manager")

        # Cache storage: OrderedDict for LRU behavior
        self.observation_cache: OrderedDict[
            str, Tuple[np.ndarray, float]
        ] = OrderedDict()
        self.signal_cache: OrderedDict[
            str, Tuple["ActionSignal", float]
        ] = OrderedDict()
        self.computation_cache: OrderedDict[str, Tuple[Any, float]] = OrderedDict()

        # Statistics
        self.cache_hits = 0
        self.cache_misses = 0
        self.evictions = 0

    def get_cached_observation(self, key: str) -> Optional[np.ndarray]:
        """
        Retrieve cached observation.

        Args:
            key: Cache key

        Returns:
            Cached observation or None if not found/expired
        """
        return self._get_from_cache(self.observation_cache, key)

    def cache_observation(self, key: str, observation: np.ndarray) -> None:
        """
        Cache observation data.

        Args:
            key: Cache key
            observation: Observation to cache
        """
        self._put_in_cache(self.observation_cache, key, observation)

    def get_cached_signal(self, cache_key: str) -> Optional[Union["ActionSignal", List["ActionSignal"]]]:
        """
        Retrieve cached signal.

        Args:
            cache_key: Cache key

        Returns:
            Cached signal or None if not found/expired
        """
        return self._get_from_cache(self.signal_cache, cache_key)

    def cache_signal(self, cache_key: str, signal: Union["ActionSignal", List["ActionSignal"]]) -> None:
        """
        Cache signal data.

        Args:
            key: Cache key
            signal: Signal or list of signals to cache
        """
        self._put_in_cache(self.signal_cache, cache_key, signal)

    def get_cached_computation(self, key: str) -> Optional[Any]:
        """
        Retrieve cached computation result.

        Args:
            key: Cache key

        Returns:
            Cached computation result or None if not found/expired
        """
        return self._get_from_cache(self.computation_cache, key)

    def cache_computation(self, key: str, result: Any) -> None:
        """
        Cache computation result.

        Args:
            key: Cache key
            result: Computation result to cache
        """
        self._put_in_cache(self.computation_cache, key, result)

    def invalidate_cache(self, pattern: Optional[str] = None) -> None:
        """
        Invalidate cache entries.

        Args:
            pattern: Optional pattern to match keys for selective invalidation
        """
        if pattern is None:
            # Clear all caches
            self.observation_cache.clear()
            self.signal_cache.clear()
            self.computation_cache.clear()
            self.logger.info("All caches invalidated")
        else:
            # Selective invalidation
            obs_keys_to_remove = [
                k for k in self.observation_cache.keys() if pattern in k
            ]
            sig_keys_to_remove = [k for k in self.signal_cache.keys() if pattern in k]
            comp_keys_to_remove = [
                k for k in self.computation_cache.keys() if pattern in k
            ]

            for key in obs_keys_to_remove:
                del self.observation_cache[key]
            for key in sig_keys_to_remove:
                del self.signal_cache[key]
            for key in comp_keys_to_remove:
                del self.computation_cache[key]

            total_removed = (
                len(obs_keys_to_remove)
                + len(sig_keys_to_remove)
                + len(comp_keys_to_remove)
            )
            self.logger.info(
                f"Invalidated {total_removed} cache entries matching pattern: {pattern}"
            )

    def get_cache_stats(self) -> "CacheStats":
        """
        Get cache statistics.

        Returns:
            Dictionary with cache statistics
        """
        total_requests = self.cache_hits + self.cache_misses
        hit_rate = self.cache_hits / total_requests if total_requests > 0 else 0

        return {
            "observation_cache_size": len(self.observation_cache),
            "signal_cache_size": len(self.signal_cache),
            "computation_cache_size": len(self.computation_cache),
            "total_cache_size": len(self.observation_cache)
            + len(self.signal_cache)
            + len(self.computation_cache),
            "max_cache_size": self.max_cache_size,
            "cache_hits": self.cache_hits,
            "cache_misses": self.cache_misses,
            "hit_rate": hit_rate,
            "evictions": self.evictions,
            "cache_ttl": self.cache_ttl,
        }

    def optimize_cache(self) -> None:
        """
        Optimize cache by removing expired entries and enforcing size limits.
        """
        current_time = time.time()

        # Remove expired entries
        self._remove_expired_entries(self.observation_cache, current_time)
        self._remove_expired_entries(self.signal_cache, current_time)
        self._remove_expired_entries(self.computation_cache, current_time)

        # Enforce size limits
        self._enforce_size_limit(self.observation_cache)
        self._enforce_size_limit(self.signal_cache)
        self._enforce_size_limit(self.computation_cache)

    def _get_from_cache(
        self, cache: OrderedDict[str, Tuple[Any, float]], key: str
    ) -> Optional[Any]:
        """
        Generic cache retrieval with LRU behavior.

        Args:
            cache: Cache dictionary
            key: Cache key

        Returns:
            Cached value or None
        """
        if key not in cache:
            self.cache_misses += 1
            return None

        value, timestamp = cache[key]
        current_time = time.time()

        # Check if expired
        if current_time - timestamp > self.cache_ttl:
            del cache[key]
            self.cache_misses += 1
            return None

        # Move to end (most recently used)
        cache.move_to_end(key)
        self.cache_hits += 1

        return value

    def _put_in_cache(
        self, cache: OrderedDict[str, Tuple[Any, float]], key: str, value: Any
    ) -> None:
        """
        Generic cache storage with size management.

        Args:
            cache: Cache dictionary
            key: Cache key
            value: Value to cache
        """
        current_time = time.time()

        # If key exists, update it
        if key in cache:
            cache[key] = (value, current_time)
            cache.move_to_end(key)
        else:
            # Add new entry
            cache[key] = (value, current_time)
            cache.move_to_end(key)

            # Enforce size limit
            if len(cache) > self.max_cache_size:
                # Remove least recently used
                removed_key, _ = cache.popitem(last=False)
                self.evictions += 1
                self.logger.debug(f"Evicted cache entry: {removed_key}")

    def _remove_expired_entries(
        self, cache: OrderedDict[str, Tuple[Any, float]], current_time: float
    ) -> None:
        """
        Remove expired entries from cache.

        Args:
            cache: Cache dictionary
            current_time: Current timestamp
        """
        expired_keys = [
            key
            for key, (_, timestamp) in cache.items()
            if current_time - timestamp > self.cache_ttl
        ]

        for key in expired_keys:
            del cache[key]

        if expired_keys:
            self.logger.debug(f"Removed {len(expired_keys)} expired cache entries")

    def _enforce_size_limit(self, cache: OrderedDict[str, Tuple[Any, float]]) -> None:
        """
        Enforce maximum size limit on cache.

        Args:
            cache: Cache dictionary
        """
        while len(cache) > self.max_cache_size:
            removed_key, _ = cache.popitem(last=False)
            self.evictions += 1
            self.logger.debug(f"Size limit eviction: {removed_key}")

    def clear_expired_cache(self) -> None:
        """
        Clear expired cache entries.
        """
        current_time = time.time()

        # Remove expired entries from all caches
        self._remove_expired_entries(self.observation_cache, current_time)
        self._remove_expired_entries(self.signal_cache, current_time)
        self._remove_expired_entries(self.computation_cache, current_time)

        self.logger.debug("Expired cache entries cleared")
