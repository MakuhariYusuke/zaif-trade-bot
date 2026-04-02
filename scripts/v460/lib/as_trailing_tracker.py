"""694# AS trailing rate tracker by regime×spread bucket."""

from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass


@dataclass(frozen=True)
class ASTrailingConfig:
    """Configuration for the AS trailing gate."""

    enabled: bool = False
    window_size: int = 100
    spread_bucket_edges: tuple[float, ...] = (1500.0, 2500.0, 3500.0)
    soft_threshold: float = 0.30
    hard_veto_threshold: float = 0.45
    offset_boost_factor: float = 1.3
    min_samples: int = 10


@dataclass(frozen=True)
class _FillEvent:
    timestamp: float
    is_adverse: bool


class ASTrailingTracker:
    """Track trailing AS rate by regime×spread bucket."""

    def __init__(self, config: ASTrailingConfig) -> None:
        self._config = config
        self._buckets: dict[tuple[str, int], deque[_FillEvent]] = {}

    @property
    def config(self) -> ASTrailingConfig:
        return self._config

    def reconfigure(self, config: ASTrailingConfig) -> None:
        """Swap runtime config while preserving existing bucket history."""
        if config == self._config:
            return
        self._config = config
        if not self._config.enabled:
            return
        trimmed: dict[tuple[str, int], deque[_FillEvent]] = {}
        for key, bucket in self._buckets.items():
            trimmed[key] = deque(
                list(bucket)[-self._config.window_size :],
                maxlen=self._config.window_size,
            )
        self._buckets = trimmed

    def _spread_bucket(self, spread: float) -> int:
        for index, edge in enumerate(self._config.spread_bucket_edges):
            if spread < edge:
                return index
        return len(self._config.spread_bucket_edges)

    def record_fill(
        self,
        *,
        regime: str,
        spread: float,
        is_adverse: bool,
        timestamp: float | None = None,
    ) -> None:
        if not self._config.enabled:
            return
        key = (regime, self._spread_bucket(spread))
        bucket = self._buckets.setdefault(
            key,
            deque(maxlen=self._config.window_size),
        )
        bucket.append(
            _FillEvent(
                timestamp=time.time() if timestamp is None else timestamp,
                is_adverse=is_adverse,
            )
        )

    def get_as_rate(self, *, regime: str, spread: float) -> tuple[float | None, int]:
        key = (regime, self._spread_bucket(spread))
        bucket = self._buckets.get(key)
        if bucket is None:
            return None, 0
        sample_count = len(bucket)
        if sample_count < self._config.min_samples:
            return None, sample_count
        adverse_count = sum(1 for event in bucket if event.is_adverse)
        return adverse_count / sample_count, sample_count

    def evaluate(
        self,
        *,
        regime: str,
        spread: float,
        side: str,
    ) -> tuple[str, float | None, float | None]:
        del side
        if not self._config.enabled:
            return "none", None, None
        as_rate, _sample_count = self.get_as_rate(regime=regime, spread=spread)
        if as_rate is None:
            return "none", None, None
        if as_rate >= self._config.hard_veto_threshold:
            return "veto", None, as_rate
        if as_rate >= self._config.soft_threshold:
            return "boost", self._config.offset_boost_factor, as_rate
        return "none", None, as_rate


__all__ = ["ASTrailingConfig", "ASTrailingTracker"]
