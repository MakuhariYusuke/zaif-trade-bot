"""690# Bucketed skip budget helper for regime×side skip-gate control."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from scripts.v460.lib.fill_config import FillTestConfig


@dataclass(frozen=True)
class BucketKey:
    regime: str
    side: str


@dataclass
class BucketState:
    skip_count: int = 0
    pass_count: int = 0
    window_start_ts: float = 0.0


class BucketedSkipBudget:
    """Maintain per-regime×side skip/pass counts in a rolling window."""

    def __init__(self, config: FillTestConfig) -> None:
        self._config = config
        self._states: dict[BucketKey, BucketState] = {}

    @staticmethod
    def normalize_regime(regime: str | None) -> str:
        normalized = (regime or "unknown").strip().lower()
        return normalized or "unknown"

    @staticmethod
    def normalize_side(side: str) -> str:
        normalized = side.strip().lower()
        return normalized or "buy"

    @staticmethod
    def _now() -> float:
        return time.time()

    def _window_seconds(self) -> float:
        return float(self._config.skip_gate_budget_window_min) * 60.0

    def _make_key(self, regime: str | None, side: str) -> BucketKey:
        return BucketKey(
            regime=self.normalize_regime(regime),
            side=self.normalize_side(side),
        )

    def _get_or_create_state(self, key: BucketKey) -> BucketState:
        state = self._states.get(key)
        if state is None:
            state = BucketState(window_start_ts=self._now())
            self._states[key] = state
        elif state.window_start_ts <= 0.0:
            state.window_start_ts = self._now()
        return state

    def _rotate_if_expired(self, key: BucketKey) -> None:
        state = self._get_or_create_state(key)
        now_ts = self._now()
        if now_ts - state.window_start_ts < self._window_seconds():
            return
        state.skip_count = 0
        state.pass_count = 0
        state.window_start_ts = now_ts

    def get_state(self, regime: str, side: str) -> BucketState:
        key = self._make_key(regime, side)
        self._rotate_if_expired(key)
        return self._get_or_create_state(key)

    def is_budget_exhausted(self, regime: str, side: str) -> bool:
        key = self._make_key(regime, side)
        self._rotate_if_expired(key)
        state = self._get_or_create_state(key)
        limit = self._config.get_budget_limit(key.regime, key.side)
        return state.skip_count >= limit

    def record_decision(self, regime: str, side: str, *, skipped: bool) -> None:
        key = self._make_key(regime, side)
        self._rotate_if_expired(key)
        state = self._get_or_create_state(key)
        if skipped:
            state.skip_count += 1
        else:
            state.pass_count += 1


__all__ = ["BucketKey", "BucketState", "BucketedSkipBudget"]
