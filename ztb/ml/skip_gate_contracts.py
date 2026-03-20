"""Shared skip-gate contracts for evaluation and runtime wiring."""

from __future__ import annotations

from pathlib import Path
from typing import Protocol

from scripts.v460.lib.ob_utils import OrderBookSnapshot


class SkipGateAdapter(Protocol):
    """Adapter methods required for skip-gate feature collection."""

    async def get_recent_trades(self, symbol: str, limit: int = 100) -> object: ...

    async def get_orderbook(
        self,
        symbol: str,
        depth: int = 1,
    ) -> OrderBookSnapshot | None: ...


class SkipGateConfigLike(Protocol):
    mode: str
    as_threshold: float
    threshold_bps: float
    max_skip_rate: float
    buy_enabled: bool
    sell_enabled: bool
    as_threshold_buy: float | None
    as_threshold_sell: float | None
    use_ob_features: bool
    adaptive_threshold: bool
    target_skip_rate_buy: float
    target_skip_rate_sell: float
    adaptive_window: int
    adaptive_min_samples: int
    adaptive_step: float
    adaptive_floor: float
    adaptive_ceiling: float
    regime_thresholds: dict[str, float]


class SkipDecisionLike(Protocol):
    should_skip: bool
    predicted_pnl_bps: float
    threshold_bps: float
    reason: str
    model_used: str
    as_probability: float | None
    threshold_used: float | None
    features_used: int


class SkipGateLike(Protocol):
    config: SkipGateConfigLike
    metadata: dict[str, object]
    feature_cols: list[str]

    def evaluate(
        self,
        features: dict[str, object],
        *,
        side: str | None = None,
        regime: str | None = None,
        threshold_offset: float = ...,
    ) -> SkipDecisionLike:
        ...


class SkipGateClassLike(Protocol):
    @staticmethod
    def load(path: Path) -> SkipGateLike:
        ...


__all__ = [
    "SkipGateAdapter",
    "SkipGateClassLike",
    "SkipGateConfigLike",
    "SkipDecisionLike",
    "SkipGateLike",
]
