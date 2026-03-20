"""Shared pricing contracts used by fill-test pricing modules."""

from __future__ import annotations

from typing import NamedTuple, Protocol

from scripts.v460.lib.ob_utils import OrderBookSnapshot


class OrderbookProvider(Protocol):
    """Adapter protocol for fetching orderbook snapshots."""

    async def get_orderbook(self, symbol: str, depth: int = 1) -> OrderBookSnapshot: ...


class MakerPriceResult(NamedTuple):
    """Return type for maker pricing computations."""

    price: float
    spread: float
    effective_offset_ratio: float


class ImbalanceResult(NamedTuple):
    """Return type for orderbook imbalance computations."""

    imbalance: float
    bid_total: float
    ask_total: float


__all__ = ["OrderbookProvider", "MakerPriceResult", "ImbalanceResult"]
