"""145# §9-#3 / §10.2-#1 / §10.1-#3: OB (orderbook) 正規化ユーティリティ.

OrderBookSnapshot.bids/asks は list[tuple[float, float]] だが、
一部モジュール (skip_gate_evaluator) では .price/.quantity アクセスをしていた。
tuple/object 両対応の安全な抽出関数を提供し、散在する dual-format ロジックを一元化する。

§10.1-#3: MarketDataAccessor — adapter を薄くラップし、
best_bid_ask / depth_volume / spread 等の市場データ取得を型安全に提供。
"""

from __future__ import annotations

import logging
from typing import Protocol, Sequence, TypeAlias, Union, cast

logger = logging.getLogger(__name__)

# 156# D-1: 型安全向上 — OB レベルの型を明示
# tuple[float, float] (price, size) or NamedTuple/dataclass with .price/.quantity
OrderBookLevel: TypeAlias = Union[tuple[float, float], "object"]
OrderBookLevels: TypeAlias = Sequence[OrderBookLevel]


def extract_price(level: OrderBookLevel) -> float:
    """板レベルから price を抽出 (tuple / object 両対応)."""
    if isinstance(level, (list, tuple)):
        if not level:
            return 0.0
        return float(level[0])
    return float(getattr(level, "price", 0.0))


def extract_size(level: OrderBookLevel) -> float:
    """板レベルから size (quantity) を抽出 (tuple / object 両対応)."""
    if isinstance(level, (list, tuple)):
        if len(level) < 2:
            return 0.0
        return float(level[1])
    return float(getattr(level, "quantity", getattr(level, "size", 0.0)))


def best_bid_ask(
    ob: object,
) -> tuple[float | None, float | None]:
    """OrderBookSnapshot から best bid/ask を安全に抽出.

    Returns:
        (best_bid, best_ask) — データ不足時は None.
    """
    bids = _coerce_levels(getattr(ob, "bids", None))
    asks = _coerce_levels(getattr(ob, "asks", None))
    bid = extract_price(bids[0]) if bids else None
    ask = extract_price(asks[0]) if asks else None
    return bid, ask


def depth_volume(levels: OrderBookLevels, depth: int = 5) -> float:
    """板の指定深さまでの合計出来高を計算."""
    return sum(extract_size(lv) for lv in levels[:depth])


def _coerce_levels(levels: object) -> OrderBookLevels:
    if isinstance(levels, Sequence):
        return cast(OrderBookLevels, levels)
    return ()


# ---------------------------------------------------------------------------
# §10.1-#3: MarketDataAccessor — adapter wrapper
# ---------------------------------------------------------------------------

class _HasGetOrderbook(Protocol):
    """Protocol for adapter with get_orderbook method."""

    async def get_orderbook(self, symbol: str, depth: int = ...) -> object: ...


class MarketDataAccessor:
    """薄い adapter ラッパー: OB 正規化を一手に引き受ける.

    §10.1-#3: 散在する ``bids[0].price if hasattr ...`` の dual-format
    ロジックを排除し、呼び出し側は ``accessor.best_bid_ask()`` で完結。
    """

    def __init__(self, adapter: _HasGetOrderbook, symbol: str = "btc_jpy") -> None:
        self._adapter = adapter
        self._symbol = symbol

    async def best_bid_ask(
        self, depth: int = 1
    ) -> tuple[float | None, float | None]:
        """Get best bid/ask via adapter, normalised through ob_utils.

        Returns:
            (best_bid, best_ask) — API 失敗時は (None, None).
        """
        try:
            ob = await self._adapter.get_orderbook(self._symbol, depth=depth)
            return best_bid_ask(ob)
        except Exception as e:
            logger.debug("MarketDataAccessor.best_bid_ask failed: %s", e)
            return None, None

    async def spread(self, depth: int = 1) -> float | None:
        """Best ask - best bid. None if data unavailable."""
        bid, ask = await self.best_bid_ask(depth)
        if bid is not None and ask is not None:
            return ask - bid
        return None

    async def mid_price(self, depth: int = 1) -> float | None:
        """Mid-price = (bid + ask) / 2. None if data unavailable."""
        bid, ask = await self.best_bid_ask(depth)
        if bid is not None and ask is not None:
            return (bid + ask) / 2.0
        return None

    async def bid_depth_volume(self, depth: int = 5) -> float:
        """Bid-side depth volume up to *depth* levels."""
        try:
            ob = await self._adapter.get_orderbook(self._symbol, depth=depth)
            bids = _coerce_levels(getattr(ob, "bids", None))
            return depth_volume(bids, depth) if bids else 0.0
        except Exception:
            # 255# bare except → debug log (OB fetch 例外可観測化)
            logger.debug("bid_depth_volume fetch failed", exc_info=True)
            return 0.0

    async def ask_depth_volume(self, depth: int = 5) -> float:
        """Ask-side depth volume up to *depth* levels."""
        try:
            ob = await self._adapter.get_orderbook(self._symbol, depth=depth)
            asks = _coerce_levels(getattr(ob, "asks", None))
            return depth_volume(asks, depth) if asks else 0.0
        except Exception:
            # 255# bare except → debug log (OB fetch 例外可観測化)
            logger.debug("ask_depth_volume fetch failed", exc_info=True)
            return 0.0
