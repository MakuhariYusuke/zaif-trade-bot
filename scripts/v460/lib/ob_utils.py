"""145# §9-#3 / §10.2-#1 / §10.1-#3: OB (orderbook) 正規化ユーティリティ.

OrderBookSnapshot.bids/asks は list[tuple[float, float]] だが、
一部モジュール (skip_gate_evaluator) では .price/.quantity アクセスをしていた。
tuple/object 両対応の安全な抽出関数を提供し、散在する dual-format ロジックを一元化する。

§10.1-#3: MarketDataAccessor — adapter を薄くラップし、
best_bid_ask / depth_volume / spread 等の市場データ取得を型安全に提供。
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, Optional, Protocol, Sequence

if TYPE_CHECKING:
    from ztb.trading.live.exchanges.base.broker_interfaces import OrderBookSnapshot

logger = logging.getLogger(__name__)


def extract_price(level: Any) -> float:
    """板レベルから price を抽出 (tuple / object 両対応)."""
    if isinstance(level, (list, tuple)):
        return float(level[0])
    return float(getattr(level, "price", 0.0))


def extract_size(level: Any) -> float:
    """板レベルから size (quantity) を抽出 (tuple / object 両対応)."""
    if isinstance(level, (list, tuple)):
        return float(level[1])
    return float(getattr(level, "quantity", getattr(level, "size", 0.0)))


def best_bid_ask(
    ob: Any,
) -> tuple[float | None, float | None]:
    """OrderBookSnapshot から best bid/ask を安全に抽出.

    Returns:
        (best_bid, best_ask) — データ不足時は None.
    """
    bid = extract_price(ob.bids[0]) if ob and ob.bids else None
    ask = extract_price(ob.asks[0]) if ob and ob.asks else None
    return bid, ask


def depth_volume(levels: Sequence[Any], depth: int = 5) -> float:
    """板の指定深さまでの合計出来高を計算."""
    return sum(extract_size(lv) for lv in levels[:depth])


# ---------------------------------------------------------------------------
# §10.1-#3: MarketDataAccessor — adapter wrapper
# ---------------------------------------------------------------------------

class _HasGetOrderbook(Protocol):
    """Protocol for adapter with get_orderbook method."""

    async def get_orderbook(self, symbol: str, depth: int = ...) -> Any: ...


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
    ) -> tuple[Optional[float], Optional[float]]:
        """Get best bid/ask via adapter, normalised through ob_utils.

        Returns:
            (best_bid, best_ask) — API 失敗時は (None, None).
        """
        try:
            ob = await self._adapter.get_orderbook(self._symbol, depth=depth)
            return best_bid_ask(ob)
        except Exception as e:
            logger.debug(f"MarketDataAccessor.best_bid_ask failed: {e}")
            return None, None

    async def spread(self, depth: int = 1) -> Optional[float]:
        """Best ask - best bid. None if data unavailable."""
        bid, ask = await self.best_bid_ask(depth)
        if bid is not None and ask is not None:
            return ask - bid
        return None

    async def mid_price(self, depth: int = 1) -> Optional[float]:
        """Mid-price = (bid + ask) / 2. None if data unavailable."""
        bid, ask = await self.best_bid_ask(depth)
        if bid is not None and ask is not None:
            return (bid + ask) / 2.0
        return None

    async def bid_depth_volume(self, depth: int = 5) -> float:
        """Bid-side depth volume up to *depth* levels."""
        try:
            ob = await self._adapter.get_orderbook(self._symbol, depth=depth)
            return depth_volume(ob.bids, depth) if ob and ob.bids else 0.0
        except Exception:
            return 0.0

    async def ask_depth_volume(self, depth: int = 5) -> float:
        """Ask-side depth volume up to *depth* levels."""
        try:
            ob = await self._adapter.get_orderbook(self._symbol, depth=depth)
            return depth_volume(ob.asks, depth) if ob and ob.asks else 0.0
        except Exception:
            return 0.0
