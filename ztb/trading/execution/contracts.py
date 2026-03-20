"""Shared execution contracts used by order monitoring modules."""

from __future__ import annotations

from typing import Protocol, runtime_checkable


@runtime_checkable
class OrderLike(Protocol):
    """Minimal order object contract."""

    @property
    def order_id(self) -> str: ...


@runtime_checkable
class OrderStatusLike(Protocol):
    """Minimal exchange order status contract."""

    @property
    def status(self) -> str: ...

    @property
    def price(self) -> float | None: ...


class ExchangeAdapter(Protocol):
    """Adapter methods required by order monitoring."""

    async def get_order_status(self, order_id: str) -> OrderStatusLike | None: ...
    async def cancel_order(self, order_id: str) -> None: ...
    async def place_order(
        self,
        symbol: str,
        side: str,
        quantity: float,
        price: float,
        order_type: str = "limit",
    ) -> OrderLike: ...
    async def get_orderbook(self, symbol: str, depth: int = 1) -> object: ...


__all__ = ["OrderLike", "OrderStatusLike", "ExchangeAdapter"]
