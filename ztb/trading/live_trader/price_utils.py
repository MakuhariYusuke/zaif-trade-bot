"""Price resolution helpers shared by LiveTrader and tests."""

from __future__ import annotations

import asyncio
from typing import Protocol, cast

from ztb.utils.errors import validate_price
from ztb.utils.logging_utils import get_logger

_FALLBACK_PRICE = 5_000_000.0


class _PriceAdapterLike(Protocol):
    async def get_current_price(self, symbol: str) -> float | None: ...


def resolve_current_price(
    *,
    exchange_adapter: object,
    last_valid_price: float,
    symbol: str = "btc_jpy",
) -> tuple[float, float]:
    """Resolve current price while preserving the previous valid fallback."""
    logger = get_logger(__name__)

    async def _async_get_price() -> float | None:
        if exchange_adapter is None:
            return None
        try:
            adapter = cast(_PriceAdapterLike, exchange_adapter)
            return await adapter.get_current_price(symbol)
        except Exception as e:
            logger.warning(f"Failed to get price from adapter: {e}")
            return None

    try:
        price = asyncio.run(_async_get_price())
        if price is not None:
            resolved = float(price)
            validate_price(resolved, "price")
            return resolved, resolved
    except Exception as e:
        logger.error(f"Failed to get current price: {e}")

    fallback = last_valid_price if last_valid_price > 0 else _FALLBACK_PRICE
    return fallback, last_valid_price
