"""Shared pure helpers for finalizing maker quote prices."""

from __future__ import annotations

import logging

from ztb.trading.pricing.contracts import MakerPriceResult


logger = logging.getLogger(__name__)


def finalize_price_with_spread_guard(
    *,
    side: str,
    best_bid: float,
    best_ask: float,
    spread: float,
    offset: float,
    effective_offset_ratio: float,
) -> MakerPriceResult:
    """Build the final maker price and guard against crossing the spread."""
    if side == "buy":
        price = best_bid + offset
        if price >= best_ask:
            logger.info(
                "Spread guard: buy price %.0f >= ask %.0f, fallback to best_bid %.0f "
                "(spread=%.0f)",
                price,
                best_ask,
                best_bid,
                spread,
            )
            return MakerPriceResult(best_bid, spread, 0.0)
        return MakerPriceResult(price, spread, effective_offset_ratio)

    price = best_ask - offset
    if price <= best_bid:
        logger.info(
            "Spread guard: sell price %.0f <= bid %.0f, fallback to best_ask %.0f "
            "(spread=%.0f)",
            price,
            best_bid,
            best_ask,
            spread,
        )
        return MakerPriceResult(best_ask, spread, 0.0)
    return MakerPriceResult(price, spread, effective_offset_ratio)


__all__ = ["finalize_price_with_spread_guard"]
