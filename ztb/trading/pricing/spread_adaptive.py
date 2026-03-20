from __future__ import annotations

import math
from dataclasses import dataclass

from ztb.trading.pricing.offset_math import scale_offset_ratio


@dataclass(frozen=True)
class SpreadAdaptiveResult:
    """Result of spread-adaptive offset adjustment."""

    updated_ratio: float
    applied_multiplier: float
    spread_bps: float | None
    mode: str


def apply_spread_adaptive_ratio(
    *,
    side: str,
    spread: float,
    mid_price: float,
    effective_offset_ratio: float,
    narrow_spread_bps: float,
    narrow_spread_boost: float,
    narrow_spread_boost_buy: float | None,
    narrow_spread_boost_sell: float | None,
    wide_spread_bps: float,
    wide_spread_ratio: float,
    min_ratio: float,
    max_ratio: float,
) -> SpreadAdaptiveResult:
    """Apply pure spread-adaptive ratio logic.

    The caller keeps ownership of logging and any sell-floor reapplication.
    """
    if mid_price <= 0 or not math.isfinite(mid_price) or not math.isfinite(spread):
        return SpreadAdaptiveResult(
            updated_ratio=effective_offset_ratio,
            applied_multiplier=1.0,
            spread_bps=None,
            mode="invalid",
        )

    spread_bps = spread / mid_price * 10_000.0
    if spread_bps < narrow_spread_bps:
        boost = narrow_spread_boost
        if side == "buy" and narrow_spread_boost_buy is not None:
            boost = narrow_spread_boost_buy
        elif side == "sell" and narrow_spread_boost_sell is not None:
            boost = narrow_spread_boost_sell
        updated_ratio, applied_multiplier = scale_offset_ratio(
            effective_offset_ratio,
            boost,
            max_ratio=max_ratio,
        )
        return SpreadAdaptiveResult(
            updated_ratio=updated_ratio,
            applied_multiplier=applied_multiplier,
            spread_bps=spread_bps,
            mode="narrow",
        )

    if spread_bps > wide_spread_bps:
        updated_ratio, applied_multiplier = scale_offset_ratio(
            effective_offset_ratio,
            wide_spread_ratio,
            min_ratio=min_ratio,
        )
        return SpreadAdaptiveResult(
            updated_ratio=updated_ratio,
            applied_multiplier=applied_multiplier,
            spread_bps=spread_bps,
            mode="wide",
        )

    return SpreadAdaptiveResult(
        updated_ratio=effective_offset_ratio,
        applied_multiplier=1.0,
        spread_bps=spread_bps,
        mode="none",
    )


__all__ = ["SpreadAdaptiveResult", "apply_spread_adaptive_ratio"]
