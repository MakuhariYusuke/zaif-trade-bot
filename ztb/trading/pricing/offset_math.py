from __future__ import annotations


def effective_max_ratio(
    *,
    side: str,
    base_ratio: float,
    sell_ceiling_ratio: float | None,
    buy_ceiling_ratio: float | None,
) -> float:
    """Resolve the intermediate max offset ratio for the given side.

    The final quote clamp can still apply a tighter side-specific ceiling later.
    This helper only determines how far intermediate stages are allowed to explore.
    """
    if side == "sell" and sell_ceiling_ratio is not None:
        return max(base_ratio, sell_ceiling_ratio)
    if side == "buy" and buy_ceiling_ratio is not None:
        return max(base_ratio, buy_ceiling_ratio)
    return base_ratio


def scale_offset_ratio(
    effective_offset_ratio: float,
    multiplier: float,
    *,
    min_ratio: float | None = None,
    max_ratio: float | None = None,
) -> tuple[float, float]:
    """Apply a multiplier to an offset ratio with safe clamping.

    Returns the updated ratio and the actual applied multiplier.
    """
    if effective_offset_ratio <= 0 or multiplier <= 0:
        return effective_offset_ratio, 1.0

    updated = effective_offset_ratio * multiplier
    if min_ratio is not None:
        updated = max(updated, min_ratio)
    if max_ratio is not None:
        updated = min(updated, max_ratio)

    applied = updated / effective_offset_ratio if effective_offset_ratio != 0 else 1.0
    return updated, applied


def discounted_sell_offset_floor(
    *,
    base_floor: float,
    bypass_threshold: float,
    inventory_imbalance: float,
    discount_ratio: float,
) -> float:
    """Resolve the dynamic sell offset floor under inventory skew.

    When inventory is buy-heavy enough, the sell floor is discounted so that
    inventory-reduction logic is not blocked by a static lower bound.
    """
    if base_floor <= 0:
        return 0.0
    if bypass_threshold > 0 and inventory_imbalance >= bypass_threshold:
        return base_floor * discount_ratio
    return base_floor


__all__ = ["discounted_sell_offset_floor", "effective_max_ratio", "scale_offset_ratio"]
