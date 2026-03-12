"""Lot/position sizing helpers for fill_test.

153# task B (initial split): Extract per-cycle lot math from FillTestRunner.
The helpers are side-effect free so they can be reused in tests and other
execution runners without depending on runner internals.
"""

from __future__ import annotations

import math
from typing import Mapping


def resolve_regime_lot_multiplier(
    multipliers: Mapping[str, float] | None,
    *,
    regime_value: str | None,
) -> float:
    """Resolve regime-specific multiplier with safe defaults."""
    if not multipliers or not regime_value:
        return 1.0
    mult = multipliers.get(regime_value, 1.0)
    if not math.isfinite(mult) or mult <= 0:
        return 1.0
    return mult


def scale_lot_by_regime(
    base_lot: float,
    *,
    multiplier: float,
    min_lot: float,
    max_lot: float,
) -> float:
    """Apply regime multiplier, then clamp to min/max bounds."""
    if not math.isfinite(base_lot) or base_lot <= 0:
        return _clamp_lot(min_lot, min_lot=min_lot, max_lot=max_lot)
    adjusted = base_lot * multiplier
    return _clamp_lot(adjusted, min_lot=min_lot, max_lot=max_lot)


def compute_confidence_lot_factor(
    *,
    enabled: bool,
    mode: str,
    as_prob: float | None,
    scale: float,
    floor: float,
    dust_sweep_active: bool = False,
) -> float:
    """Compute confidence-based lot shrink factor.

    Returns a factor in [0, 1]. For disabled mode, unsupported mode, missing or
    non-finite probability inputs, returns 1.0.
    """
    if not enabled or dust_sweep_active:
        return 1.0
    if mode == "pnl":
        return 1.0
    if as_prob is None or not math.isfinite(as_prob):
        return 1.0

    scale_safe = scale if math.isfinite(scale) else 0.0
    floor_safe = floor if math.isfinite(floor) else 0.0
    floor_safe = max(0.0, min(1.0, floor_safe))
    raw = 1.0 - scale_safe * as_prob
    return max(floor_safe, min(1.0, max(0.0, raw)))


def compute_effective_order_lot(
    *,
    regime_lot: float,
    confidence_factor: float,
    min_lot: float,
    max_lot: float,
) -> float:
    """Combine regime lot and confidence factor, then clamp to bounds."""
    lot = regime_lot * confidence_factor
    return _clamp_lot(lot, min_lot=min_lot, max_lot=max_lot)


def _clamp_lot(value: float, *, min_lot: float, max_lot: float) -> float:
    """Clamp to [min_lot, max_lot] (max_lot<=0 means no upper bound)."""
    if not math.isfinite(value):
        clamped = min_lot
    else:
        clamped = max(value, min_lot)
    if max_lot > 0:
        clamped = min(clamped, max_lot)
    return clamped

