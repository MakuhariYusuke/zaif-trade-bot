"""lot_manager helper tests (153# task B initial split)."""

from __future__ import annotations

import pytest

from scripts.v460.lib.lot_manager import (
    compute_confidence_lot_factor,
    compute_effective_order_lot,
    resolve_regime_lot_multiplier,
    scale_lot_by_regime,
)


def test_resolve_regime_lot_multiplier_defaults_to_one() -> None:
    assert resolve_regime_lot_multiplier({}, regime_value="high_vol") == 1.0
    assert resolve_regime_lot_multiplier({"high_vol": 0.7}, regime_value=None) == 1.0
    assert resolve_regime_lot_multiplier({"high_vol": 0.7}, regime_value="unknown") == 1.0


def test_resolve_regime_lot_multiplier_resolves_known_value() -> None:
    assert resolve_regime_lot_multiplier({"high_vol": 0.7}, regime_value="high_vol") == 0.7


def test_scale_lot_by_regime_clamps_min_and_max() -> None:
    # min clamp
    assert scale_lot_by_regime(0.002, multiplier=0.2, min_lot=0.001, max_lot=0.01) == 0.001
    # max clamp
    assert scale_lot_by_regime(0.01, multiplier=2.0, min_lot=0.001, max_lot=0.01) == 0.01


def test_compute_confidence_lot_factor_shrink_only() -> None:
    factor = compute_confidence_lot_factor(
        enabled=True,
        mode="as",
        as_prob=0.5,
        scale=1.0,
        floor=0.3,
    )
    assert factor == pytest.approx(0.5)


def test_compute_confidence_lot_factor_returns_one_for_disabled_or_frozen_mode() -> None:
    assert compute_confidence_lot_factor(
        enabled=False,
        mode="as",
        as_prob=0.9,
        scale=1.0,
        floor=0.3,
    ) == 1.0
    assert compute_confidence_lot_factor(
        enabled=True,
        mode="pnl",
        as_prob=0.9,
        scale=1.0,
        floor=0.3,
    ) == 1.0


def test_compute_effective_order_lot_clamps_bounds() -> None:
    assert compute_effective_order_lot(
        regime_lot=0.003,
        confidence_factor=0.5,
        min_lot=0.001,
        max_lot=0.01,
    ) == pytest.approx(0.0015)
    assert compute_effective_order_lot(
        regime_lot=0.001,
        confidence_factor=0.1,
        min_lot=0.001,
        max_lot=0.01,
    ) == 0.001
