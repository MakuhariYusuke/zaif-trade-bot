from __future__ import annotations

import math

import pytest

from ztb.trading.pricing.boost_math import decayed_loss_boost_multiplier
from ztb.trading.pricing.offset_math import (
    discounted_sell_offset_floor,
    effective_max_ratio,
    scale_offset_ratio,
)
from ztb.trading.pricing.price_finalization import finalize_price_with_spread_guard
from ztb.trading.pricing.spread_adaptive import apply_spread_adaptive_ratio


class TestPricingOffsetMathMigration:
    def test_effective_max_ratio_prefers_sell_ceiling_when_higher(self) -> None:
        assert effective_max_ratio(
            side="sell",
            base_ratio=0.30,
            sell_ceiling_ratio=0.45,
            buy_ceiling_ratio=None,
        ) == 0.45

    def test_effective_max_ratio_keeps_base_when_sell_ceiling_is_lower(self) -> None:
        assert effective_max_ratio(
            side="sell",
            base_ratio=0.30,
            sell_ceiling_ratio=0.20,
            buy_ceiling_ratio=None,
        ) == 0.30

    def test_scale_offset_ratio_applies_multiplier_and_clamp(self) -> None:
        updated, applied = scale_offset_ratio(
            0.10,
            2.0,
            max_ratio=0.15,
        )
        assert updated == pytest.approx(0.15)
        assert applied == pytest.approx(1.5)

    def test_scale_offset_ratio_returns_noop_for_non_positive_multiplier(self) -> None:
        updated, applied = scale_offset_ratio(0.10, 0.0)
        assert updated == 0.10
        assert applied == 1.0

    def test_discounted_sell_offset_floor_returns_discounted_value(self) -> None:
        assert discounted_sell_offset_floor(
            base_floor=0.20,
            bypass_threshold=0.30,
            inventory_imbalance=0.45,
            discount_ratio=0.5,
        ) == pytest.approx(0.10)

    def test_discounted_sell_offset_floor_keeps_base_below_threshold(self) -> None:
        assert discounted_sell_offset_floor(
            base_floor=0.20,
            bypass_threshold=0.30,
            inventory_imbalance=0.10,
            discount_ratio=0.5,
        ) == pytest.approx(0.20)

    def test_finalize_price_with_spread_guard_buy_cross_falls_back(self) -> None:
        result = finalize_price_with_spread_guard(
            side="buy",
            best_bid=1000.0,
            best_ask=1010.0,
            spread=10.0,
            offset=20.0,
            effective_offset_ratio=0.2,
        )
        assert result.price == 1000.0
        assert result.effective_offset_ratio == 0.0

    def test_finalize_price_with_spread_guard_sell_non_cross_keeps_offset(self) -> None:
        result = finalize_price_with_spread_guard(
            side="sell",
            best_bid=1000.0,
            best_ask=1010.0,
            spread=10.0,
            offset=3.0,
            effective_offset_ratio=0.2,
        )
        assert result.price == pytest.approx(1007.0)
        assert result.effective_offset_ratio == pytest.approx(0.2)

    def test_decayed_loss_boost_multiplier_decays_toward_one(self) -> None:
        assert decayed_loss_boost_multiplier(
            base_multiplier=1.5,
            elapsed_sec=100.0,
            tau_sec=100.0,
        ) == pytest.approx(1.0 + 0.5 * math.exp(-1.0))

    def test_decayed_loss_boost_multiplier_returns_raw_when_tau_invalid(self) -> None:
        assert decayed_loss_boost_multiplier(
            base_multiplier=1.5,
            elapsed_sec=100.0,
            tau_sec=0.0,
        ) == pytest.approx(1.5)

    def test_spread_adaptive_ratio_narrow_uses_side_specific_boost(self) -> None:
        result = apply_spread_adaptive_ratio(
            side="buy",
            spread=10.0,
            mid_price=10_000.0,
            effective_offset_ratio=0.05,
            narrow_spread_bps=20.0,
            narrow_spread_boost=1.5,
            narrow_spread_boost_buy=2.0,
            narrow_spread_boost_sell=None,
            wide_spread_bps=50.0,
            wide_spread_ratio=0.7,
            min_ratio=0.01,
            max_ratio=0.30,
        )
        assert result.mode == "narrow"
        assert result.updated_ratio == pytest.approx(0.10)
        assert result.applied_multiplier == pytest.approx(2.0)

    def test_spread_adaptive_ratio_invalid_mid_is_noop(self) -> None:
        result = apply_spread_adaptive_ratio(
            side="sell",
            spread=10.0,
            mid_price=0.0,
            effective_offset_ratio=0.05,
            narrow_spread_bps=20.0,
            narrow_spread_boost=1.5,
            narrow_spread_boost_buy=None,
            narrow_spread_boost_sell=None,
            wide_spread_bps=50.0,
            wide_spread_ratio=0.7,
            min_ratio=0.01,
            max_ratio=0.30,
        )
        assert result.mode == "invalid"
        assert result.updated_ratio == pytest.approx(0.05)
        assert result.spread_bps is None
