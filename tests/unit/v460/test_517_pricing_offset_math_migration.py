from __future__ import annotations

import pytest

from ztb.trading.pricing.offset_math import effective_max_ratio, scale_offset_ratio


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
