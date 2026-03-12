"""346# pre_order_adjustments.py テスト.

323# で分離された PreOrderAdjustmentsMixin の2メソッドをテスト。

- _recalc_price_with_new_offset: offset 変更後の maker 価格再計算
- _apply_offset_multiplier: offset 倍率適用 (保守的/アグレッシブ)
"""

from __future__ import annotations

import pytest

from scripts.v460.lib.pre_order_adjustments import PreOrderAdjustmentsMixin


# ============================================================
# _recalc_price_with_new_offset
# ============================================================


class TestRecalcPriceWithNewOffset:
    """offset 変更後の価格再計算テスト."""

    def test_buy_price_recalculated(self) -> None:
        """buy: mid 逆推定 → 新 offset で再計算."""
        # mid = 10000 + 100 * 0.5 / 2 = 10025
        # new_price = 10025 - 100 * 0.8 / 2 = 9985
        result = PreOrderAdjustmentsMixin._recalc_price_with_new_offset(
            side="buy",
            order_price=10000,
            spread_at_order=100,
            old_ratio=0.5,
            new_ratio=0.8,
        )
        assert result == round(10025 - 100 * 0.8 / 2)

    def test_sell_price_recalculated(self) -> None:
        """sell: mid 逆推定 → 新 offset で再計算."""
        # mid = 10000 - 100 * 0.5 / 2 = 9975
        # new_price = 9975 + 100 * 0.8 / 2 = 10015
        result = PreOrderAdjustmentsMixin._recalc_price_with_new_offset(
            side="sell",
            order_price=10000,
            spread_at_order=100,
            old_ratio=0.5,
            new_ratio=0.8,
        )
        assert result == round(9975 + 100 * 0.8 / 2)

    def test_spread_none_returns_original(self) -> None:
        """spread=None → order_price そのまま返却."""
        result = PreOrderAdjustmentsMixin._recalc_price_with_new_offset(
            side="buy", order_price=10000, spread_at_order=None,
            old_ratio=0.5, new_ratio=0.8,
        )
        assert result == 10000

    def test_spread_zero_returns_original(self) -> None:
        """spread=0 → order_price そのまま返却."""
        result = PreOrderAdjustmentsMixin._recalc_price_with_new_offset(
            side="sell", order_price=10000, spread_at_order=0,
            old_ratio=0.5, new_ratio=0.8,
        )
        assert result == 10000

    def test_same_ratio_returns_original(self) -> None:
        """old_ratio == new_ratio → 元の価格に戻る."""
        result = PreOrderAdjustmentsMixin._recalc_price_with_new_offset(
            side="buy", order_price=10000, spread_at_order=100,
            old_ratio=0.5, new_ratio=0.5,
        )
        assert result == 10000

    def test_negative_spread_returns_original(self) -> None:
        """spread < 0 → order_price そのまま返却."""
        result = PreOrderAdjustmentsMixin._recalc_price_with_new_offset(
            side="buy", order_price=10000, spread_at_order=-10,
            old_ratio=0.5, new_ratio=0.8,
        )
        assert result == 10000


# ============================================================
# _apply_offset_multiplier
# ============================================================


class TestApplyOffsetMultiplier:
    """offset 倍率適用テスト."""

    def test_none_multiplier_noop(self) -> None:
        """offset_mult=None → no-op."""
        price, ratio, mult, delta = PreOrderAdjustmentsMixin._apply_offset_multiplier(
            side="buy", order_price=10000, spread_at_order=100,
            effective_offset_ratio=0.5, offset_mult=None,
        )
        assert price == 10000
        assert ratio == 0.5
        assert mult is None
        assert delta is None

    def test_zero_multiplier_noop(self) -> None:
        """offset_mult=0 → no-op."""
        price, _, _, _ = PreOrderAdjustmentsMixin._apply_offset_multiplier(
            side="buy", order_price=10000, spread_at_order=100,
            effective_offset_ratio=0.5, offset_mult=0.0,
        )
        assert price == 10000

    def test_one_multiplier_noop(self) -> None:
        """offset_mult=1.0 → no-op."""
        price, ratio, mult, delta = PreOrderAdjustmentsMixin._apply_offset_multiplier(
            side="sell", order_price=10000, spread_at_order=100,
            effective_offset_ratio=0.5, offset_mult=1.0,
        )
        assert price == 10000
        assert ratio == 0.5
        assert mult is None

    def test_spread_none_noop(self) -> None:
        """spread=None → no-op."""
        price, _, _, _ = PreOrderAdjustmentsMixin._apply_offset_multiplier(
            side="buy", order_price=10000, spread_at_order=None,
            effective_offset_ratio=0.5, offset_mult=1.5,
        )
        assert price == 10000

    def test_price_zero_noop(self) -> None:
        """order_price=0 → no-op."""
        price, _, _, _ = PreOrderAdjustmentsMixin._apply_offset_multiplier(
            side="buy", order_price=0, spread_at_order=100,
            effective_offset_ratio=0.5, offset_mult=1.5,
        )
        assert price == 0

    # --- conservative (default) ---

    def test_conservative_buy_gt_one_widens(self) -> None:
        """保守的 (default): mult>1 → buy 価格を mid から遠ざける (下げる)."""
        # old_offset = 100 * 0.5 = 50, new_offset = 75, delta = 25
        # buy conservative: price - delta = 10000 - 25 = 9975
        price, ratio, mult, delta = PreOrderAdjustmentsMixin._apply_offset_multiplier(
            side="buy", order_price=10000, spread_at_order=100,
            effective_offset_ratio=0.5, offset_mult=1.5,
        )
        assert price == 9975
        assert ratio == pytest.approx(0.75)
        assert mult == 1.5
        assert delta == pytest.approx(25.0)

    def test_conservative_sell_gt_one_widens(self) -> None:
        """保守的: mult>1 → sell 価格を mid から遠ざける (上げる)."""
        price, _, _, _ = PreOrderAdjustmentsMixin._apply_offset_multiplier(
            side="sell", order_price=10000, spread_at_order=100,
            effective_offset_ratio=0.5, offset_mult=1.5,
        )
        assert price == 10025  # 10000 + 25

    def test_conservative_lt_one_is_noop(self) -> None:
        """保守的: mult<1 → no-op (aggressive_when_multiplier_gt_one=False)."""
        price, ratio, mult, delta = PreOrderAdjustmentsMixin._apply_offset_multiplier(
            side="buy", order_price=10000, spread_at_order=100,
            effective_offset_ratio=0.5, offset_mult=0.8,
        )
        assert price == 10000
        assert ratio == 0.5
        assert mult is None

    # --- aggressive ---

    def test_aggressive_buy_gt_one_narrows(self) -> None:
        """アグレッシブ: mult>1 → buy 価格を mid に近づける (上げる)."""
        price, ratio, _, _ = PreOrderAdjustmentsMixin._apply_offset_multiplier(
            side="buy", order_price=10000, spread_at_order=100,
            effective_offset_ratio=0.5, offset_mult=1.5,
            aggressive_when_multiplier_gt_one=True,
        )
        assert price == 10025  # 10000 + 25

    def test_aggressive_sell_gt_one_narrows(self) -> None:
        """アグレッシブ: mult>1 → sell 価格を mid に近づける (下げる)."""
        price, _, _, _ = PreOrderAdjustmentsMixin._apply_offset_multiplier(
            side="sell", order_price=10000, spread_at_order=100,
            effective_offset_ratio=0.5, offset_mult=1.5,
            aggressive_when_multiplier_gt_one=True,
        )
        assert price == 9975  # 10000 - 25

    def test_aggressive_lt_one_also_applied(self) -> None:
        """アグレッシブ: mult<1 も適用される."""
        # old_offset = 50, new_offset = 40, delta = -10
        # buy aggressive: price + delta = 10000 + (-10) = 9990
        price, ratio, mult, delta = PreOrderAdjustmentsMixin._apply_offset_multiplier(
            side="buy", order_price=10000, spread_at_order=100,
            effective_offset_ratio=0.5, offset_mult=0.8,
            aggressive_when_multiplier_gt_one=True,
        )
        assert price == 9990
        assert ratio == pytest.approx(0.4)
        assert delta == pytest.approx(-10.0)

    def test_ratio_updated_correctly(self) -> None:
        """effective_offset_ratio が倍率で更新されること."""
        _, ratio, _, _ = PreOrderAdjustmentsMixin._apply_offset_multiplier(
            side="sell", order_price=10000, spread_at_order=100,
            effective_offset_ratio=0.5, offset_mult=2.0,
        )
        assert ratio == pytest.approx(1.0)
