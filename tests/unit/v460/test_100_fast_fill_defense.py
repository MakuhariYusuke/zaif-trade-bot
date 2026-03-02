"""100# FastFillDefense unit tests.

God Object 分割 + P0/P1 修正の検証:
- side-aware boost (sell→buy 伝播防止)
- two-layer negative edge detection
- side-specific boost cap
- unfilled reset
"""

from __future__ import annotations

import pytest

from scripts.v460.lib.fast_fill_defense import (
    FastFillDefense,
    FastFillDefenseConfig,
)


class TestFastFillDefenseSideIsolation:
    """P0-5: sell fast-fill boost が buy に伝播しないこと."""

    def test_sell_boost_does_not_affect_buy(self) -> None:
        cfg = FastFillDefenseConfig(
            enabled=True, threshold_sec=5.0, offset_boost=2.0,
        )
        defense = FastFillDefense(cfg, base_offset_ratio=0.05)

        # sell fast-fill → sell boost activates
        defense.evaluate_fill(
            "sell", queue_wait_sec=3.0,
            fill_price=99_000, mid_at_fill=100_000,
        )
        assert defense.is_boost_active("sell")
        assert defense.get_boost_multiplier("sell") > 1.0

        # buy は影響を受けない
        assert not defense.is_boost_active("buy")
        assert defense.get_boost_multiplier("buy") == 1.0

    def test_buy_boost_does_not_affect_sell(self) -> None:
        cfg = FastFillDefenseConfig(
            enabled=True, threshold_sec=5.0, offset_boost=2.0,
        )
        defense = FastFillDefense(cfg, base_offset_ratio=0.05)

        defense.evaluate_fill(
            "buy", queue_wait_sec=3.0,
            fill_price=101_000, mid_at_fill=100_000,
        )
        assert defense.is_boost_active("buy")
        assert not defense.is_boost_active("sell")


class TestTwoLayerNegativeEdge:
    """P0-3: has_negative_edge の 50% 見逃し問題の修正."""

    def test_layer1_fill_price_vs_mid(self) -> None:
        """Layer 1: fill_price vs mid_at_fill."""
        cfg = FastFillDefenseConfig(enabled=True, threshold_sec=5.0, offset_boost=2.0)
        defense = FastFillDefense(cfg, base_offset_ratio=0.05)

        # buy: fill_price > mid → negative edge
        defense.evaluate_fill(
            "buy", queue_wait_sec=2.0,
            fill_price=100_500, mid_at_fill=100_000,
        )
        assert defense.is_boost_active("buy")

    def test_layer2_post_fill_pnl_negative(self) -> None:
        """Layer 2: post_fill_pnl_bps < -deadzone で検出 (L1 は non-negative)."""
        cfg = FastFillDefenseConfig(
            enabled=True, threshold_sec=5.0, offset_boost=2.0,
            l2_deadzone_bps=2.0,  # 230# H-1: deadzone 2bps
        )
        defense = FastFillDefense(cfg, base_offset_ratio=0.05)

        # sell: fill_price > mid (= non-negative edge by L1),
        # but post_fill_pnl is worse than -deadzone → L2 detects
        defense.evaluate_fill(
            "sell", queue_wait_sec=3.0,
            fill_price=100_500, mid_at_fill=100_000,
            post_fill_pnl_bps=-2.5,
        )
        assert defense.is_boost_active("sell")

    def test_no_negative_edge_no_boost(self) -> None:
        """正常約定 → boost 不発動."""
        cfg = FastFillDefenseConfig(enabled=True, threshold_sec=5.0, offset_boost=2.0)
        defense = FastFillDefense(cfg, base_offset_ratio=0.05)

        # buy: fill_price < mid → no negative edge
        defense.evaluate_fill(
            "buy", queue_wait_sec=2.0,
            fill_price=99_500, mid_at_fill=100_000,
        )
        assert not defense.is_boost_active("buy")


class TestSideSpecificBoostCap:
    """P1-2: boost cap が side 別 base_offset_ratio を使用すること."""

    def test_sell_cap_uses_sell_base(self) -> None:
        """sell base=0.12 → cap = 0.30/0.12 = 2.5."""
        cfg = FastFillDefenseConfig(
            enabled=True, threshold_sec=5.0,
            offset_boost=5.0,  # 意図的に大きな値
        )
        defense = FastFillDefense(
            cfg, base_offset_ratio=0.05,
            base_offset_ratio_sell=0.12,
        )

        defense.evaluate_fill(
            "sell", queue_wait_sec=2.0,
            fill_price=99_000, mid_at_fill=100_000,
        )
        # cap = 0.30 / 0.12 = 2.5
        assert defense.get_boost_multiplier("sell") == pytest.approx(2.5)

    def test_buy_cap_uses_buy_base_or_common(self) -> None:
        """buy base=None → common 0.05 → cap = 0.30/0.05 = 6.0."""
        cfg = FastFillDefenseConfig(
            enabled=True, threshold_sec=5.0,
            offset_boost=10.0,  # 意図的に大きな値
        )
        defense = FastFillDefense(
            cfg, base_offset_ratio=0.05,
        )

        defense.evaluate_fill(
            "buy", queue_wait_sec=2.0,
            fill_price=101_000, mid_at_fill=100_000,
        )
        # cap = 0.30 / 0.05 = 6.0
        assert defense.get_boost_multiplier("buy") == pytest.approx(6.0)

    def test_subunit_boost_is_clamped_to_one(self) -> None:
        """誤設定で boost<1 でも防御側が攻め方向に反転しない."""
        cfg = FastFillDefenseConfig(
            enabled=True,
            threshold_sec=5.0,
            offset_boost=0.5,
        )
        defense = FastFillDefense(cfg, base_offset_ratio=0.05)

        defense.evaluate_fill(
            "buy", queue_wait_sec=2.0,
            fill_price=101_000, mid_at_fill=100_000,
        )
        assert defense.get_boost_multiplier("buy") == pytest.approx(1.0)

    def test_zero_floor_misconfig_avoids_division_by_zero(self) -> None:
        """min_offset_ratio<=0 でも cap 計算が壊れない."""
        cfg = FastFillDefenseConfig(
            min_offset_ratio=0.0,
            max_offset_ratio=0.0,
        )
        defense = FastFillDefense(cfg, base_offset_ratio=0.0)

        assert defense._compute_capped_multiplier("sell", 5.0) == pytest.approx(1.0)


class TestSideSpecificThreshold:
    """P0-4: threshold_sec_buy=10.0 で buy defense が有効になること."""

    def test_buy_threshold_10s(self) -> None:
        cfg = FastFillDefenseConfig(
            enabled=True,
            threshold_sec=5.0,
            threshold_sec_buy=10.0,
            offset_boost=2.0,
        )
        defense = FastFillDefense(cfg, base_offset_ratio=0.05)

        # 7s は共通5s超だが buy固有10s以下 → fast fill 判定
        defense.evaluate_fill(
            "buy", queue_wait_sec=7.0,
            fill_price=101_000, mid_at_fill=100_000,
        )
        assert defense.is_boost_active("buy")

    def test_sell_uses_common_threshold_when_null(self) -> None:
        cfg = FastFillDefenseConfig(
            enabled=True,
            threshold_sec=5.0,
            threshold_sec_buy=10.0,
            threshold_sec_sell=None,  # common fallback
            offset_boost=2.0,
        )
        defense = FastFillDefense(cfg, base_offset_ratio=0.05)

        # 7s > common 5s → not fast fill for sell
        defense.evaluate_fill(
            "sell", queue_wait_sec=7.0,
            fill_price=99_000, mid_at_fill=100_000,
        )
        assert not defense.is_boost_active("sell")


class TestResetOnUnfilled:
    """096# unfilled 時のブースト永続化防止."""

    def test_unfilled_resets_boost(self) -> None:
        cfg = FastFillDefenseConfig(enabled=True, threshold_sec=5.0, offset_boost=2.0)
        defense = FastFillDefense(cfg, base_offset_ratio=0.05)

        defense.evaluate_fill(
            "buy", queue_wait_sec=2.0,
            fill_price=101_000, mid_at_fill=100_000,
        )
        assert defense.is_boost_active("buy")

        defense.reset_on_unfilled("buy")
        assert not defense.is_boost_active("buy")
        assert defense.get_boost_multiplier("buy") == 1.0


class TestBoostDeactivation:
    """正常約定でブースト解除."""

    def test_normal_fill_deactivates(self) -> None:
        cfg = FastFillDefenseConfig(
            enabled=True, threshold_sec=5.0, offset_boost=2.0,
            boost_release_streak=1,  # 230# H-2: 1 回で即解除
        )
        defense = FastFillDefense(cfg, base_offset_ratio=0.05)

        # 1st: fast + neg edge → activate
        defense.evaluate_fill(
            "buy", queue_wait_sec=2.0,
            fill_price=101_000, mid_at_fill=100_000,
        )
        assert defense.is_boost_active("buy")

        # 2nd: normal fill → deactivate (streak=1)
        defense.evaluate_fill(
            "buy", queue_wait_sec=60.0,
            fill_price=99_500, mid_at_fill=100_000,
        )
        assert not defense.is_boost_active("buy")
        assert defense.get_boost_multiplier("buy") == 1.0


class TestDisabledConfig:
    """enabled=False で何もしない."""

    def test_disabled_no_boost(self) -> None:
        cfg = FastFillDefenseConfig(enabled=False)
        defense = FastFillDefense(cfg, base_offset_ratio=0.05)

        defense.evaluate_fill(
            "sell", queue_wait_sec=1.0,
            fill_price=99_000, mid_at_fill=100_000,
        )
        assert not defense.is_boost_active("sell")
        assert defense.get_boost_multiplier("sell") == 1.0


class TestUpdateBaseOffsets:
    """param_adapter 更新時の同期."""

    def test_update_base_offsets(self) -> None:
        cfg = FastFillDefenseConfig(enabled=True, threshold_sec=5.0, offset_boost=5.0)
        defense = FastFillDefense(cfg, base_offset_ratio=0.05)

        defense.update_base_offsets(0.08, buy=0.06, sell=0.15)
        assert defense._base_offset_ratio == 0.08
        assert defense._base_offset_ratio_buy == 0.06
        assert defense._base_offset_ratio_sell == 0.15
        # cap recalculation: sell cap = 0.30/0.15 = 2.0
        assert defense._compute_capped_multiplier("sell", 5.0) == pytest.approx(2.0)
