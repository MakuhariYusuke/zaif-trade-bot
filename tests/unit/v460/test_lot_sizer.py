"""Tests for 方策 B: 動的ロットサイジング (lot_sizer)."""

from __future__ import annotations

import pytest

from ztb.trading.sizing.lot_sizer import (
    LotSizingConfig,
    LotSizingResult,
    clamp_lot,
    compute_cumulative_pnl_jpy,
    compute_lot_size,
    compute_recent_pnl_bps,
)
from ztb.metrics.fill_quality import FillRecord


# ======================================================================
# compute_lot_size
# ======================================================================

class TestComputeLotSize:
    """compute_lot_size のテスト."""

    def _default_config(self, **overrides: object) -> LotSizingConfig:
        defaults = dict(
            current_lot=0.001,
            min_lot=0.001,
            max_lot=0.005,
            lot_step=0.001,
            min_fill_rate=0.70,
            max_as_ratio=0.30,
            min_recent_pnl_bps=0.0,
            loss_cap_jpy=10_000.0,
            loss_cap_warning_ratio=0.7,
            min_samples=50,
        )
        defaults.update(overrides)
        return LotSizingConfig(**defaults)

    def test_increase_when_all_conditions_good(self) -> None:
        """全条件クリア → 1段階増量."""
        config = self._default_config(current_lot=0.001)
        result = compute_lot_size(
            fill_rate=0.80,
            as_ratio=0.10,
            recent_pnl_bps=0.5,
            cumulative_pnl_jpy=100.0,
            sample_count=100,
            config=config,
        )
        assert result.action == "increase"
        assert result.new_lot == 0.002
        assert result.changed

    def test_hold_at_max_lot(self) -> None:
        """上限到達時 → hold."""
        config = self._default_config(current_lot=0.005)
        result = compute_lot_size(
            fill_rate=0.80,
            as_ratio=0.10,
            recent_pnl_bps=0.5,
            cumulative_pnl_jpy=100.0,
            sample_count=100,
            config=config,
        )
        assert result.action == "hold"
        assert result.new_lot == 0.005
        assert not result.changed

    def test_decrease_low_fill_rate(self) -> None:
        """fill_rate 低下 → 減量."""
        config = self._default_config(current_lot=0.003)
        result = compute_lot_size(
            fill_rate=0.50,
            as_ratio=0.10,
            recent_pnl_bps=0.5,
            cumulative_pnl_jpy=100.0,
            sample_count=100,
            config=config,
        )
        assert result.action == "decrease"
        assert result.new_lot == 0.002

    def test_decrease_high_as_ratio(self) -> None:
        """AS 超過 → 減量."""
        config = self._default_config(current_lot=0.003)
        result = compute_lot_size(
            fill_rate=0.80,
            as_ratio=0.40,
            recent_pnl_bps=0.5,
            cumulative_pnl_jpy=100.0,
            sample_count=100,
            config=config,
        )
        assert result.action == "decrease"
        assert result.new_lot == 0.002

    def test_decrease_negative_pnl(self) -> None:
        """PnL マイナス → 減量."""
        config = self._default_config(current_lot=0.002)
        result = compute_lot_size(
            fill_rate=0.80,
            as_ratio=0.10,
            recent_pnl_bps=-1.0,
            cumulative_pnl_jpy=50.0,
            sample_count=100,
            config=config,
        )
        assert result.action == "decrease"
        assert result.new_lot == 0.001

    def test_hold_at_min_lot_when_bad(self) -> None:
        """最小ロットで条件未達 → hold (これ以上減らせない)."""
        config = self._default_config(current_lot=0.001)
        result = compute_lot_size(
            fill_rate=0.50,
            as_ratio=0.40,
            recent_pnl_bps=-2.0,
            cumulative_pnl_jpy=-5000.0,
            sample_count=100,
            config=config,
        )
        assert result.action == "hold"
        assert result.new_lot == 0.001
        assert not result.changed

    def test_cap_shrink(self) -> None:
        """損失キャップ接近 → 強制最小ロット."""
        config = self._default_config(current_lot=0.003)
        result = compute_lot_size(
            fill_rate=0.80,
            as_ratio=0.10,
            recent_pnl_bps=0.5,
            cumulative_pnl_jpy=-7500.0,  # -10000 * 0.7 = -7000 を超過
            sample_count=100,
            config=config,
        )
        assert result.action == "cap_shrink"
        assert result.new_lot == 0.001

    def test_cap_shrink_overrides_good_conditions(self) -> None:
        """損失キャップは全条件クリアでも最優先."""
        config = self._default_config(current_lot=0.005)
        result = compute_lot_size(
            fill_rate=0.90,
            as_ratio=0.05,
            recent_pnl_bps=5.0,
            cumulative_pnl_jpy=-8000.0,
            sample_count=200,
            config=config,
        )
        assert result.action == "cap_shrink"
        assert result.new_lot == 0.001

    def test_hold_insufficient_samples(self) -> None:
        """サンプル不足 → hold."""
        config = self._default_config(current_lot=0.001)
        result = compute_lot_size(
            fill_rate=0.90,
            as_ratio=0.05,
            recent_pnl_bps=5.0,
            cumulative_pnl_jpy=500.0,
            sample_count=30,
            config=config,
        )
        assert result.action == "hold"
        assert result.new_lot == 0.001
        assert "サンプル不足" in result.reason

    def test_default_config(self) -> None:
        """設定なし (デフォルト) でも動作する. 348# lot_step=1satoshi."""
        result = compute_lot_size(
            fill_rate=0.80,
            as_ratio=0.10,
            recent_pnl_bps=0.5,
            cumulative_pnl_jpy=100.0,
            sample_count=100,
        )
        assert result.action == "increase"
        # 348# lot_step=0.00000001 (satoshi): 0.001 + 1e-8
        assert result.new_lot == pytest.approx(0.00100001)

    def test_step_increments(self) -> None:
        """複数回連続で増量すると段階的に上がる."""
        config = self._default_config(current_lot=0.001)
        for expected in [0.002, 0.003, 0.004, 0.005]:
            result = compute_lot_size(
                fill_rate=0.80,
                as_ratio=0.10,
                recent_pnl_bps=0.5,
                cumulative_pnl_jpy=100.0,
                sample_count=100,
                config=config,
            )
            assert result.new_lot == expected
            config = self._default_config(current_lot=result.new_lot)


# ======================================================================
# clamp_lot
# ======================================================================

class TestClampLot:
    """clamp_lot のテスト."""

    def test_clamp_below_min(self) -> None:
        config = LotSizingConfig(min_lot=0.001, max_lot=0.005)
        assert clamp_lot(0.0001, config) == 0.001

    def test_clamp_above_max(self) -> None:
        config = LotSizingConfig(min_lot=0.001, max_lot=0.005)
        assert clamp_lot(0.01, config) == 0.005

    def test_clamp_within_range(self) -> None:
        config = LotSizingConfig(min_lot=0.001, max_lot=0.005)
        assert clamp_lot(0.003, config) == 0.003

    def test_clamp_rounding(self) -> None:
        """浮動小数点の丸めが正しい. 348# satoshi 精度 (8桁)."""
        config = LotSizingConfig(min_lot=0.001, max_lot=0.005)
        assert clamp_lot(0.00299999, config) == 0.00299999
        # 9桁以上は丸められる
        assert clamp_lot(0.001000005, config) == 0.001

    def test_default_config(self) -> None:
        """デフォルト設定でも動作."""
        assert clamp_lot(0.003) == 0.003


# ======================================================================
# compute_cumulative_pnl_jpy
# ======================================================================

class TestComputeCumulativePnlJpy:
    """compute_cumulative_pnl_jpy のテスト."""

    def _make_record(
        self,
        pnl_bps: float | None = None,
        fill_price: float = 10_000_000.0,
        quantity: float = 0.001,
        filled: bool = True,
    ) -> FillRecord:
        return FillRecord(
            cycle_id="test",
            timestamp=1000000.0,
            side="buy",
            order_price=fill_price,
            order_quantity=quantity,
            fill_price=fill_price if filled else None,
            filled=filled,
            post_fill_30s_pnl=pnl_bps,
        )

    def test_positive_pnl(self) -> None:
        records = [self._make_record(pnl_bps=1.0)]
        # 1 bps * 1e-4 * 10M * 0.001 = 1.0 JPY
        result = compute_cumulative_pnl_jpy(records)
        assert abs(result - 1.0) < 0.01

    def test_negative_pnl(self) -> None:
        records = [self._make_record(pnl_bps=-2.0)]
        result = compute_cumulative_pnl_jpy(records)
        assert abs(result - (-2.0)) < 0.01

    def test_mixed_records(self) -> None:
        records = [
            self._make_record(pnl_bps=1.0),
            self._make_record(pnl_bps=-0.5),
            self._make_record(pnl_bps=None),  # PnL なし → スキップ
            self._make_record(filled=False),   # 未約定 → スキップ
        ]
        result = compute_cumulative_pnl_jpy(records)
        assert abs(result - 0.5) < 0.01

    def test_empty_records(self) -> None:
        assert compute_cumulative_pnl_jpy([]) == 0.0

    def test_different_quantities(self) -> None:
        """ロットが異なるレコードを正しく重み付け."""
        records = [
            self._make_record(pnl_bps=1.0, quantity=0.001),  # 1.0 JPY
            self._make_record(pnl_bps=1.0, quantity=0.002),  # 2.0 JPY
        ]
        result = compute_cumulative_pnl_jpy(records)
        assert abs(result - 3.0) < 0.01


# ======================================================================
# compute_recent_pnl_bps
# ======================================================================

class TestComputeRecentPnlBps:
    """compute_recent_pnl_bps のテスト."""

    def _make_record(
        self,
        pnl_bps: float | None = None,
        filled: bool = True,
    ) -> FillRecord:
        return FillRecord(
            cycle_id="test",
            timestamp=1000000.0,
            side="buy",
            order_price=10_000_000.0,
            order_quantity=0.001,
            filled=filled,
            post_fill_30s_pnl=pnl_bps,
        )

    def test_recent_window(self) -> None:
        """window=3 で直近 3 件のみ使う."""
        records = [
            self._make_record(pnl_bps=-10.0),  # 古い → 含まない
            self._make_record(pnl_bps=1.0),
            self._make_record(pnl_bps=2.0),
            self._make_record(pnl_bps=3.0),
        ]
        result = compute_recent_pnl_bps(records, window=3)
        assert abs(result - 2.0) < 0.01

    def test_window_larger_than_records(self) -> None:
        """レコードが window より少ない → 全件使う."""
        records = [
            self._make_record(pnl_bps=1.0),
            self._make_record(pnl_bps=3.0),
        ]
        result = compute_recent_pnl_bps(records, window=50)
        assert abs(result - 2.0) < 0.01

    def test_skip_unfilled(self) -> None:
        """未約定レコードは除外."""
        records = [
            self._make_record(pnl_bps=1.0),
            self._make_record(filled=False),
            self._make_record(pnl_bps=3.0),
        ]
        result = compute_recent_pnl_bps(records, window=50)
        assert abs(result - 2.0) < 0.01

    def test_empty_records(self) -> None:
        assert compute_recent_pnl_bps([]) == 0.0

    def test_no_filled_records(self) -> None:
        records = [self._make_record(filled=False)]
        assert compute_recent_pnl_bps(records) == 0.0


# ======================================================================
# LotSizingResult.changed
# ======================================================================

class TestLotSizingResult:
    """LotSizingResult のテスト."""

    def test_changed_true(self) -> None:
        result = LotSizingResult(
            previous_lot=0.001,
            new_lot=0.002,
            action="increase",
            reason="test",
            fill_rate=0.8,
            as_ratio=0.1,
            recent_pnl_bps=0.5,
            cumulative_pnl_jpy=100.0,
            sample_count=100,
        )
        assert result.changed

    def test_changed_false(self) -> None:
        result = LotSizingResult(
            previous_lot=0.001,
            new_lot=0.001,
            action="hold",
            reason="test",
            fill_rate=0.8,
            as_ratio=0.1,
            recent_pnl_bps=0.5,
            cumulative_pnl_jpy=100.0,
            sample_count=100,
        )
        assert not result.changed
