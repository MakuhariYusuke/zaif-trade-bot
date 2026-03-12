"""264# Kelly Criterion lot sizing テスト."""

from __future__ import annotations

import math
from dataclasses import dataclass

import pytest

from scripts.v460.lib.lot_sizer import (
    KellyEstimate,
    LotSizingConfig,
    clamp_lot,
    compute_kelly_fraction,
    compute_lot_size,
    kelly_recommended_lot,
)


# ------------------------------------------------------------------
#  FillRecord モック
# ------------------------------------------------------------------
@dataclass
class _MockFillRecord:
    """テスト用 FillRecord 互換オブジェクト."""

    filled: bool = True
    post_fill_30s_pnl: float | None = None
    fill_price: float | None = 15_000_000.0
    order_quantity: float = 0.001
    side: str = "buy"


def _make_records(
    wins: int,
    losses: int,
    win_pnl_bps: float = 5.0,
    loss_pnl_bps: float = -3.0,
) -> list[_MockFillRecord]:
    """指定勝敗数のレコードリストを生成."""
    records: list[_MockFillRecord] = []
    for _ in range(wins):
        records.append(_MockFillRecord(post_fill_30s_pnl=win_pnl_bps))
    for _ in range(losses):
        records.append(_MockFillRecord(post_fill_30s_pnl=loss_pnl_bps))
    return records


# ======================================================================
# compute_kelly_fraction
# ======================================================================
class TestComputeKellyFraction:
    """compute_kelly_fraction のテスト."""

    def test_sample_insufficient_returns_none(self) -> None:
        """サンプル不足で None を返す."""
        records = _make_records(wins=10, losses=5)
        result = compute_kelly_fraction(records, min_samples=30)
        assert result is None

    def test_basic_kelly_calculation(self) -> None:
        """基本的な Kelly 計算: p=0.6, b=5/3≈1.667."""
        records = _make_records(wins=60, losses=40, win_pnl_bps=5.0, loss_pnl_bps=-3.0)
        result = compute_kelly_fraction(records, min_samples=30, fractional=1.0)
        assert result is not None
        # p=0.6, b=5/3, f* = (0.6*5/3 - 0.4) / (5/3) = (1.0 - 0.4) / 1.667 = 0.36
        assert abs(result.win_rate - 0.6) < 0.01
        assert abs(result.win_loss_ratio - 5 / 3) < 0.01
        expected_f = (0.6 * (5 / 3) - 0.4) / (5 / 3)
        assert abs(result.kelly_fraction - expected_f) < 0.01
        assert result.sample_count == 100

    def test_half_kelly(self) -> None:
        """Fractional Kelly (half-Kelly): f*/2."""
        records = _make_records(wins=60, losses=40, win_pnl_bps=5.0, loss_pnl_bps=-3.0)
        result = compute_kelly_fraction(records, min_samples=30, fractional=0.5)
        assert result is not None
        full_kelly = (0.6 * (5 / 3) - 0.4) / (5 / 3)
        expected_half = full_kelly * 0.5
        assert abs(result.fractional_kelly - expected_half) < 0.01

    def test_no_edge_returns_zero(self) -> None:
        """勝率が低すぎると Kelly ≤ 0 → fractional_kelly=0."""
        # p=0.3, b=1.0, f* = (0.3*1 - 0.7)/1 = -0.4
        records = _make_records(wins=30, losses=70, win_pnl_bps=3.0, loss_pnl_bps=-3.0)
        result = compute_kelly_fraction(records, min_samples=30)
        assert result is not None
        assert result.kelly_fraction < 0
        assert result.fractional_kelly == 0.0
        assert result.recommended_lot == 0.0

    def test_max_fraction_cap(self) -> None:
        """Kelly > max_fraction のとき天井でキャップ."""
        # p=0.9, b=10/1=10, f* = (0.9*10-0.1)/10 = 0.89
        records = _make_records(wins=90, losses=10, win_pnl_bps=10.0, loss_pnl_bps=-1.0)
        result = compute_kelly_fraction(
            records, min_samples=30, fractional=1.0, max_fraction=0.25,
        )
        assert result is not None
        assert result.fractional_kelly == 0.25  # capped

    def test_all_wins_no_losses(self) -> None:
        """全勝 (loss=0): avg_loss=0 → b=inf → kelly=max_fraction."""
        records = _make_records(wins=50, losses=0, win_pnl_bps=5.0)
        result = compute_kelly_fraction(records, min_samples=30, max_fraction=0.25)
        assert result is not None
        # avg_loss=0 → b=inf → kelly_f = max_fraction = 0.25
        assert result.fractional_kelly <= 0.25
        assert result.sample_count == 50

    def test_all_wins_with_min_loss(self) -> None:
        """勝ち圧倒的 + 損失1件: edge 巨大 → max_fraction でキャップ."""
        records = _make_records(wins=50, losses=1, win_pnl_bps=5.0, loss_pnl_bps=-1.0)
        result = compute_kelly_fraction(records, min_samples=30, max_fraction=0.25)
        assert result is not None
        assert result.fractional_kelly <= 0.25

    def test_all_losses(self) -> None:
        """全敗: avg_win=0 → b=0 → kelly_f=0 → fractional_kelly=0."""
        records = _make_records(wins=0, losses=50, loss_pnl_bps=-5.0)
        result = compute_kelly_fraction(records, min_samples=30)
        assert result is not None
        assert result.kelly_fraction <= 0
        assert result.fractional_kelly == 0.0
        assert result.recommended_lot == 0.0

    def test_all_losses_with_min_win(self) -> None:
        """負け圧倒的 + 勝ち1件 → Kelly ≤ 0."""
        records = _make_records(wins=1, losses=50, win_pnl_bps=5.0, loss_pnl_bps=-5.0)
        result = compute_kelly_fraction(records, min_samples=30)
        assert result is not None
        assert result.kelly_fraction < 0
        assert result.fractional_kelly == 0.0

    def test_unfilled_records_excluded(self) -> None:
        """未約定レコードは除外."""
        records = _make_records(wins=20, losses=15)
        # 未約定を追加
        for _ in range(20):
            records.append(_MockFillRecord(filled=False, post_fill_30s_pnl=5.0))
        result = compute_kelly_fraction(records, min_samples=30)
        assert result is not None
        assert result.sample_count == 35

    def test_zero_pnl_excluded(self) -> None:
        """PnL=0 のレコードは勝ちにも負けにもカウントしない."""
        records = _make_records(wins=20, losses=15)
        for _ in range(20):
            records.append(_MockFillRecord(post_fill_30s_pnl=0.0))
        result = compute_kelly_fraction(records, min_samples=30)
        assert result is not None
        assert result.sample_count == 35  # 0.0 は win でも loss でもない


# ======================================================================
# kelly_recommended_lot
# ======================================================================
class TestKellyRecommendedLot:
    """kelly_recommended_lot のテスト."""

    def _make_kelly(
        self, fractional_kelly: float = 0.1, **kwargs: object
    ) -> KellyEstimate:
        defaults = dict(
            win_rate=0.6,
            win_loss_ratio=1.5,
            kelly_fraction=0.2,
            fractional_kelly=fractional_kelly,
            recommended_lot=0.0,
            sample_count=100,
            reason="test",
        )
        defaults.update(kwargs)
        return KellyEstimate(**defaults)

    def test_basic_lot_calculation(self) -> None:
        """fractional_kelly=0.1, equity=0.05 BTC → lot=0.005."""
        kelly = self._make_kelly(fractional_kelly=0.1)
        config = LotSizingConfig(min_lot=0.001, max_lot=0.010, lot_step=0.001)
        lot = kelly_recommended_lot(kelly, equity_btc=0.05, config=config)
        assert lot == 0.005  # 0.1 * 0.05 = 0.005

    def test_clamp_to_min(self) -> None:
        """算出ロットが min_lot 未満 → min_lot."""
        kelly = self._make_kelly(fractional_kelly=0.001)
        config = LotSizingConfig(min_lot=0.001, max_lot=0.010, lot_step=0.001)
        lot = kelly_recommended_lot(kelly, equity_btc=0.01, config=config)
        assert lot == 0.001  # 0.001 * 0.01 = 0.00001 → clamp to 0.001

    def test_clamp_to_max(self) -> None:
        """算出ロットが max_lot 超過 → max_lot."""
        kelly = self._make_kelly(fractional_kelly=0.5)
        config = LotSizingConfig(min_lot=0.001, max_lot=0.005, lot_step=0.001)
        lot = kelly_recommended_lot(kelly, equity_btc=1.0, config=config)
        assert lot == 0.005  # 0.5 * 1.0 = 0.5 → clamp to 0.005

    def test_step_rounding(self) -> None:
        """lot_step 刻みに切り捨て."""
        kelly = self._make_kelly(fractional_kelly=0.15)
        config = LotSizingConfig(min_lot=0.001, max_lot=0.010, lot_step=0.001)
        lot = kelly_recommended_lot(kelly, equity_btc=0.025, config=config)
        # 0.15 * 0.025 = 0.00375 → floor to 0.003
        assert lot == 0.003

    def test_zero_equity(self) -> None:
        """equity=0 → min_lot."""
        kelly = self._make_kelly(fractional_kelly=0.1)
        config = LotSizingConfig(min_lot=0.001)
        lot = kelly_recommended_lot(kelly, equity_btc=0.0, config=config)
        assert lot == 0.001

    def test_negative_fractional_kelly(self) -> None:
        """fractional_kelly≤0 → min_lot."""
        kelly = self._make_kelly(fractional_kelly=0.0)
        config = LotSizingConfig(min_lot=0.001)
        lot = kelly_recommended_lot(kelly, equity_btc=1.0, config=config)
        assert lot == 0.001


# ======================================================================
# compute_lot_size with Kelly ceiling
# ======================================================================
class TestComputeLotSizeKellyCeiling:
    """compute_lot_size の Kelly 天井統合テスト."""

    def _default_config(self, **overrides: object) -> LotSizingConfig:
        defaults = dict(
            current_lot=0.002,
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

    def _make_kelly(self, recommended_lot: float) -> KellyEstimate:
        return KellyEstimate(
            win_rate=0.6,
            win_loss_ratio=1.5,
            kelly_fraction=0.2,
            fractional_kelly=0.1,
            recommended_lot=recommended_lot,
            sample_count=100,
            reason="test Kelly",
        )

    def test_increase_without_kelly(self) -> None:
        """Kelly なし → 通常通り1段階増量."""
        config = self._default_config(current_lot=0.002)
        result = compute_lot_size(
            fill_rate=0.80, as_ratio=0.10, recent_pnl_bps=0.5,
            cumulative_pnl_jpy=100.0, sample_count=100,
            config=config, kelly_estimate=None,
        )
        assert result.new_lot == 0.003
        assert result.action == "increase"

    def test_kelly_ceiling_blocks_increase(self) -> None:
        """Kelly 推奨ロット < step-based 増量 → 天井適用."""
        config = self._default_config(current_lot=0.002)
        kelly = self._make_kelly(recommended_lot=0.002)  # 天井 = 現在値
        result = compute_lot_size(
            fill_rate=0.80, as_ratio=0.10, recent_pnl_bps=0.5,
            cumulative_pnl_jpy=100.0, sample_count=100,
            config=config, kelly_estimate=kelly,
        )
        assert result.new_lot == 0.002  # 増量ブロック
        assert result.action == "hold"

    def test_kelly_ceiling_allows_lower_increase(self) -> None:
        """Kelly 推奨ロット < step-based だが > 現在値 → Kelly まで増量."""
        config = self._default_config(current_lot=0.001, lot_step=0.002)
        # step-based: 0.001 + 0.002 = 0.003
        # Kelly: 0.002 (中間)
        kelly = self._make_kelly(recommended_lot=0.002)
        result = compute_lot_size(
            fill_rate=0.80, as_ratio=0.10, recent_pnl_bps=0.5,
            cumulative_pnl_jpy=100.0, sample_count=100,
            config=config, kelly_estimate=kelly,
        )
        # Kelly 天井: new(0.003) > kelly(0.002) → max(current(0.001), kelly(0.002)) = 0.002
        assert result.new_lot == 0.002

    def test_kelly_ceiling_allows_full_increase(self) -> None:
        """Kelly 推奨ロット >= step-based → 通常増量."""
        config = self._default_config(current_lot=0.002)
        kelly = self._make_kelly(recommended_lot=0.005)  # 天井に余裕あり
        result = compute_lot_size(
            fill_rate=0.80, as_ratio=0.10, recent_pnl_bps=0.5,
            cumulative_pnl_jpy=100.0, sample_count=100,
            config=config, kelly_estimate=kelly,
        )
        assert result.new_lot == 0.003  # 通常増量
        assert result.action == "increase"

    def test_kelly_zero_lot_does_not_affect(self) -> None:
        """Kelly recommended_lot=0 → 天井なし."""
        config = self._default_config(current_lot=0.002)
        kelly = self._make_kelly(recommended_lot=0.0)
        result = compute_lot_size(
            fill_rate=0.80, as_ratio=0.10, recent_pnl_bps=0.5,
            cumulative_pnl_jpy=100.0, sample_count=100,
            config=config, kelly_estimate=kelly,
        )
        assert result.new_lot == 0.003  # Kelly 天井なし → 通常増量

    def test_condition_bad_still_decreases_with_kelly(self) -> None:
        """条件未達 → Kelly 関係なく減量."""
        config = self._default_config(current_lot=0.003)
        kelly = self._make_kelly(recommended_lot=0.005)
        result = compute_lot_size(
            fill_rate=0.30, as_ratio=0.50, recent_pnl_bps=-1.0,
            cumulative_pnl_jpy=-1000.0, sample_count=100,
            config=config, kelly_estimate=kelly,
        )
        assert result.new_lot == 0.002
        assert result.action == "decrease"

    def test_loss_cap_overrides_kelly(self) -> None:
        """損失キャップ接近 → Kelly より優先で min_lot."""
        config = self._default_config(current_lot=0.003)
        kelly = self._make_kelly(recommended_lot=0.005)
        result = compute_lot_size(
            fill_rate=0.80, as_ratio=0.10, recent_pnl_bps=0.5,
            cumulative_pnl_jpy=-8000.0,  # warning threshold = -7000
            sample_count=100,
            config=config, kelly_estimate=kelly,
        )
        assert result.new_lot == 0.001
        assert result.action == "cap_shrink"
