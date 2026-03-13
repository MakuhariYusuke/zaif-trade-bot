"""336# velocity_math.py のユニットテスト.

208# SSOT として集約された velocity 計算ロジックの正当性を検証する。
2 関数: compute_instant_velocity_bps, compute_velocity_offset_multiplier

テスト観点:
  - 正常系: 上昇/下降 velocity の符号規約 (正=上昇, 負=下降)
  - 境界値: dt=0, dt>=max_dt, prev_mid=0 → None
  - 定量: 既知の入力に対する期待 bps 値
  - offset multiplier: fixed mode, proportional mode, 0 除算, 上限・下限
"""

from __future__ import annotations

import math

import pytest

from scripts.v460.lib.velocity_math import (
    compute_instant_velocity_bps,
    compute_velocity_offset_multiplier,
)


# ═══════════════════════════════════════════════════════════════════════
# compute_instant_velocity_bps
# ═══════════════════════════════════════════════════════════════════════


class TestComputeInstantVelocityBps:
    """208# SSOT: orderbook mid-price → 瞬間 velocity (bps)."""

    def test_upward_movement_positive(self) -> None:
        """価格上昇 → 正の velocity."""
        result = compute_instant_velocity_bps(
            current_mid=10_050.0, prev_mid=10_000.0, dt=1.0, max_dt=10.0,
        )
        assert result is not None
        # (10050 - 10000) / 10000 * 10000 = 50 bps
        assert result == pytest.approx(50.0)

    def test_downward_movement_negative(self) -> None:
        """価格下降 → 負の velocity."""
        result = compute_instant_velocity_bps(
            current_mid=9_950.0, prev_mid=10_000.0, dt=1.0, max_dt=10.0,
        )
        assert result is not None
        assert result == pytest.approx(-50.0)

    def test_no_movement_zero(self) -> None:
        """価格変動なし → 0."""
        result = compute_instant_velocity_bps(
            current_mid=10_000.0, prev_mid=10_000.0, dt=1.0, max_dt=10.0,
        )
        assert result == pytest.approx(0.0)

    def test_small_btc_movement(self) -> None:
        """BTC 14,000,000 JPY → 1 tick (100 JPY) = ~0.0714 bps."""
        result = compute_instant_velocity_bps(
            current_mid=14_000_100.0,
            prev_mid=14_000_000.0,
            dt=0.5,
            max_dt=5.0,
        )
        assert result is not None
        expected = 100.0 / 14_000_000.0 * 10_000
        assert result == pytest.approx(expected, rel=1e-6)

    # --- 境界値: None を返すケース ---

    def test_stale_dt_returns_none(self) -> None:
        """dt >= max_dt → stale → None."""
        assert compute_instant_velocity_bps(
            current_mid=10_050.0, prev_mid=10_000.0, dt=10.0, max_dt=10.0,
        ) is None

    def test_dt_exceeds_max_returns_none(self) -> None:
        """dt > max_dt → None."""
        assert compute_instant_velocity_bps(
            current_mid=10_050.0, prev_mid=10_000.0, dt=15.0, max_dt=10.0,
        ) is None

    def test_zero_dt_returns_none(self) -> None:
        """dt=0 → 除算不能 → None."""
        assert compute_instant_velocity_bps(
            current_mid=10_050.0, prev_mid=10_000.0, dt=0.0, max_dt=10.0,
        ) is None

    def test_negative_dt_returns_none(self) -> None:
        """dt<0 → 不正 → None."""
        assert compute_instant_velocity_bps(
            current_mid=10_050.0, prev_mid=10_000.0, dt=-1.0, max_dt=10.0,
        ) is None

    def test_zero_prev_mid_returns_none(self) -> None:
        """prev_mid=0 → 除算不能 → None."""
        assert compute_instant_velocity_bps(
            current_mid=10_050.0, prev_mid=0.0, dt=1.0, max_dt=10.0,
        ) is None

    def test_negative_prev_mid_returns_none(self) -> None:
        """prev_mid<0 → 不正 → None."""
        assert compute_instant_velocity_bps(
            current_mid=10_050.0, prev_mid=-100.0, dt=1.0, max_dt=10.0,
        ) is None

    # --- dt は velocity 値に影響しないことの確認 ---

    def test_dt_does_not_scale_result(self) -> None:
        """dt は stale gate のみ。velocity 自体は dt に依存しない."""
        v1 = compute_instant_velocity_bps(
            current_mid=10_050.0, prev_mid=10_000.0, dt=1.0, max_dt=10.0,
        )
        v2 = compute_instant_velocity_bps(
            current_mid=10_050.0, prev_mid=10_000.0, dt=5.0, max_dt=10.0,
        )
        assert v1 == v2  # 同一 velocity、dt は gate のみ


# ═══════════════════════════════════════════════════════════════════════
# compute_velocity_offset_multiplier
# ═══════════════════════════════════════════════════════════════════════


class TestComputeVelocityOffsetMultiplier:
    """200# L / 208# SSOT: velocity → offset 乗数の計算ロジック."""

    # --- Fixed mode (proportional=False) ---

    def test_fixed_mode_returns_base(self) -> None:
        """proportional=False → base_multiplier をそのまま返す."""
        mult, was_prop = compute_velocity_offset_multiplier(
            observed_velocity_bps=10.0,
            threshold_bps=5.0,
            base_multiplier=2.0,
            max_multiplier=4.0,
            proportional=False,
        )
        assert mult == pytest.approx(2.0)
        assert was_prop is False

    def test_fixed_mode_clamps_to_max(self) -> None:
        """base > max → max にクランプ."""
        mult, _ = compute_velocity_offset_multiplier(
            observed_velocity_bps=10.0,
            threshold_bps=5.0,
            base_multiplier=5.0,
            max_multiplier=3.0,
            proportional=False,
        )
        assert mult == pytest.approx(3.0)

    def test_fixed_mode_floor_at_1(self) -> None:
        """base < 1.0 → 1.0 にフロア."""
        mult, _ = compute_velocity_offset_multiplier(
            observed_velocity_bps=10.0,
            threshold_bps=5.0,
            base_multiplier=0.5,
            max_multiplier=4.0,
            proportional=False,
        )
        assert mult == pytest.approx(1.0)

    # --- Proportional mode ---

    def test_proportional_exact_threshold(self) -> None:
        """velocity == threshold → excess_ratio=1.0 → boost = 1 + (base-1)*1 = base."""
        mult, was_prop = compute_velocity_offset_multiplier(
            observed_velocity_bps=5.0,
            threshold_bps=5.0,
            base_multiplier=2.0,
            max_multiplier=4.0,
            proportional=True,
        )
        assert mult == pytest.approx(2.0)
        assert was_prop is True

    def test_proportional_double_threshold(self) -> None:
        """velocity == 2*threshold → excess_ratio=2.0 → boost = 1 + (2-1)*2 = 3.0."""
        mult, was_prop = compute_velocity_offset_multiplier(
            observed_velocity_bps=10.0,
            threshold_bps=5.0,
            base_multiplier=2.0,
            max_multiplier=4.0,
            proportional=True,
        )
        assert mult == pytest.approx(3.0)
        assert was_prop is True

    def test_proportional_capped_at_max(self) -> None:
        """過大 velocity → max_multiplier にクランプ."""
        mult, _ = compute_velocity_offset_multiplier(
            observed_velocity_bps=100.0,
            threshold_bps=5.0,
            base_multiplier=2.0,
            max_multiplier=4.0,
            proportional=True,
        )
        assert mult == pytest.approx(4.0)

    def test_proportional_half_threshold(self) -> None:
        """velocity < threshold → excess_ratio < 1 → boost < base."""
        mult, was_prop = compute_velocity_offset_multiplier(
            observed_velocity_bps=2.5,
            threshold_bps=5.0,
            base_multiplier=2.0,
            max_multiplier=4.0,
            proportional=True,
        )
        # excess_ratio = 2.5/5.0 = 0.5
        # boost = 1 + (2-1)*0.5 = 1.5
        assert mult == pytest.approx(1.5)
        assert was_prop is True

    def test_proportional_negative_velocity_uses_abs(self) -> None:
        """負の velocity → abs で処理."""
        mult, _ = compute_velocity_offset_multiplier(
            observed_velocity_bps=-10.0,
            threshold_bps=5.0,
            base_multiplier=2.0,
            max_multiplier=4.0,
            proportional=True,
        )
        # abs(-10) / 5 = 2.0 → boost = 1 + 1*2 = 3.0
        assert mult == pytest.approx(3.0)

    # --- ゼロ除算防御 ---

    def test_zero_threshold_falls_back_to_fixed(self) -> None:
        """threshold=0 → proportional 不可 → fixed fallback."""
        mult, was_prop = compute_velocity_offset_multiplier(
            observed_velocity_bps=10.0,
            threshold_bps=0.0,
            base_multiplier=2.0,
            max_multiplier=4.0,
            proportional=True,
        )
        assert mult == pytest.approx(2.0)
        assert was_prop is False  # fallback to fixed

    def test_negative_threshold_falls_back_to_fixed(self) -> None:
        """threshold<0 → abs(0) check 回避, 実質 0 ではない → proportional 動作."""
        mult, was_prop = compute_velocity_offset_multiplier(
            observed_velocity_bps=10.0,
            threshold_bps=-5.0,
            base_multiplier=2.0,
            max_multiplier=4.0,
            proportional=True,
        )
        # abs(-5) = 5, excess = 10/5 = 2 → boost = 1 + 1*2 = 3
        assert mult == pytest.approx(3.0)
        assert was_prop is True

    # --- max_multiplier boundary ---

    def test_max_below_1_clamped_to_1(self) -> None:
        """max_multiplier < 1.0 → capped_max = 1.0 → mult = 1.0."""
        mult, _ = compute_velocity_offset_multiplier(
            observed_velocity_bps=10.0,
            threshold_bps=5.0,
            base_multiplier=2.0,
            max_multiplier=0.5,
            proportional=False,
        )
        assert mult == pytest.approx(1.0)

    def test_zero_velocity_returns_1(self) -> None:
        """velocity=0 → proportional excess_ratio=0 → boost=1.0."""
        mult, was_prop = compute_velocity_offset_multiplier(
            observed_velocity_bps=0.0,
            threshold_bps=5.0,
            base_multiplier=2.0,
            max_multiplier=4.0,
            proportional=True,
        )
        assert mult == pytest.approx(1.0)
        assert was_prop is True
