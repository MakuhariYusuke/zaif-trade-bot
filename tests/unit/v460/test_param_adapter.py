"""
方策 A: パラメータ適応 単体テスト.

scripts/v460/lib/param_adapter.py の compute_adaptation / clamp_offset を検証。
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

from scripts.v460.lib.param_adapter import (
    AdaptationConfig,
    AdaptationResult,
    clamp_offset,
    compute_adaptation,
)


# =====================================================================
# compute_adaptation テスト
# =====================================================================


class TestComputeAdaptation:
    """compute_adaptation のテスト."""

    def _make_config(self, current: float = 0.05) -> AdaptationConfig:
        return AdaptationConfig(
            current_offset_ratio=current,
            min_fill_rate=0.80,
            max_as_ratio=0.15,
            step_ratio=0.01,
            min_offset_ratio=0.01,
            max_offset_ratio=0.30,
            min_samples=50,
        )

    def test_hold_normal(self) -> None:
        """fill_rate/AS 両方正常 → hold."""
        config = self._make_config(current=0.05)
        result = compute_adaptation(
            fill_rate=0.90, as_ratio=0.10, sample_count=100, config=config,
        )
        assert result.action == "hold"
        assert result.new_offset == 0.05
        assert not result.changed

    def test_hold_insufficient_samples(self) -> None:
        """サンプル不足 → hold (異常値でも変更しない)."""
        config = self._make_config(current=0.05)
        result = compute_adaptation(
            fill_rate=0.50, as_ratio=0.50, sample_count=10, config=config,
        )
        assert result.action == "hold"
        assert result.new_offset == 0.05
        assert "サンプル不足" in result.reason

    def test_increase_low_fill_rate(self) -> None:
        """fill_rate 低下 → offset 増加."""
        config = self._make_config(current=0.05)
        result = compute_adaptation(
            fill_rate=0.70, as_ratio=0.10, sample_count=100, config=config,
        )
        assert result.action == "increase"
        assert result.new_offset == pytest.approx(0.06)
        assert result.changed

    def test_decrease_high_as(self) -> None:
        """AS 超過 → offset 減少."""
        config = self._make_config(current=0.05)
        result = compute_adaptation(
            fill_rate=0.90, as_ratio=0.25, sample_count=100, config=config,
        )
        assert result.action == "decrease"
        assert result.new_offset == pytest.approx(0.04)
        assert result.changed

    def test_both_abnormal_as_priority(self) -> None:
        """fill_rate 低 + AS 高 → AS 回避優先で decrease."""
        config = self._make_config(current=0.05)
        result = compute_adaptation(
            fill_rate=0.70, as_ratio=0.25, sample_count=100, config=config,
        )
        assert result.action == "decrease"
        assert result.new_offset == pytest.approx(0.04)
        assert "AS 回避優先" in result.reason

    def test_clamp_min(self) -> None:
        """offset がハードリミット下限に達する."""
        config = self._make_config(current=0.01)
        result = compute_adaptation(
            fill_rate=0.90, as_ratio=0.25, sample_count=100, config=config,
        )
        assert result.action == "decrease"
        # 0.01 - 0.01 = 0.00 → clamped to 0.01
        assert result.new_offset == 0.01

    def test_clamp_max(self) -> None:
        """offset がハードリミット上限に達する."""
        config = self._make_config(current=0.30)
        result = compute_adaptation(
            fill_rate=0.50, as_ratio=0.10, sample_count=100, config=config,
        )
        assert result.action == "increase"
        # 0.30 + 0.01 = 0.31 → clamped to 0.30
        assert result.new_offset == 0.30

    def test_default_config(self) -> None:
        """config=None でデフォルト設定が使われる."""
        result = compute_adaptation(
            fill_rate=0.90, as_ratio=0.10, sample_count=100, config=None,
        )
        assert result.action == "hold"

    def test_boundary_fill_rate(self) -> None:
        """fill_rate がちょうど閾値 → hold (< ではなく)."""
        config = self._make_config(current=0.05)
        result = compute_adaptation(
            fill_rate=0.80, as_ratio=0.10, sample_count=100, config=config,
        )
        assert result.action == "hold"

    def test_boundary_as_ratio(self) -> None:
        """AS がちょうど閾値 → hold (> ではなく)."""
        config = self._make_config(current=0.05)
        result = compute_adaptation(
            fill_rate=0.90, as_ratio=0.15, sample_count=100, config=config,
        )
        assert result.action == "hold"

    def test_result_fields(self) -> None:
        """AdaptationResult のフィールドが正しく設定される."""
        config = self._make_config(current=0.10)
        result = compute_adaptation(
            fill_rate=0.75, as_ratio=0.08, sample_count=200, config=config,
        )
        assert result.previous_offset == 0.10
        assert result.fill_rate == 0.75
        assert result.as_ratio == 0.08
        assert result.sample_count == 200

    def test_repeated_adaptation(self) -> None:
        """連続適応でも step ずつしか変化しない."""
        offset = 0.05
        for _ in range(3):
            config = AdaptationConfig(current_offset_ratio=offset, min_samples=10)
            result = compute_adaptation(
                fill_rate=0.50, as_ratio=0.10, sample_count=100, config=config,
            )
            assert result.action == "increase"
            offset = result.new_offset
        # 3 回増加: 0.05 → 0.06 → 0.07 → 0.08
        assert offset == pytest.approx(0.08)


# =====================================================================
# clamp_offset テスト
# =====================================================================


class TestClampOffset:
    """clamp_offset のテスト."""

    def test_within_range(self) -> None:
        assert clamp_offset(0.10) == 0.10

    def test_below_min(self) -> None:
        assert clamp_offset(-0.05) == 0.01

    def test_above_max(self) -> None:
        assert clamp_offset(0.50) == 0.30

    def test_custom_config(self) -> None:
        config = AdaptationConfig(min_offset_ratio=0.02, max_offset_ratio=0.20)
        assert clamp_offset(0.01, config) == 0.02
        assert clamp_offset(0.25, config) == 0.20
