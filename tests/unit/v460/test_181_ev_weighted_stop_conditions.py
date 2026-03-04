"""Tests for 181# EV_weighted + Stop Condition Monitor.

- _compute_ev_weighted: pnl30/pnl120 加重平均 (178# §1.3)
- _check_regime_stop_conditions: fill_rate / pnl30 安全弁
- ev_weighted_pnl: FillRecord フィールド追加
"""

from __future__ import annotations

from dataclasses import dataclass
from unittest.mock import MagicMock

import pytest

from scripts.v460.lib.fill_cycle_executor import FillCycleExecutorMixin
from scripts.v460.lib.regime_policy import DefaultCycleStrategy, RegimePolicyConfig
from ztb.metrics.fill_quality import FillRecord


# ======================================================================
# _compute_ev_weighted
# ======================================================================


class TestComputeEvWeighted:
    """181# EV_weighted: 30s/120s PnL 加重平均."""

    compute = staticmethod(FillCycleExecutorMixin._compute_ev_weighted)

    def test_both_available(self):
        """pnl30=1.0, pnl120=2.0 → 0.4*1+0.6*2 = 1.6."""
        result = self.compute(1.0, 2.0)
        assert result == pytest.approx(1.6)

    def test_pnl120_none_fallback_to_pnl30(self):
        """pnl120=None (E3 サンプリング外) → pnl30 単独値."""
        result = self.compute(0.5, None)
        assert result == pytest.approx(0.5)

    def test_pnl30_none_returns_none(self):
        """pnl30=None → 計算不能."""
        result = self.compute(None, 2.0)
        assert result is None

    def test_both_none_returns_none(self):
        result = self.compute(None, None)
        assert result is None

    def test_custom_weights(self):
        """カスタム w30/w120 指定."""
        result = self.compute(10.0, 20.0, w30=0.5, w120=0.5)
        assert result == pytest.approx(15.0)

    def test_negative_pnl(self):
        """マイナス PnL でも正しく計算."""
        result = self.compute(-1.0, -2.0)
        assert result == pytest.approx(-1.6)

    def test_zero_pnl(self):
        result = self.compute(0.0, 0.0)
        assert result == pytest.approx(0.0)

    def test_pnl30_zero_pnl120_none(self):
        """pnl30=0.0, pnl120=None → 0.0 (falsy だが None ではない)."""
        result = self.compute(0.0, None)
        assert result == pytest.approx(0.0)


# ======================================================================
# FillRecord.ev_weighted_pnl フィールド
# ======================================================================


class TestFillRecordEvWeightedField:
    """ev_weighted_pnl フィールドが FillRecord に存在し正しく動作する."""

    def _make_record(self, **kwargs) -> FillRecord:
        """最小限フィールドでFillRecord生成."""
        defaults = dict(
            cycle_id="test-001",
            timestamp=1700000000.0,
            side="buy",
            order_price=15000000.0,
            order_quantity=0.001,
            filled=True,
            queue_wait_sec=5.0,
            post_fill_30s_pnl=1.0,
        )
        defaults.update(kwargs)
        return FillRecord(**defaults)

    def test_field_default_none(self):
        r = self._make_record()
        assert r.ev_weighted_pnl is None

    def test_field_set_value(self):
        r = self._make_record(ev_weighted_pnl=1.6)
        assert r.ev_weighted_pnl == pytest.approx(1.6)

    def test_to_dict_includes_ev_weighted(self):
        r = self._make_record(ev_weighted_pnl=2.5)
        d = r.to_dict()
        assert d["ev_weighted_pnl"] == pytest.approx(2.5)

    def test_from_dict_round_trip(self):
        r = self._make_record(ev_weighted_pnl=-0.3)
        r2 = FillRecord.from_dict(r.to_dict())
        assert r2.ev_weighted_pnl == pytest.approx(-0.3)


# ======================================================================
# _check_regime_stop_conditions
# ======================================================================


@dataclass
class _FakeRecord:
    """テスト用の軽量レコード."""
    filled: bool = True
    post_fill_30s_pnl: float | None = 0.5


class TestCheckRegimeStopConditions:
    """181# 停止条件モニター."""

    @pytest.fixture
    def orchestrator(self):
        """Orchestrator 相当の Mock オブジェクト."""
        from scripts.v460.lib.fill_loop_orchestrator import FillLoopOrchestratorMixin

        obj = MagicMock(spec=FillLoopOrchestratorMixin)
        policy = RegimePolicyConfig(
            dynamic_cycle_enabled=True,
            chase_enabled=True,
            fill_rate_floor=0.35,
            pnl_floor_bps=-0.8,
        )
        strategy = DefaultCycleStrategy(
            base_interval=120.0,
            base_wait_buy=30.0,
            base_wait_sell=90.0,
            policy=policy,
        )
        obj._cycle_strategy = strategy
        # 277# config mock: fallback_duration_sec, sell_dynamic_kill_window, min_adapt_samples
        _mock_config = MagicMock()
        _mock_config.fallback_duration_sec = 3600.0
        _mock_config.sell_dynamic_kill_window = 50
        _mock_config.min_adapt_samples = 50
        obj.config = _mock_config
        # Bind the real method
        obj._check_regime_stop_conditions = (
            FillLoopOrchestratorMixin._check_regime_stop_conditions.__get__(obj)
        )
        return obj

    def test_fill_rate_below_floor_triggers_fallback(self, orchestrator):
        """fill_rate < fill_rate_floor → fallback 起動."""
        orchestrator._recent_records = []
        orchestrator._check_regime_stop_conditions(filled_count=10, total_count=100)
        # fill_rate = 10% < 35% → must have called activate_fallback
        assert orchestrator._cycle_strategy._fallback_until is not None

    def test_fill_rate_above_floor_no_fallback(self, orchestrator):
        """fill_rate >= fill_rate_floor → fallback 不発."""
        orchestrator._recent_records = [
            _FakeRecord(filled=True, post_fill_30s_pnl=1.0) for _ in range(20)
        ]
        before = orchestrator._cycle_strategy._fallback_until
        orchestrator._check_regime_stop_conditions(filled_count=50, total_count=100)
        assert orchestrator._cycle_strategy._fallback_until == before

    def test_pnl_below_floor_triggers_fallback(self, orchestrator):
        """avg_pnl30 < pnl_floor_bps → fallback 起動."""
        orchestrator._recent_records = [
            _FakeRecord(filled=True, post_fill_30s_pnl=-1.5) for _ in range(20)
        ]
        orchestrator._check_regime_stop_conditions(filled_count=50, total_count=100)
        assert orchestrator._cycle_strategy._fallback_until is not None

    def test_pnl_above_floor_no_fallback(self, orchestrator):
        """avg_pnl30 >= pnl_floor_bps → fallback 不発."""
        orchestrator._recent_records = [
            _FakeRecord(filled=True, post_fill_30s_pnl=1.0) for _ in range(20)
        ]
        before = orchestrator._cycle_strategy._fallback_until
        orchestrator._check_regime_stop_conditions(filled_count=50, total_count=100)
        assert orchestrator._cycle_strategy._fallback_until == before

    def test_too_few_samples_no_pnl_check(self, orchestrator):
        """filled records < 10 → pnl チェックスキップ."""
        orchestrator._recent_records = [
            _FakeRecord(filled=True, post_fill_30s_pnl=-5.0) for _ in range(5)
        ]
        before = orchestrator._cycle_strategy._fallback_until
        orchestrator._check_regime_stop_conditions(filled_count=50, total_count=100)
        assert orchestrator._cycle_strategy._fallback_until == before

    def test_disabled_policy_skips(self, orchestrator):
        """C/D/Chase 全 disabled → チェック不要."""
        disabled_policy = RegimePolicyConfig(
            dynamic_cycle_enabled=False,
            chase_enabled=False,
        )
        # 新しい strategy を作り直す (他テストの fallback 影響を排除)
        fresh = DefaultCycleStrategy(
            base_interval=120.0,
            base_wait_buy=30.0,
            base_wait_sell=90.0,
            policy=disabled_policy,
        )
        orchestrator._cycle_strategy = fresh
        orchestrator._recent_records = [
            _FakeRecord(filled=True, post_fill_30s_pnl=-5.0) for _ in range(50)
        ]
        # Should return without calling fallback — _fallback_until stays 0.0
        orchestrator._check_regime_stop_conditions(filled_count=1, total_count=100)
        assert orchestrator._cycle_strategy._fallback_until == 0.0

    def test_zero_total_count_no_crash(self, orchestrator):
        """total_count=0 → ゼロ除算なし."""
        orchestrator._recent_records = []
        orchestrator._check_regime_stop_conditions(filled_count=0, total_count=0)
        # No exception = pass

    def test_mixed_records_with_none_pnl(self, orchestrator):
        """pnl=None のレコードが混在しても正しく集計."""
        records = [
            _FakeRecord(filled=True, post_fill_30s_pnl=1.0) for _ in range(12)
        ]
        records += [
            _FakeRecord(filled=True, post_fill_30s_pnl=None) for _ in range(5)
        ]
        records += [
            _FakeRecord(filled=False, post_fill_30s_pnl=-3.0) for _ in range(3)
        ]
        orchestrator._recent_records = records
        before = orchestrator._cycle_strategy._fallback_until
        orchestrator._check_regime_stop_conditions(filled_count=50, total_count=100)
        # 12 records with pnl=1.0 → avg=1.0 > -0.8 → no fallback
        assert orchestrator._cycle_strategy._fallback_until == before


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
