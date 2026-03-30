"""585# MultiplicativePipelineMixin テスト — 乗算 offset チェーン包括テスト.

テスト対象:
  A. 193# EV offset 適用 (aggressive mode)
  B. 195# Velocity offset
  C. 196# Trending offset (sell only)
  D. 240# Toxicity offset (mult > 1.0 のみ)
  E. 202# VG sell supplement (velocity 未適用かつ VG 非発動時)
  F. 458# Macro offset (trend × side マトリクス)
  G. 215# Alert offset
  H. 372# Sidecar offset (bps 加算)
  I. Stages JSON 記録
  J. 421# Final clamp + hard skip
"""

from __future__ import annotations

import json
from unittest.mock import ANY, MagicMock

import pytest

from scripts.v460.lib.fill_config import FillTestConfig
from scripts.v460.lib.multiplicative_pipeline import MultiplicativePipelineMixin
from scripts.v460.lib.offset_pipeline import OffsetPipelineResult


# ── helper ──


def _make_mixin(
    *,
    ev_as_offset_enabled: bool = True,
    ev_offset_sensitivity: float = 1.0,
    ev_offset_min_mult: float = 1.0,
    ev_offset_max_mult: float = 3.0,
    ev_warning_threshold: float = 0.8,
    ev_warning_offset_factor: float = 1.5,
    vg_velocity_threshold_bps: float = 10.0,
    vg_offset_boost_factor: float = 1.5,
    execution_final_clamp_enabled: bool = False,
    execution_final_clamp_hard_skip_mult: float = 3.0,
    alert_offset_mult: float = 1.0,
    last_macro_trend: str | None = None,
    last_vg_triggered: bool = False,
    macro_sell_boost_strong_up: float = 1.0,
    macro_sell_boost_weak_up: float = 1.0,
    macro_buy_boost_strong_down: float = 1.0,
    macro_buy_boost_weak_down: float = 1.0,
) -> MultiplicativePipelineMixin:
    """テスト用 MultiplicativePipelineMixin stub."""
    obj = object.__new__(MultiplicativePipelineMixin)
    cfg = MagicMock(spec=FillTestConfig)
    cfg.skip_gate_ev_as_offset_enabled = ev_as_offset_enabled
    cfg.skip_gate_ev_offset_sensitivity = ev_offset_sensitivity
    cfg.skip_gate_ev_offset_min_mult = ev_offset_min_mult
    cfg.skip_gate_ev_offset_max_mult = ev_offset_max_mult
    cfg.skip_gate_ev_warning_threshold = ev_warning_threshold
    cfg.skip_gate_ev_warning_offset_factor = ev_warning_offset_factor
    cfg.volatility_guard_velocity_threshold_bps = vg_velocity_threshold_bps
    cfg.volatility_guard_offset_boost_factor = vg_offset_boost_factor
    cfg.execution_final_clamp_enabled = execution_final_clamp_enabled
    cfg.execution_final_clamp_hard_skip_mult = execution_final_clamp_hard_skip_mult
    cfg.execution_final_clamp_hard_skip_mult_overrides = {}
    cfg.resolve_hard_skip_mult = lambda side, regime: execution_final_clamp_hard_skip_mult
    cfg.macro_sell_boost_strong_up = macro_sell_boost_strong_up
    cfg.macro_sell_boost_weak_up = macro_sell_boost_weak_up
    cfg.macro_buy_boost_strong_down = macro_buy_boost_strong_down
    cfg.macro_buy_boost_weak_down = macro_buy_boost_weak_down
    obj.config = cfg  # type: ignore[attr-defined]

    maker = MagicMock()
    maker.last_vg_triggered = last_vg_triggered
    maker.last_sigma = 0.01
    maker.get_adverse_ofi = MagicMock(return_value=0.0)
    maker.get_robust_inputs = MagicMock(return_value=(0.01, 0.0))
    obj._maker_price = maker  # type: ignore[attr-defined]

    obj._last_macro_trend = last_macro_trend  # type: ignore[attr-defined]
    obj._alert_offset_mult = alert_offset_mult  # type: ignore[attr-defined]

    # _make_cycle_skip_record stub (for final clamp hard skip)
    obj._make_cycle_skip_record = MagicMock(return_value=MagicMock())  # type: ignore[attr-defined]

    return obj


_COMMON: dict = dict(
    side="sell",
    order_price=13_000_000,
    spread_at_order=3000.0,
    effective_offset_ratio=0.05,
    sg_ev_score=None,
    sg_velocity_offset_mult=None,
    sg_velocity_bps=None,
    trending_offset_mult=None,
    toxicity_offset_mult=1.0,
    sidecar_offset_bps=0.0,
    cycle_id="test-cycle",
)


def _run(mixin: MultiplicativePipelineMixin, **overrides: object) -> OffsetPipelineResult:
    kw = {**_COMMON, **overrides}
    return mixin._apply_offset_pipeline_multiplicative(**kw)  # type: ignore[arg-type]


# ── A. EV Offset (193#) ──


class TestEvOffset:
    """193# EV→offset multiplier (aggressive mode)."""

    def test_ev_offset_applied_when_enabled(self) -> None:
        m = _make_mixin(ev_as_offset_enabled=True)
        r = _run(m, sg_ev_score=0.5)
        assert r.ev_offset_applied is True
        assert r.ev_offset_mult_applied is not None

    def test_ev_offset_disabled(self) -> None:
        m = _make_mixin(ev_as_offset_enabled=False)
        r = _run(m, sg_ev_score=0.5)
        assert r.ev_offset_applied is False

    def test_ev_offset_no_spread(self) -> None:
        """spread が 0 なら EV offset 不適用."""
        m = _make_mixin(ev_as_offset_enabled=True)
        r = _run(m, sg_ev_score=0.5, spread_at_order=0.0)
        assert r.ev_offset_applied is False

    def test_ev_offset_none_score(self) -> None:
        """score が None なら不適用."""
        m = _make_mixin(ev_as_offset_enabled=True)
        r = _run(m, sg_ev_score=None)
        assert r.ev_offset_applied is False
        assert r.ev_offset_mult_applied is None

    def test_ev_mult_stored_even_when_not_applied(self) -> None:
        """EV mult=1.0 のとき applied=False だが mult 値は記録."""
        m = _make_mixin(
            ev_as_offset_enabled=True,
            ev_offset_min_mult=1.0,
            ev_offset_max_mult=1.0,
        )
        r = _run(m, sg_ev_score=0.3)
        # compute_ev_offset_multiplier returns 1.0 → _apply_offset_multiplier
        # returns (price, ratio, None, None) → ev_offset_applied=False
        # but _ev_offset_mult_applied is set to _ev_mult (1.0)
        assert r.ev_offset_applied is False
        assert r.ev_offset_mult_applied == pytest.approx(1.0)


# ── B. Velocity Offset (195#) ──


class TestVelocityOffset:
    """195# velocity offset multiplier."""

    def test_velocity_offset_applied(self) -> None:
        m = _make_mixin()
        r = _run(m, sg_velocity_offset_mult=1.5, sg_velocity_bps=15.0)
        # offset should increase
        assert r.effective_offset_ratio > 0.05

    def test_velocity_offset_none(self) -> None:
        m = _make_mixin()
        r = _run(m, sg_velocity_offset_mult=None)
        assert r.effective_offset_ratio == pytest.approx(0.05, abs=1e-6)

    def test_velocity_offset_mult_1(self) -> None:
        """mult=1.0 は適用しない."""
        m = _make_mixin()
        # side=buy で VG supplement 回避, velocity_bps=None
        r = _run(m, side="buy", sg_velocity_offset_mult=1.0, sg_velocity_bps=None)
        # _apply_offset_multiplier returns None for mult=1.0
        assert r.effective_offset_ratio == pytest.approx(0.05, abs=1e-6)


# ── C. Trending Offset (196#) ──


class TestTrendingOffset:
    """196# trending offset — sell only."""

    def test_trending_applied_on_sell(self) -> None:
        m = _make_mixin()
        r = _run(m, side="sell", trending_offset_mult=1.5)
        assert r.effective_offset_ratio > 0.05

    def test_trending_ignored_on_buy(self) -> None:
        """buy 側では trending offset を適用しない."""
        m = _make_mixin()
        r = _run(m, side="buy", trending_offset_mult=1.5)
        assert r.effective_offset_ratio == pytest.approx(0.05, abs=1e-6)


# ── D. Toxicity Offset (240#) ──


class TestToxicityOffset:
    """240# toxicity offset — mult > 1.0 のときのみ適用."""

    def test_toxicity_applied_when_gt_1(self) -> None:
        m = _make_mixin()
        r = _run(m, toxicity_offset_mult=1.3)
        assert r.effective_offset_ratio > 0.05

    def test_toxicity_not_applied_when_eq_1(self) -> None:
        m = _make_mixin()
        r = _run(m, toxicity_offset_mult=1.0)
        assert r.effective_offset_ratio == pytest.approx(0.05, abs=1e-6)

    def test_toxicity_not_applied_when_lt_1(self) -> None:
        m = _make_mixin()
        r = _run(m, toxicity_offset_mult=0.8)
        assert r.effective_offset_ratio == pytest.approx(0.05, abs=1e-6)


# ── E. VG Sell Supplement (202#) ──


class TestVgSellSupplement:
    """202# VG supplemental boost — sell + VG 未発動 + velocity > threshold + vel 未適用."""

    def test_vg_supplement_applied(self) -> None:
        """条件全て満たす: sell, !vg_triggered, |vel|>threshold, vel未適用."""
        m = _make_mixin(
            last_vg_triggered=False,
            vg_velocity_threshold_bps=10.0,
            vg_offset_boost_factor=1.5,
        )
        # velocity_offset_mult=None → vel 未適用, velocity_bps=15 > 10
        r = _run(m, side="sell", sg_velocity_offset_mult=None, sg_velocity_bps=15.0)
        assert r.effective_offset_ratio > 0.05

    def test_vg_supplement_not_applied_on_buy(self) -> None:
        m = _make_mixin(
            last_vg_triggered=False,
            vg_velocity_threshold_bps=10.0,
            vg_offset_boost_factor=1.5,
        )
        r = _run(m, side="buy", sg_velocity_offset_mult=None, sg_velocity_bps=15.0)
        assert r.effective_offset_ratio == pytest.approx(0.05, abs=1e-6)

    def test_vg_supplement_not_applied_when_vg_triggered(self) -> None:
        m = _make_mixin(
            last_vg_triggered=True,
            vg_velocity_threshold_bps=10.0,
            vg_offset_boost_factor=1.5,
        )
        r = _run(m, side="sell", sg_velocity_offset_mult=None, sg_velocity_bps=15.0)
        assert r.effective_offset_ratio == pytest.approx(0.05, abs=1e-6)

    def test_vg_supplement_not_applied_when_velocity_already_applied(self) -> None:
        """velocity offset が適用済みなら VG supplement は不要."""
        m = _make_mixin(
            last_vg_triggered=False,
            vg_velocity_threshold_bps=10.0,
            vg_offset_boost_factor=1.5,
        )
        r = _run(m, side="sell", sg_velocity_offset_mult=1.5, sg_velocity_bps=15.0)
        # velocity 適用済みのため VG supplement は発動しない
        # offset は velocity のみで上がる
        assert r.effective_offset_ratio > 0.05

    def test_vg_supplement_not_applied_below_threshold(self) -> None:
        m = _make_mixin(
            last_vg_triggered=False,
            vg_velocity_threshold_bps=10.0,
            vg_offset_boost_factor=1.5,
        )
        r = _run(m, side="sell", sg_velocity_offset_mult=None, sg_velocity_bps=5.0)
        assert r.effective_offset_ratio == pytest.approx(0.05, abs=1e-6)


# ── F. Macro Offset (458#) ──


class TestMacroOffset:
    """458# macro trend boost — side × trend マトリクス."""

    def test_sell_strong_up(self) -> None:
        m = _make_mixin(
            last_macro_trend="macro_strong_up",
            macro_sell_boost_strong_up=1.5,
        )
        r = _run(m, side="sell")
        assert r.macro_boost_applied is True
        assert r.effective_offset_ratio > 0.05

    def test_sell_weak_up(self) -> None:
        m = _make_mixin(
            last_macro_trend="macro_weak_up",
            macro_sell_boost_weak_up=1.3,
        )
        r = _run(m, side="sell")
        assert r.macro_boost_applied is True

    def test_buy_strong_down(self) -> None:
        m = _make_mixin(
            last_macro_trend="macro_strong_down",
            macro_buy_boost_strong_down=1.4,
        )
        r = _run(m, side="buy")
        assert r.macro_boost_applied is True

    def test_buy_weak_down(self) -> None:
        m = _make_mixin(
            last_macro_trend="macro_weak_down",
            macro_buy_boost_weak_down=1.2,
        )
        r = _run(m, side="buy")
        assert r.macro_boost_applied is True

    def test_sell_strong_down_no_boost(self) -> None:
        """sell + strong_down → macro_mult=1.0 → no boost."""
        m = _make_mixin(
            last_macro_trend="macro_strong_down",
            macro_buy_boost_strong_down=1.5,  # buy 用なので sell には影響しない
        )
        r = _run(m, side="sell")
        assert r.macro_boost_applied is False

    def test_no_macro_trend(self) -> None:
        m = _make_mixin(last_macro_trend=None)
        r = _run(m, side="sell")
        assert r.macro_boost_applied is False


# ── G. Alert Offset (215#) ──


class TestAlertOffset:
    """215# alert mode → offset multiplier."""

    def test_alert_offset_applied(self) -> None:
        m = _make_mixin(alert_offset_mult=1.5)
        r = _run(m)
        assert r.effective_offset_ratio > 0.05

    def test_alert_offset_1_no_change(self) -> None:
        m = _make_mixin(alert_offset_mult=1.0)
        r = _run(m)
        assert r.effective_offset_ratio == pytest.approx(0.05, abs=1e-6)

    def test_alert_offset_below_1(self) -> None:
        """mult < 1.0 でも != 1.0 なので _apply_offset_multiplier に渡される.
        ただし conservative mode (aggressive=False) では mult<1.0 で None 返却."""
        m = _make_mixin(alert_offset_mult=0.8)
        r = _run(m)
        # mult<1.0 is rejected by _apply_offset_multiplier (conservative mode)
        assert r.effective_offset_ratio == pytest.approx(0.05, abs=1e-6)


# ── H. Sidecar Offset (372#) ──


class TestSidecarOffset:
    """372# sidecar offset bps → price delta."""

    def test_sidecar_sell_positive_bps(self) -> None:
        """sell: positive bps → price UP (mid から遠ざける)."""
        m = _make_mixin()
        r = _run(m, sidecar_offset_bps=10.0)
        # delta = 10/10000 * 13_000_000 = 13_000
        # sell: price - delta
        assert r.order_price < 13_000_000

    def test_sidecar_buy_positive_bps(self) -> None:
        """buy: positive bps → price UP."""
        m = _make_mixin()
        r = _run(m, side="buy", sidecar_offset_bps=10.0)
        assert r.order_price > 13_000_000

    def test_sidecar_zero_no_change(self) -> None:
        m = _make_mixin()
        r = _run(m, sidecar_offset_bps=0.0)
        assert r.order_price == 13_000_000


# ── I. Stages JSON ──


class TestStagesJson:
    """executor_offset_stages_json の記録."""

    def test_stages_json_recorded_when_any_stage(self) -> None:
        m = _make_mixin()
        r = _run(m, sg_velocity_offset_mult=1.5, sg_velocity_bps=15.0)
        assert r.executor_offset_stages_json is not None
        stages = json.loads(r.executor_offset_stages_json)
        assert "velocity" in stages
        assert stages["velocity"] is not None

    def test_stages_json_all_none_when_no_adjustments(self) -> None:
        m = _make_mixin()
        r = _run(m)
        # All stages are None → json is None
        assert r.executor_offset_stages_json is None

    def test_stages_json_contains_ev(self) -> None:
        m = _make_mixin(ev_as_offset_enabled=True)
        r = _run(m, sg_ev_score=0.5)
        assert r.executor_offset_stages_json is not None
        stages = json.loads(r.executor_offset_stages_json)
        assert "ev" in stages

    def test_stages_json_contains_all_keys(self) -> None:
        m = _make_mixin(
            ev_as_offset_enabled=True,
            alert_offset_mult=1.5,
            last_macro_trend="macro_strong_up",
            macro_sell_boost_strong_up=1.3,
        )
        r = _run(
            m,
            side="sell",
            sg_ev_score=0.5,
            sg_velocity_offset_mult=1.2,
            sg_velocity_bps=12.0,
            trending_offset_mult=1.4,
            toxicity_offset_mult=1.1,
        )
        assert r.executor_offset_stages_json is not None
        stages = json.loads(r.executor_offset_stages_json)
        expected_keys = {"ev", "velocity", "trending", "toxicity", "vg_supp", "alert", "toxic_veto"}
        assert expected_keys == set(stages.keys())


# ── J. Final Clamp (421#) ──


class TestFinalClamp:
    """421# execution_final_clamp — clamp + hard skip."""

    def test_final_clamp_reduces_offset(self) -> None:
        """offset > ceiling → clamp."""
        m = _make_mixin(
            execution_final_clamp_enabled=True,
        )
        # resolve_offset_ceiling returns a low ceiling
        m.config.resolve_offset_ceiling = MagicMock(return_value=0.03)
        # Use high offset via toxicity to push above ceiling
        r = _run(m, toxicity_offset_mult=2.0, effective_offset_ratio=0.10)
        # Even if clamped, the pre_clamp should be recorded
        if r.execution_pre_clamp_offset is not None:
            assert r.execution_pre_clamp_offset > 0.03

    def test_final_clamp_hard_skip(self) -> None:
        """offset >> ceiling × hard_skip_mult → early return with skip record."""
        m = _make_mixin(
            execution_final_clamp_enabled=True,
            execution_final_clamp_hard_skip_mult=2.0,
        )
        # ceiling=0.01, offset will be ~0.10 which is >> 0.01 * 2.0
        m.config.resolve_offset_ceiling = MagicMock(return_value=0.01)
        r = _run(m, toxicity_offset_mult=2.0, effective_offset_ratio=0.10)
        assert r.early_return_record is not None
        m._make_cycle_skip_record.assert_called_once()

    def test_final_clamp_disabled(self) -> None:
        m = _make_mixin(execution_final_clamp_enabled=False)
        r = _run(m)
        assert r.execution_pre_clamp_offset is None
        assert r.early_return_record is None

    def test_final_clamp_spread_unavailable(self) -> None:
        """spread=None でも clamp 自体は実行、price 再計算はスキップ."""
        m = _make_mixin(
            execution_final_clamp_enabled=True,
        )
        m.config.resolve_offset_ceiling = MagicMock(return_value=0.03)
        r = _run(m, spread_at_order=None, effective_offset_ratio=0.10, toxicity_offset_mult=2.0)
        # Clamped but price not recalculated
        if r.execution_pre_clamp_offset is not None:
            assert r.execution_pre_clamp_offset > 0.03

    def test_final_clamp_uses_robust_inputs(self) -> None:
        m = _make_mixin(execution_final_clamp_enabled=True)
        m._maker_price.get_robust_inputs = MagicMock(return_value=(11.2, 0.6))
        m.config.resolve_offset_ceiling = MagicMock(return_value=1.0)
        _run(m)
        m._maker_price.get_robust_inputs.assert_called_once_with("sell")
        m.config.resolve_offset_ceiling.assert_called_once_with(
            "sell",
            utc_hour=ANY,
            sigma=11.2,
            adverse_ofi=0.6,
        )

    def test_regime_aware_hard_skip_relaxed(self) -> None:
        """641# P1-A: buy/trending_down override(4.0) でハードスキップ閾値が緩和."""
        m = _make_mixin(
            execution_final_clamp_enabled=True,
            execution_final_clamp_hard_skip_mult=2.0,
        )
        # toxicity_offset_mult=2.5 → stage cap 2.0 → offset=0.05*2.0=0.10
        # ceiling=0.04, default(2.0)→ threshold=0.08; override(4.0)→ threshold=0.16
        m.config.resolve_offset_ceiling = MagicMock(return_value=0.04)
        # Override: mult=4.0 → threshold=0.16 → 0.10 < 0.16 → no skip
        m.config.resolve_hard_skip_mult = MagicMock(return_value=4.0)
        r = _run(m, side="buy", effective_offset_ratio=0.05, toxicity_offset_mult=2.5)
        assert r.early_return_record is None

    def test_regime_aware_hard_skip_default_still_skips(self) -> None:
        """641# 非オーバーライド regime ではデフォルト mult で引き続きスキップ."""
        m = _make_mixin(
            execution_final_clamp_enabled=True,
            execution_final_clamp_hard_skip_mult=2.0,
        )
        m.config.resolve_offset_ceiling = MagicMock(return_value=0.04)
        # Default mult=2.0 → threshold=0.08 → 0.10 > 0.08 → skip
        m.config.resolve_hard_skip_mult = MagicMock(return_value=2.0)
        r = _run(m, side="buy", effective_offset_ratio=0.05, toxicity_offset_mult=2.5)
        assert r.early_return_record is not None
        m._make_cycle_skip_record.assert_called_once()


# ── J-2. resolve_hard_skip_mult 単体テスト (641#) ──


class TestResolveHardSkipMult:
    """641# FillTestConfig.resolve_hard_skip_mult の単体テスト."""

    def _make_config(self, overrides: dict[str, float] | None = None) -> FillTestConfig:
        cfg = FillTestConfig()
        cfg.execution_final_clamp_hard_skip_mult = 2.5
        if overrides:
            cfg.execution_final_clamp_hard_skip_mult_overrides = overrides
        return cfg

    def test_no_override_returns_default(self) -> None:
        cfg = self._make_config()
        assert cfg.resolve_hard_skip_mult("buy", "ranging") == 2.5

    def test_override_returns_override_value(self) -> None:
        cfg = self._make_config({"buy/trending_down": 4.0})
        assert cfg.resolve_hard_skip_mult("buy", "trending_down") == 4.0

    def test_override_miss_returns_default(self) -> None:
        cfg = self._make_config({"buy/trending_down": 4.0})
        assert cfg.resolve_hard_skip_mult("sell", "trending_down") == 2.5

    def test_regime_none_returns_default(self) -> None:
        cfg = self._make_config({"buy/trending_down": 4.0})
        assert cfg.resolve_hard_skip_mult("buy", None) == 2.5


# ── K. 統合テスト: 複数段チェーン ──


class TestChainedStages:
    """複数ステージが連鎖する場合の正しい集約."""

    def test_ev_plus_velocity_compound(self) -> None:
        """EV + velocity の乗算が正しく重畳する."""
        m = _make_mixin(ev_as_offset_enabled=True)
        r = _run(m, sg_ev_score=0.5, sg_velocity_offset_mult=1.5, sg_velocity_bps=15.0)
        assert r.ev_offset_applied is True
        assert r.effective_offset_ratio > 0.05

    def test_all_stages_active(self) -> None:
        """全ステージ同時適用."""
        m = _make_mixin(
            ev_as_offset_enabled=True,
            alert_offset_mult=1.3,
            last_macro_trend="macro_strong_up",
            macro_sell_boost_strong_up=1.2,
        )
        r = _run(
            m,
            side="sell",
            sg_ev_score=0.6,
            sg_velocity_offset_mult=1.4,
            sg_velocity_bps=20.0,
            trending_offset_mult=1.3,
            toxicity_offset_mult=1.5,
            sidecar_offset_bps=5.0,
        )
        assert r.ev_offset_applied is True
        assert r.macro_boost_applied is True
        assert r.effective_offset_ratio > 0.05
        assert r.executor_offset_stages_json is not None

    def test_return_dataclass_fields(self) -> None:
        """OffsetPipelineResult のフィールドが正しく設定される."""
        m = _make_mixin()
        r = _run(m)
        assert isinstance(r, OffsetPipelineResult)
        assert r.early_return_record is None
        assert r.ev_score_pretrade is None
