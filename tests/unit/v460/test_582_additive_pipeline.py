"""582# Additive Pipeline テスト — RMS Toxicity/Liquidity バッファ分離.

テスト対象:
  A. _apply_offset_pipeline dispatcher: config flag で加法/乗法を切り替え
  B. _apply_offset_pipeline_additive: RMS 結合ロジック
  C. Toxicity/Liquidity 分類の正確性
  D. 全 multiplier ≤ 1.0 のとき base_ratio 保持
  E. VG sell supplement (additive path)
  F. Macro buy/weak_down (additive path)
  G. Alert > 1.0 (additive path)
  H. Final clamp hard skip (additive path)
  I. Final clamp with spread (normal clamp)
  J. _scale_lot
"""

from __future__ import annotations

import inspect
import json
import math
from unittest.mock import MagicMock, PropertyMock

import pytest

from scripts.v460.lib.fill_config import FillTestConfig
from scripts.v460.lib.offset_pipeline import OffsetPipelineMixin, OffsetPipelineResult


def _make_mixin(
    *,
    experimental_additive: bool = True,
    ev_as_offset_enabled: bool = True,
    ev_offset_sensitivity: float = 1.0,
    ev_offset_min_mult: float = 1.0,
    ev_offset_max_mult: float = 3.0,
    ev_warning_threshold: float = 0.8,
    ev_warning_offset_factor: float = 1.5,
    vg_velocity_threshold_bps: float = 10.0,
    vg_offset_boost_factor: float = 1.5,
    execution_final_clamp_enabled: bool = False,
    alert_offset_mult: float = 1.0,
    last_macro_trend: str | None = None,
    last_vg_triggered: bool = False,
    macro_sell_boost_strong_up: float = 1.0,
    macro_sell_boost_weak_up: float = 1.0,
    macro_buy_boost_strong_down: float = 1.0,
    macro_buy_boost_weak_down: float = 1.0,
) -> OffsetPipelineMixin:
    """テスト用 mixin stub を構築."""
    obj = object.__new__(OffsetPipelineMixin)
    cfg = MagicMock(spec=FillTestConfig)
    cfg.experimental_additive_pipeline = experimental_additive
    cfg.skip_gate_ev_as_offset_enabled = ev_as_offset_enabled
    cfg.skip_gate_ev_offset_sensitivity = ev_offset_sensitivity
    cfg.skip_gate_ev_offset_min_mult = ev_offset_min_mult
    cfg.skip_gate_ev_offset_max_mult = ev_offset_max_mult
    cfg.skip_gate_ev_warning_threshold = ev_warning_threshold
    cfg.skip_gate_ev_warning_offset_factor = ev_warning_offset_factor
    cfg.volatility_guard_velocity_threshold_bps = vg_velocity_threshold_bps
    cfg.volatility_guard_offset_boost_factor = vg_offset_boost_factor
    cfg.execution_final_clamp_enabled = execution_final_clamp_enabled
    cfg.macro_sell_boost_strong_up = macro_sell_boost_strong_up
    cfg.macro_sell_boost_weak_up = macro_sell_boost_weak_up
    cfg.macro_buy_boost_strong_down = macro_buy_boost_strong_down
    cfg.macro_buy_boost_weak_down = macro_buy_boost_weak_down
    obj.config = cfg  # type: ignore[attr-defined]

    maker = MagicMock()
    maker.last_vg_triggered = last_vg_triggered
    maker.last_sigma = 0.01
    maker.get_adverse_ofi = MagicMock(return_value=0.0)
    obj._maker_price = maker  # type: ignore[attr-defined]

    obj._last_macro_trend = last_macro_trend  # type: ignore[attr-defined]
    obj._alert_offset_mult = alert_offset_mult  # type: ignore[attr-defined]

    # _recalc_price_with_new_offset: identity stub
    def _recalc(side: str, order_price: float, spread: float, old_r: float, new_r: float) -> float:
        # 簡易: new_ratio に比例して価格をずらす
        delta = spread * (new_r - old_r) / 2
        return round(order_price - delta) if side == "sell" else round(order_price + delta)

    obj._recalc_price_with_new_offset = _recalc  # type: ignore[attr-defined]
    return obj


_COMMON_KWARGS: dict = dict(
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


# ── A. Dispatcher ──


class TestDispatcher:
    """_apply_offset_pipeline が flag に応じて正しいメソッドを呼ぶ."""

    def test_dispatcher_routes_to_multiplicative_by_default(self) -> None:
        mixin = _make_mixin(experimental_additive=False)
        with pytest.MonkeyPatch.context() as mp:
            called: list[str] = []
            mp.setattr(
                OffsetPipelineMixin,
                "_apply_offset_pipeline_multiplicative",
                lambda self, **kw: (called.append("mult"), OffsetPipelineResult(
                    order_price=0, effective_offset_ratio=0,
                    ev_offset_applied=False, ev_score_pretrade=None,
                    ev_offset_mult_applied=None, macro_boost_applied=False,
                    execution_pre_clamp_offset=None, executor_offset_stages_json=None,
                ))[1],
            )
            mixin._apply_offset_pipeline(**_COMMON_KWARGS)
            assert called == ["mult"]

    def test_dispatcher_routes_to_additive_when_flag_true(self) -> None:
        mixin = _make_mixin(experimental_additive=True)
        with pytest.MonkeyPatch.context() as mp:
            called: list[str] = []
            mp.setattr(
                OffsetPipelineMixin,
                "_apply_offset_pipeline_additive",
                lambda self, **kw: (called.append("add"), OffsetPipelineResult(
                    order_price=0, effective_offset_ratio=0,
                    ev_offset_applied=False, ev_score_pretrade=None,
                    ev_offset_mult_applied=None, macro_boost_applied=False,
                    execution_pre_clamp_offset=None, executor_offset_stages_json=None,
                ))[1],
            )
            mixin._apply_offset_pipeline(**_COMMON_KWARGS)
            assert called == ["add"]

    def test_dispatcher_does_not_call_multiplicative_when_additive_enabled(self) -> None:
        mixin = _make_mixin(experimental_additive=True)
        with pytest.MonkeyPatch.context() as mp:
            mp.setattr(
                OffsetPipelineMixin,
                "_apply_offset_pipeline_multiplicative",
                lambda self, **kw: (_ for _ in ()).throw(AssertionError("multiplicative path called")),
            )
            result = mixin._apply_offset_pipeline(**_COMMON_KWARGS)
        assert result.executor_offset_stages_json is not None


# ── B. RMS 計算ロジック ──


class TestRmsComputation:
    """RMS 結合が正しく動作すること."""

    def test_no_multipliers_returns_base_ratio(self) -> None:
        """全 multiplier ≤ 1.0 → base_ratio が保持される."""
        mixin = _make_mixin(ev_as_offset_enabled=False)
        result = mixin._apply_offset_pipeline_additive(**_COMMON_KWARGS)
        assert result.effective_offset_ratio == pytest.approx(0.05)

    def test_single_toxicity_factor(self) -> None:
        """velocity mult=1.5 → tox_rms = base*(1.5-1.0) = 0.025."""
        mixin = _make_mixin(ev_as_offset_enabled=False)
        kw = {**_COMMON_KWARGS, "sg_velocity_offset_mult": 1.5}
        result = mixin._apply_offset_pipeline_additive(**kw)
        expected = 0.05 + 0.05 * 0.5  # base + sqrt((base*0.5)^2) = base + base*0.5
        assert result.effective_offset_ratio == pytest.approx(expected, abs=1e-6)

    def test_two_toxicity_factors_rms(self) -> None:
        """velocity=1.5, toxicity=2.0 → RMS 結合."""
        mixin = _make_mixin(ev_as_offset_enabled=False)
        kw = {**_COMMON_KWARGS, "sg_velocity_offset_mult": 1.5, "toxicity_offset_mult": 2.0}
        result = mixin._apply_offset_pipeline_additive(**kw)
        d_vel = 0.05 * 0.5
        d_tox = 0.05 * 1.0
        tox_rms = math.sqrt(d_vel**2 + d_tox**2)
        expected = 0.05 + tox_rms
        assert result.effective_offset_ratio == pytest.approx(expected, abs=1e-6)

    def test_toxicity_and_liquidity_independent(self) -> None:
        """tox と liq が独立に RMS 結合される。"""
        from scripts.v460.lib.macro_regime import MacroTrend

        mixin = _make_mixin(
            ev_as_offset_enabled=False,
            last_macro_trend=MacroTrend.STRONG_UP.value,
            macro_sell_boost_strong_up=1.8,
        )
        kw = {
            **_COMMON_KWARGS,
            "sg_velocity_offset_mult": 1.5,  # Toxicity
        }
        result = mixin._apply_offset_pipeline_additive(**kw)
        d_vel = 0.05 * 0.5
        d_macro = 0.05 * 0.8
        expected = 0.05 + math.sqrt(d_vel**2) + math.sqrt(d_macro**2)
        assert result.effective_offset_ratio == pytest.approx(expected, abs=1e-6)

    def test_ev_score_is_classified_into_liquidity_buffer(self) -> None:
        mixin = _make_mixin(
            ev_as_offset_enabled=True,
            ev_offset_sensitivity=1.0,
            ev_offset_min_mult=1.0,
            ev_offset_max_mult=2.0,
        )
        result = mixin._apply_offset_pipeline_additive(
            **{**_COMMON_KWARGS, "sg_ev_score": 1.0},
        )
        stages = json.loads(result.executor_offset_stages_json)  # type: ignore[arg-type]
        assert stages["ev"] is not None
        assert stages["liq_buffer"] > 0.0

    def test_trending_offset_is_ignored_on_buy_side(self) -> None:
        mixin = _make_mixin(ev_as_offset_enabled=False)
        result = mixin._apply_offset_pipeline_additive(
            **{
                **_COMMON_KWARGS,
                "side": "buy",
                "trending_offset_mult": 3.0,
            },
        )
        stages = json.loads(result.executor_offset_stages_json)  # type: ignore[arg-type]
        assert stages["trending"] is None
        assert result.effective_offset_ratio == pytest.approx(0.05, abs=1e-6)


# ── C. Stages JSON 出力 ──


class TestStagesJson:
    """executor_offset_stages_json に tox_buffer / liq_buffer が記録される."""

    def test_stages_json_contains_buffers(self) -> None:
        mixin = _make_mixin(ev_as_offset_enabled=False, alert_offset_mult=1.3)
        result = mixin._apply_offset_pipeline_additive(**_COMMON_KWARGS)
        assert result.executor_offset_stages_json is not None
        stages = json.loads(result.executor_offset_stages_json)
        assert "tox_buffer" in stages
        assert "liq_buffer" in stages
        # alert=1.3 → tox_buffer > 0, liq_buffer = 0
        assert stages["tox_buffer"] > 0
        assert stages["liq_buffer"] == 0.0

    def test_stages_json_records_individual_mults(self) -> None:
        mixin = _make_mixin(ev_as_offset_enabled=False)
        kw = {**_COMMON_KWARGS, "sg_velocity_offset_mult": 1.5}
        result = mixin._apply_offset_pipeline_additive(**kw)
        stages = json.loads(result.executor_offset_stages_json)  # type: ignore[arg-type]
        assert stages["velocity"] == 1.5
        assert stages["trending"] is None
        assert stages["toxicity"] is None

    def test_final_clamp_applies_in_additive_pipeline(self) -> None:
        mixin = _make_mixin(
            ev_as_offset_enabled=False,
            execution_final_clamp_enabled=True,
        )
        mixin.config.resolve_offset_ceiling = MagicMock(return_value=0.06)
        mixin.config.execution_final_clamp_hard_skip_mult = 2.0
        result = mixin._apply_offset_pipeline_additive(
            **{**_COMMON_KWARGS, "sg_velocity_offset_mult": 2.0},
        )
        assert result.execution_pre_clamp_offset == pytest.approx(0.10, abs=1e-6)
        assert result.effective_offset_ratio == pytest.approx(0.06, abs=1e-6)
        assert result.early_return_record is None


# ── D. Sidecar bps ──


class TestSidecar:
    """Sidecar bps が RMS の外側で加算される."""

    def test_sidecar_adjusts_price(self) -> None:
        mixin = _make_mixin(ev_as_offset_enabled=False)
        kw_no_sc = {**_COMMON_KWARGS}
        kw_with_sc = {**_COMMON_KWARGS, "sidecar_offset_bps": 10.0}
        r_no = mixin._apply_offset_pipeline_additive(**kw_no_sc)
        r_sc = mixin._apply_offset_pipeline_additive(**kw_with_sc)
        # sidecar_offset_bps=10 → 13_000_000 * 10/10000 = 13000 delta for sell: price -= delta
        assert r_sc.order_price < r_no.order_price


# ── E. メソッド存在テスト ──


class TestMethodExistence:
    """OffsetPipelineMixin に 3 つのメソッドが存在する."""

    def test_dispatcher_exists(self) -> None:
        assert hasattr(OffsetPipelineMixin, "_apply_offset_pipeline")

    def test_additive_exists(self) -> None:
        assert hasattr(OffsetPipelineMixin, "_apply_offset_pipeline_additive")

    def test_multiplicative_exists(self) -> None:
        assert hasattr(OffsetPipelineMixin, "_apply_offset_pipeline_multiplicative")

    def test_dispatcher_source_references_flag(self) -> None:
        src = inspect.getsource(OffsetPipelineMixin._apply_offset_pipeline)
        assert "experimental_additive_pipeline" in src


# ── F. VG sell supplement (additive path) ──


class TestVgSellSupplementAdditive:
    """202# VG supplement — additive path での Toxicity バッファ加算."""

    def test_vg_supplement_fire(self) -> None:
        """sell + !vg_triggered + |vel|>threshold + vel 未適用 → tox_buffer に加算."""
        mixin = _make_mixin(
            ev_as_offset_enabled=False,
            vg_velocity_threshold_bps=10.0,
            vg_offset_boost_factor=1.5,
        )
        mixin._maker_price.last_vg_triggered = False
        # velocity_offset_mult=None → vel 未適用, velocity_bps=15 > threshold
        result = mixin._apply_offset_pipeline_additive(
            **{**_COMMON_KWARGS, "side": "sell", "sg_velocity_bps": 15.0},
        )
        stages = json.loads(result.executor_offset_stages_json)
        assert stages["vg_supp"] == 1.5
        assert stages["tox_buffer"] > 0.0
        assert result.effective_offset_ratio > 0.05

    def test_vg_supplement_skip_when_boost_factor_1(self) -> None:
        """boost_factor=1.0 → delta=0 → tox_buffer には加算されない."""
        mixin = _make_mixin(
            ev_as_offset_enabled=False,
            vg_velocity_threshold_bps=10.0,
            vg_offset_boost_factor=1.0,
        )
        mixin._maker_price.last_vg_triggered = False
        result = mixin._apply_offset_pipeline_additive(
            **{**_COMMON_KWARGS, "side": "sell", "sg_velocity_bps": 15.0},
        )
        stages = json.loads(result.executor_offset_stages_json)
        # _vg_supp_mult=1.0 だが条件分岐 if _vg_supp_mult > 1.0 で弾かれる
        # → tox_buffer=0
        assert stages["tox_buffer"] == 0.0


# ── G. Macro buy/weak_down (additive path) ──


class TestMacroBuyWeakDownAdditive:
    """458# macro buy/weak_down — additive path."""

    def test_buy_weak_down(self) -> None:
        mixin = _make_mixin(
            ev_as_offset_enabled=False,
            last_macro_trend="macro_weak_down",
            macro_buy_boost_weak_down=1.3,
        )
        result = mixin._apply_offset_pipeline_additive(
            **{**_COMMON_KWARGS, "side": "buy"},
        )
        assert result.macro_boost_applied is True
        stages = json.loads(result.executor_offset_stages_json)
        assert stages["macro"] == 1.3
        assert stages["liq_buffer"] > 0.0

    def test_buy_strong_down(self) -> None:
        mixin = _make_mixin(
            ev_as_offset_enabled=False,
            last_macro_trend="macro_strong_down",
            macro_buy_boost_strong_down=1.4,
        )
        result = mixin._apply_offset_pipeline_additive(
            **{**_COMMON_KWARGS, "side": "buy"},
        )
        assert result.macro_boost_applied is True


# ── H. Alert > 1.0 (additive path) ──


class TestAlertAdditiveOffset:
    """215# alert multiplier — additive path の Toxicity バッファ."""

    def test_alert_offset_tox_buffer(self) -> None:
        mixin = _make_mixin(ev_as_offset_enabled=False, alert_offset_mult=1.5)
        result = mixin._apply_offset_pipeline_additive(**_COMMON_KWARGS)
        stages = json.loads(result.executor_offset_stages_json)
        assert stages["alert"] == 1.5
        assert stages["tox_buffer"] > 0.0

    def test_alert_offset_le_1_no_effect(self) -> None:
        mixin = _make_mixin(ev_as_offset_enabled=False, alert_offset_mult=0.8)
        result = mixin._apply_offset_pipeline_additive(**_COMMON_KWARGS)
        stages = json.loads(result.executor_offset_stages_json)
        assert stages["alert"] is None
        assert stages["tox_buffer"] == 0.0


# ── I. Final clamp hard skip + normal clamp (additive path) ──


class TestFinalClampAdditive:
    """582# additive path の final clamp 経路."""

    def test_hard_skip_returns_early_record(self) -> None:
        """offset >> ceiling × hard_skip_mult → early return."""
        mixin = _make_mixin(
            ev_as_offset_enabled=False,
            execution_final_clamp_enabled=True,
        )
        mixin.config.resolve_offset_ceiling = MagicMock(return_value=0.01)
        mixin.config.execution_final_clamp_hard_skip_mult = 2.0
        mixin._make_cycle_skip_record = MagicMock(return_value=MagicMock())
        # velocity=2.0 → offset ≈ 0.10 >> 0.01*2.0=0.02
        result = mixin._apply_offset_pipeline_additive(
            **{**_COMMON_KWARGS, "sg_velocity_offset_mult": 2.0},
        )
        assert result.early_return_record is not None
        mixin._make_cycle_skip_record.assert_called_once()

    def test_normal_clamp_with_spread(self) -> None:
        """offset > ceiling but < ceiling × hard_skip → clamp + price recalc."""
        mixin = _make_mixin(
            ev_as_offset_enabled=False,
            execution_final_clamp_enabled=True,
        )
        mixin.config.resolve_offset_ceiling = MagicMock(return_value=0.06)
        mixin.config.execution_final_clamp_hard_skip_mult = 5.0
        # velocity=2.0 → offset ≈ 0.10, ceiling=0.06
        result = mixin._apply_offset_pipeline_additive(
            **{**_COMMON_KWARGS, "sg_velocity_offset_mult": 2.0},
        )
        assert result.execution_pre_clamp_offset is not None
        assert result.effective_offset_ratio == pytest.approx(0.06, abs=1e-6)
        assert result.early_return_record is None


# ── J. _scale_lot ──


class TestScaleLot:
    """_scale_lot の挙動テスト."""

    def test_basic_scaling(self) -> None:
        result = OffsetPipelineMixin._scale_lot(
            lot=0.01, scale=0.5, min_lot=0.001, tag="test"
        )
        assert result == pytest.approx(0.005)

    def test_min_lot_guard(self) -> None:
        result = OffsetPipelineMixin._scale_lot(
            lot=0.001, scale=0.1, min_lot=0.005, tag="test"
        )
        # 0.001 * 0.1 = 0.0001 < 0.005 → min_lot を返す
        assert result == pytest.approx(0.005)

    def test_warn_flag(self) -> None:
        """warn=True でも正しい値を返す."""
        result = OffsetPipelineMixin._scale_lot(
            lot=0.01, scale=2.0, min_lot=0.001, tag="test", warn=True
        )
        assert result == pytest.approx(0.02)


# ── K. 追加カバレッジ — trending sell / macro weak_up / sidecar buy ──


class TestTrendingSellAdditive:
    """196# trending offset — additive path sell 側."""

    def test_trending_sell_adds_to_tox_buffer(self) -> None:
        mixin = _make_mixin(ev_as_offset_enabled=False)
        result = mixin._apply_offset_pipeline_additive(
            **{**_COMMON_KWARGS, "side": "sell", "trending_offset_mult": 1.5},
        )
        stages = json.loads(result.executor_offset_stages_json)
        assert stages["trending"] == 1.5
        assert stages["tox_buffer"] > 0.0
        assert result.effective_offset_ratio > 0.05


class TestMacroWeakUpSellAdditive:
    """458# macro sell/weak_up — additive path."""

    def test_sell_weak_up(self) -> None:
        mixin = _make_mixin(
            ev_as_offset_enabled=False,
            last_macro_trend="macro_weak_up",
            macro_sell_boost_weak_up=1.3,
        )
        result = mixin._apply_offset_pipeline_additive(
            **{**_COMMON_KWARGS, "side": "sell"},
        )
        assert result.macro_boost_applied is True
        stages = json.loads(result.executor_offset_stages_json)
        assert stages["macro"] == 1.3
        assert stages["liq_buffer"] > 0.0


class TestSidecarBuyAdditive:
    """Sidecar bps — additive buy."""

    def test_sidecar_buy(self) -> None:
        mixin = _make_mixin(ev_as_offset_enabled=False)
        result = mixin._apply_offset_pipeline_additive(
            **{**_COMMON_KWARGS, "side": "buy", "sidecar_offset_bps": 10.0},
        )
        # buy: price + delta
        assert result.order_price != 13_000_000
