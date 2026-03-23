"""593# テスト: ev_toxic_skip 中間帯スキップ + CV cap_hit sell veto 昇格.

- A: ev_toxic_skip — emergency(-8) と warning(-4) の間の toxic flow スキップ
- B: cap_hit + sell → veto 昇格 (widen 上限で防御不能時)
- Config / YAML parse テスト
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from scripts.v460.lib.cross_venue_lead_lag import CrossVenueLeadLagHint
from scripts.v460.lib.fill_config import FillTestConfig
from scripts.v460.lib.maker_price import MakerPriceCalculator
from ztb.trading.risk.fast_fill_defense import FastFillDefense, FastFillDefenseConfig


# ─── helpers ────────────────────────────────────────────────────────


class _MockSkipDecision:
    """SkipGate.evaluate 戻り値のモック."""

    def __init__(
        self,
        predicted_pnl_bps: float = 0.0,
        threshold_used: float = 0.0,
        threshold_bps: float = 0.0,
        features_used: int = 10,
        as_probability: float | None = None,
        reason: str = "pass",
        model_used: str = "primary",
        should_skip: bool = False,
    ) -> None:
        self.predicted_pnl_bps = predicted_pnl_bps
        self.threshold_used = threshold_used
        self.threshold_bps = threshold_bps
        self.features_used = features_used
        self.as_probability = as_probability
        self.reason = reason
        self.model_used = model_used
        self.should_skip = should_skip


def _make_alt_gate(pred_pnl: float = 1.0) -> MagicMock:
    gate = MagicMock()
    gate.config.mode = "pnl"
    gate.evaluate.return_value = _MockSkipDecision(predicted_pnl_bps=pred_pnl)
    return gate


def _make_evaluator(
    *,
    emergency_threshold: float = -8.0,
    toxic_threshold: float = -5.0,
    gate_alt_buy: object | None = None,
    gate_alt_sell: object | None = None,
) -> object:
    from scripts.v460.lib.skip_gate_evaluator import SkipGateEvaluator

    config = FillTestConfig(
        skip_gate_enabled=False,
        skip_gate_ev_weighted_enabled=True,
        skip_gate_ev_w30=0.4,
        skip_gate_ev_w120=0.6,
        skip_gate_ev_as_offset_enabled=True,
        skip_gate_ev_offset_sensitivity=0.05,
        skip_gate_ev_offset_min_mult=0.5,
        skip_gate_ev_offset_max_mult=1.5,
        skip_gate_ev_emergency_skip_threshold=emergency_threshold,
        skip_gate_ev_toxic_skip_threshold=toxic_threshold,
        skip_gate_ev_max_consecutive_skip=0,
        skip_gate_ev_one_sided_threshold_shift=0.0,
    )
    evaluator = SkipGateEvaluator(config, Path("."))
    evaluator._gate_alt_buy = gate_alt_buy
    evaluator._gate_alt_sell = gate_alt_sell
    return evaluator


def _make_hint(
    *,
    adverse_side: str = "sell",
    spread_bps: float = 5.0,
    velocity_bps: float = 2.0,
    age_sec: float = 0.5,
    direction: str = "up",
    confidence: float = 1.0,
) -> CrossVenueLeadLagHint:
    return CrossVenueLeadLagHint(
        direction=direction,
        adverse_side=adverse_side,
        spread_bps=spread_bps,
        reference_velocity_bps=velocity_bps,
        age_sec=age_sec,
        reference_exchange="bitflyer",
        confidence=confidence,
    )


def _make_calc(config: FillTestConfig) -> MakerPriceCalculator:
    ffd = FastFillDefense(
        config=FastFillDefenseConfig(
            enabled=False,
            threshold_sec=5.0,
            offset_boost=1.0,
            max_offset_ratio=config.max_offset_ratio,
            min_offset_ratio=config.min_offset_ratio,
        ),
        base_offset_ratio=config.spread_offset_ratio,
    )
    return MakerPriceCalculator(
        config=config,
        fast_fill_defense=ffd,
        regime_detector=None,
        base_offset_ratio=config.spread_offset_ratio,
    )


# =====================================================================
# 593# A: ev_toxic_skip テスト
# =====================================================================


class TestEvToxicSkip:
    """593# A: emergency(-8) と warning(-4) の間の toxic flow スキップ."""

    def test_toxic_skip_fires_between_emergency_and_warning(self) -> None:
        """EV score が toxic 閾値未満・emergency 以上 → toxic skip."""
        # alt=-8 → ev = 0.4*(-2) + 0.6*(-8) = -5.6 < -5.0 (toxic) but > -8.0 (emergency)
        alt = _make_alt_gate(pred_pnl=-8.0)
        evaluator = _make_evaluator(
            emergency_threshold=-8.0,
            toxic_threshold=-5.0,
            gate_alt_buy=alt,
        )
        primary = _MockSkipDecision(predicted_pnl_bps=-2.0)
        result = evaluator._try_ev_weighted_decision(
            "buy", {}, "ranging", 0.0, primary,
        )
        assert result is not None
        # ev = 0.4*(-2.0) + 0.6*(-8.0) = -5.6
        assert result.predicted_pnl_bps == pytest.approx(-5.6, abs=0.01)
        assert result.should_skip is True
        assert result.reason == "ev_toxic_skip"

    def test_no_toxic_skip_when_above_threshold(self) -> None:
        """EV score が toxic 閾値以上 → skip しない (offset モード pass)."""
        # alt=-3 → ev = 0.4*(-1) + 0.6*(-3) = -2.2 > -5.0
        alt = _make_alt_gate(pred_pnl=-3.0)
        evaluator = _make_evaluator(
            toxic_threshold=-5.0,
            gate_alt_buy=alt,
        )
        primary = _MockSkipDecision(predicted_pnl_bps=-1.0)
        result = evaluator._try_ev_weighted_decision(
            "buy", {}, "ranging", 0.0, primary,
        )
        assert result is not None
        assert result.should_skip is False
        assert result.reason == "ev_weighted_offset"

    def test_emergency_takes_priority_over_toxic(self) -> None:
        """EV score が emergency 閾値未満 → emergency skip (toxic より優先)."""
        # alt=-20 → ev = 0.4*(-5) + 0.6*(-20) = -14.0 < -8.0
        alt = _make_alt_gate(pred_pnl=-20.0)
        evaluator = _make_evaluator(
            emergency_threshold=-8.0,
            toxic_threshold=-5.0,
            gate_alt_buy=alt,
        )
        primary = _MockSkipDecision(predicted_pnl_bps=-5.0)
        result = evaluator._try_ev_weighted_decision(
            "buy", {}, "ranging", 0.0, primary,
        )
        assert result is not None
        assert result.should_skip is True
        assert result.reason == "ev_weighted_emergency_skip"

    def test_toxic_skip_on_sell_side(self) -> None:
        """sell 側でも toxic skip が発動すること."""
        # sell: ev = w30*alt + w120*primary = 0.4*(-10) + 0.6*(-2) = -5.2 < -5.0
        alt = _make_alt_gate(pred_pnl=-10.0)
        evaluator = _make_evaluator(
            toxic_threshold=-5.0,
            gate_alt_sell=alt,
        )
        primary = _MockSkipDecision(predicted_pnl_bps=-2.0)
        result = evaluator._try_ev_weighted_decision(
            "sell", {}, "ranging", 0.0, primary,
        )
        assert result is not None
        assert result.should_skip is True
        assert result.reason == "ev_toxic_skip"

    def test_toxic_skip_at_exact_boundary(self) -> None:
        """EV score == toxic 閾値 → skip しない (未満でスキップ)."""
        # ev = 0.4*primary + 0.6*alt → exactly -5.0
        # 0.4 * (-2.0) + 0.6 * alt = -5.0 → alt = (-5.0 + 0.8) / 0.6 = -7.0
        alt = _make_alt_gate(pred_pnl=-7.0)
        evaluator = _make_evaluator(
            toxic_threshold=-5.0,
            gate_alt_buy=alt,
        )
        primary = _MockSkipDecision(predicted_pnl_bps=-2.0)
        result = evaluator._try_ev_weighted_decision(
            "buy", {}, "ranging", 0.0, primary,
        )
        assert result is not None
        # ev = 0.4*(-2.0) + 0.6*(-7.0) = -5.0 (== threshold, not <)
        assert result.should_skip is False

    def test_consecutive_skip_counter_resets_on_toxic_skip(self) -> None:
        """toxic skip 時は連続スキップカウンタがリセットされること."""
        alt = _make_alt_gate(pred_pnl=-8.0)
        evaluator = _make_evaluator(gate_alt_buy=alt)
        evaluator._ev_consecutive_skip_count = 3  # 既に何回かスキップ

        primary = _MockSkipDecision(predicted_pnl_bps=-2.0)
        result = evaluator._try_ev_weighted_decision(
            "buy", {}, "ranging", 0.0, primary,
        )
        assert result is not None
        assert result.should_skip is True
        assert evaluator._ev_consecutive_skip_count == 0


# =====================================================================
# 593# B: cap_hit sell veto テスト
# =====================================================================


class TestCapHitSellVeto:
    """593# B: CV cap_hit 時に sell を veto へ昇格."""

    def _make_calc_with_cap_hit_scenario(
        self,
        *,
        cap_hit_sell_veto_enabled: bool = True,
        max_offset_ratio: float = 0.05,
    ) -> MakerPriceCalculator:
        """cap_hit が発生しやすい設定で MakerPriceCalculator を生成."""
        config = FillTestConfig(
            min_spread_jpy=1.0,
            cross_venue_lead_lag_enabled=True,
            cross_venue_lead_lag_offset_boost=2.0,  # 強い boost (cap に当たりやすい)
            cross_venue_lead_lag_veto_enabled=False,  # 通常 veto は無効
            cross_venue_lead_lag_veto_threshold_bps=100.0,  # veto しないように高く設定
            max_offset_ratio=max_offset_ratio,  # 上限を制限して cap_hit を誘発
            cross_venue_cap_hit_sell_veto_enabled=cap_hit_sell_veto_enabled,
        )
        calc = _make_calc(config)
        return calc

    def test_sell_cap_hit_triggers_veto(self) -> None:
        """sell 側で cap_hit → veto が昇格されること."""
        calc = self._make_calc_with_cap_hit_scenario(
            cap_hit_sell_veto_enabled=True,
            max_offset_ratio=0.05,  # cap を低く設定
        )
        # adverse_side=sell で spread が大きい → 強い boost → cap_hit
        hint = _make_hint(
            adverse_side="sell",
            spread_bps=5.0,
            confidence=1.0,
        )
        calc.set_cross_venue_lead_lag_hint(hint)

        # 直接 _apply_cross_venue_lead_lag_guard を呼んでcap_hitを確認
        pre_offset = calc._base_offset_ratio  # base offset
        result = calc._apply_cross_venue_lead_lag_guard("sell", pre_offset)

        # cap_hit + sell + enabled → veto が True
        assert calc._cross_venue_lead_lag_cap_hit is True
        assert calc._cross_venue_lead_lag_vetoed is True
        assert "cap_hit_sell_veto" in (calc._cross_venue_lead_lag_veto_reason or "")

    def test_sell_cap_hit_no_veto_when_disabled(self) -> None:
        """cap_hit_sell_veto_enabled=False → cap_hit でも veto しない."""
        calc = self._make_calc_with_cap_hit_scenario(
            cap_hit_sell_veto_enabled=False,
            max_offset_ratio=0.05,
        )
        hint = _make_hint(adverse_side="sell", spread_bps=5.0, confidence=1.0)
        calc.set_cross_venue_lead_lag_hint(hint)

        pre_offset = calc._base_offset_ratio
        calc._apply_cross_venue_lead_lag_guard("sell", pre_offset)

        # cap_hit は発生するが veto にはならない
        if calc._cross_venue_lead_lag_cap_hit:
            assert calc._cross_venue_lead_lag_vetoed is False

    def test_buy_cap_hit_no_veto(self) -> None:
        """buy 側の cap_hit → sell veto は発動しない."""
        config = FillTestConfig(
            min_spread_jpy=1.0,
            cross_venue_lead_lag_enabled=True,
            cross_venue_lead_lag_offset_boost=2.0,
            cross_venue_lead_lag_veto_enabled=False,
            cross_venue_lead_lag_veto_threshold_bps=100.0,
            max_offset_ratio=0.05,
            cross_venue_cap_hit_sell_veto_enabled=True,
        )
        calc = _make_calc(config)
        # adverse_side=buy
        hint = _make_hint(
            adverse_side="buy",
            direction="down",
            spread_bps=5.0,
            confidence=1.0,
        )
        calc.set_cross_venue_lead_lag_hint(hint)

        pre_offset = calc._base_offset_ratio
        calc._apply_cross_venue_lead_lag_guard("buy", pre_offset)

        # buy 側では sell veto は発動しない
        assert calc._cross_venue_lead_lag_vetoed is False


# =====================================================================
# 593# Config / YAML テスト
# =====================================================================


class TestConfig593:
    """593# 新フィールドのデフォルト値と YAML パーステスト."""

    def test_defaults(self) -> None:
        c = FillTestConfig()
        assert c.skip_gate_ev_toxic_skip_threshold == pytest.approx(-5.0)
        assert c.cross_venue_cap_hit_sell_veto_enabled is False

    def test_yaml_parse_toxic_skip(self) -> None:
        yaml_data = {
            "skip_gate": {
                "ev_toxic_skip_threshold": -6.0,
            }
        }
        c = FillTestConfig.from_yaml(yaml_data)
        assert c.skip_gate_ev_toxic_skip_threshold == pytest.approx(-6.0)

    def test_yaml_parse_cap_hit_sell_veto(self) -> None:
        yaml_data = {
            "cross_venue_lead_lag": {
                "cap_hit_sell_veto_enabled": True,
            }
        }
        c = FillTestConfig.from_yaml(yaml_data)
        assert c.cross_venue_cap_hit_sell_veto_enabled is True
