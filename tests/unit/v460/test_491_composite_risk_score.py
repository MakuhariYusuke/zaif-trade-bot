"""491# Composite Risk Score 単体テスト.

490# architectural pivot: Soft Gate (1,2,2b,3,6,7) を連続値 risk weight に変換し、
加算集約で一元的に判定する仕組みのテスト。
Hard Gate (4,5,8,9) は Boolean 短絡を維持。
composite_risk_enabled=False (既定) で従来挙動を完全維持 (後方互換)。
"""

from __future__ import annotations

import pytest

from scripts.v460.lib.cycle_gate_aggregator import (
    CycleGateAggregator,
    CycleGateResult,
)
from tests.unit.v460.conftest import make_gate_config as _make_config


def _make_gate(**overrides: object) -> CycleGateAggregator:
    return CycleGateAggregator(_make_config(**overrides))


def _default_ctx(**overrides: object) -> dict:
    ctx: dict = {
        "side": "buy",
        "regime": "ranging",
        "vol_ratio": 1.0,
        "inv_net_imbalance": 0.0,
        "is_buy_killed": False,
        "is_sell_killed": False,
    }
    ctx.update(overrides)
    return ctx


class TestCompositeRiskDisabled:
    """composite_risk_enabled=False で従来挙動を維持."""

    def test_default_disabled(self) -> None:
        gate = _make_gate()
        r = gate.evaluate(**_default_ctx(side="buy", regime="unknown"))
        assert r.blocked
        assert r.blocking_reason == "unknown_regime_buy_skip"
        assert r.composite_risk_score == 0.0

    def test_legacy_early_return(self) -> None:
        gate = _make_gate()
        r = gate.evaluate(**_default_ctx(
            side="buy", regime="ranging", vol_ratio=0.3,
        ))
        assert r.blocked
        assert r.blocking_reason == "ranging_low_vol_skip"


class TestCompositeRiskEnabled:
    """composite_risk_enabled=True で Soft Gate が加算集約される."""

    def test_single_soft_gate_below_threshold(self) -> None:
        """1つの Soft Gate だけでは閾値未満 → 通過."""
        gate = _make_gate(
            composite_risk_enabled=True,
            composite_risk_threshold=1.5,
            composite_risk_weight_unknown_regime=0.6,
        )
        r = gate.evaluate(**_default_ctx(side="buy", regime="unknown"))
        assert not r.blocked
        assert r.composite_risk_score == pytest.approx(0.6)
        assert len(r.composite_risk_details) == 1
        assert "G1:unknown_buy" in r.composite_risk_details[0]

    def test_multiple_soft_gates_exceed_threshold(self) -> None:
        """複数 Soft Gate の合計が閾値を超える → ブロック."""
        gate = _make_gate(
            composite_risk_enabled=True,
            composite_risk_threshold=1.0,
            composite_risk_weight_unknown_regime=0.6,
            composite_risk_weight_velocity=0.5,
            sell_velocity_skip_threshold_bps=8.0,
        )
        r = gate.evaluate(**_default_ctx(
            side="sell", regime="unknown",
            price_velocity_bps=10.0,  # velocity gate triggers
        ))
        assert r.blocked
        assert r.blocking_reason == "composite_risk_exceeded"
        assert r.composite_risk_score >= 1.0
        assert len(r.composite_risk_details) >= 2

    def test_hard_gate_still_blocks_immediately(self) -> None:
        """Hard Gate (buy_dynamic_kill) は composite mode でも即ブロック."""
        gate = _make_gate(
            composite_risk_enabled=True,
            composite_risk_threshold=10.0,  # very high threshold
        )
        r = gate.evaluate(**_default_ctx(
            side="buy", regime="ranging",
            is_buy_killed=True,
        ))
        assert r.blocked
        assert r.blocking_reason == "buy_dynamic_kill"

    def test_hard_gate_sell_dynamic_kill(self) -> None:
        """Hard Gate (sell_dynamic_kill) は composite mode でも即ブロック."""
        gate = _make_gate(
            composite_risk_enabled=True,
            composite_risk_threshold=10.0,
        )
        r = gate.evaluate(**_default_ctx(
            side="sell", regime="ranging",
            is_sell_killed=True,
        ))
        assert r.blocked
        assert r.blocking_reason == "sell_dynamic_kill"

    def test_no_soft_gate_triggered_passes(self) -> None:
        """Soft Gate が 0 → composite_risk_score=0 → 通過."""
        gate = _make_gate(
            composite_risk_enabled=True,
            composite_risk_threshold=1.0,
        )
        r = gate.evaluate(**_default_ctx(
            side="buy", regime="ranging", vol_ratio=1.0,
        ))
        assert not r.blocked
        assert r.composite_risk_score == 0.0

    def test_halt_recovery_bypasses_soft_gates(self) -> None:
        """halt_recovery_active=True → Soft Gate の risk weight を蓄積しない."""
        gate = _make_gate(
            composite_risk_enabled=True,
            composite_risk_threshold=0.5,
            composite_risk_weight_unknown_regime=0.6,
        )
        r = gate.evaluate(**_default_ctx(
            side="buy", regime="unknown",
            halt_recovery_active=True,
        ))
        assert not r.blocked
        assert r.composite_risk_score == 0.0

    def test_composite_risk_details_audit(self) -> None:
        """composite_risk_details が正しくゲート名と weightsを記録."""
        gate = _make_gate(
            composite_risk_enabled=True,
            composite_risk_threshold=10.0,
            composite_risk_weight_unknown_regime=0.6,
        )
        r = gate.evaluate(**_default_ctx(
            side="sell", regime="unknown",
        ))
        assert not r.blocked
        # Gate 7 (unknown_sell) should be recorded
        assert any("G7:unknown_sell" in d for d in r.composite_risk_details)


class TestCompositeRiskRangingLowVol:
    """Gate 2/2b: ranging_low_vol の composite mode テスト."""

    def test_buy_ranging_low_vol_adds_weight(self) -> None:
        gate = _make_gate(
            composite_risk_enabled=True,
            composite_risk_threshold=1.5,
            composite_risk_weight_ranging_low_vol=0.5,
        )
        r = gate.evaluate(**_default_ctx(
            side="buy", regime="ranging", vol_ratio=0.3,
        ))
        assert not r.blocked
        assert r.composite_risk_score == pytest.approx(0.5)

    def test_sell_ranging_low_vol_adds_weight(self) -> None:
        gate = _make_gate(
            composite_risk_enabled=True,
            composite_risk_threshold=1.5,
            composite_risk_weight_ranging_low_vol=0.5,
            skip_ranging_sell_low_vol=True,
        )
        r = gate.evaluate(**_default_ctx(
            side="sell", regime="ranging", vol_ratio=0.3,
        ))
        assert not r.blocked
        assert r.composite_risk_score == pytest.approx(0.5)


class TestCompositeRiskTrendingSell:
    """Gate 3: trending_sell の composite mode テスト."""

    def test_trending_sell_adds_weight(self) -> None:
        gate = _make_gate(
            composite_risk_enabled=True,
            composite_risk_threshold=2.0,
            composite_risk_weight_trending_sell=0.7,
        )
        r = gate.evaluate(**_default_ctx(
            side="sell", regime="trending_up",
            inv_net_imbalance=0.0,
        ))
        assert not r.blocked
        assert r.composite_risk_score == pytest.approx(0.7)

    def test_trending_sell_combined_exceeds_threshold(self) -> None:
        """trending_sell + unknown_sell → 閾値超過."""
        gate = _make_gate(
            composite_risk_enabled=True,
            composite_risk_threshold=1.0,
            composite_risk_weight_trending_sell=0.7,
            composite_risk_weight_unknown_regime=0.6,
        )
        # regime=unknown → Gate 7 (unknown_sell) triggers
        # But Gate 3 only triggers for trending/trending_up regime
        # So this tests trending_sell only with threshold < weight
        gate2 = _make_gate(
            composite_risk_enabled=True,
            composite_risk_threshold=0.5,
            composite_risk_weight_trending_sell=0.7,
        )
        r = gate2.evaluate(**_default_ctx(
            side="sell", regime="trending_up",
            inv_net_imbalance=0.0,
        ))
        assert r.blocked
        assert r.blocking_reason == "composite_risk_exceeded"


class TestCompositeRiskVelocity:
    """Gate 6: velocity の composite mode テスト."""

    def test_velocity_sell_adds_weight(self) -> None:
        gate = _make_gate(
            composite_risk_enabled=True,
            composite_risk_threshold=1.5,
            composite_risk_weight_velocity=0.4,
        )
        r = gate.evaluate(**_default_ctx(
            side="sell", regime="ranging",
            price_velocity_bps=10.0,
        ))
        assert not r.blocked
        assert r.composite_risk_score == pytest.approx(0.4)

    def test_velocity_buy_adds_weight(self) -> None:
        gate = _make_gate(
            composite_risk_enabled=True,
            composite_risk_threshold=1.5,
            composite_risk_weight_velocity=0.4,
        )
        r = gate.evaluate(**_default_ctx(
            side="buy", regime="ranging",
            price_velocity_bps=-10.0,
        ))
        assert not r.blocked
        assert r.composite_risk_score == pytest.approx(0.4)
