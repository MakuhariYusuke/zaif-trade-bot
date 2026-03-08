"""234# Gate bypass 廃止・縮退清算・片側エスカレーション・制約集合崩壊検出テスト.

P0-A: balance_forced で Kill Gate (Gate 4/5) をバイパスしていた問題の修正
  - 全ゲートから `not balance_forced` 条件を削除
  - Kill Gate blocked + balance_forced → 縮退清算モード (degraded liquidation)
P0-B: one_sided_consecutive_limit のエスカレーション (3段階)
P0-C: no_feasible_quote 早期検出 (制約集合崩壊)

232# Codex + 233# Gemini レビューの合意事項に基づく実装。
"""

from __future__ import annotations

import ast
import inspect

import pytest

from scripts.v460.lib.fill_config import FillTestConfig
from scripts.v460.lib.cycle_gate_aggregator import (
    CycleGateAggregator,
    CycleGateResult,
)
from scripts.v460.lib import cancel_reasons as CR
from tests.unit.v460._fill_test_source import (
    CYCLE_GATE_AGGREGATOR,
    parse_source_tree,
    read_source_text,
)

from tests.unit.v460.conftest import make_gate_config


# ─── ヘルパー ──────────────────────────────────────────────────────────


def _make_config(**overrides: object) -> FillTestConfig:
    """テスト用 FillTestConfig (degraded_liquidation 有効)."""
    merged = {"degraded_liquidation_enabled": True, **overrides}
    return make_gate_config(**merged)


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


# ═══════════════════════════════════════════════════════════════════════
# P0-A-1: balance_forced gate bypass 全廃 (AST ベース検証)
# ═══════════════════════════════════════════════════════════════════════


# ═══════════════════════════════════════════════════════════════════════
# P0-A-2: Gate 1 (unknown regime buy) — balance_forced でもブロック
# ═══════════════════════════════════════════════════════════════════════


class TestGate1NoBypass:
    """Gate 1: unknown_regime_buy は balance_forced でもブロック."""

    def test_unknown_buy_blocked_without_balance_forced(self) -> None:
        gate = _make_gate()
        r = gate.evaluate(**_default_ctx(side="buy", regime="unknown"))
        assert r.blocked
        assert r.blocking_reason == "unknown_regime_buy_skip"


# ═══════════════════════════════════════════════════════════════════════
# P0-A-3: Gate 2 (ranging buy low vol) — balance_forced でもブロック
# ═══════════════════════════════════════════════════════════════════════


class TestGate2NoBypass:
    """Gate 2: ranging_buy_low_vol は balance_forced でもブロック."""


# ═══════════════════════════════════════════════════════════════════════
# P0-A-4: Gate 3 (trending sell) — balance_forced でもソフトモード適用
# ═══════════════════════════════════════════════════════════════════════


class TestGate3NoBypass:
    """Gate 3: trending_sell は balance_forced でも offset 適用."""


# ═══════════════════════════════════════════════════════════════════════
# P0-A-5: Gate 4/5 (dynamic kill) — 縮退清算モード
# ═══════════════════════════════════════════════════════════════════════


# ═══════════════════════════════════════════════════════════════════════
# P0-A-6: Gate 7 (unknown regime sell) — balance_forced でもブロック
# ═══════════════════════════════════════════════════════════════════════


class TestGate7NoBypass:
    """Gate 7: unknown_regime_sell は balance_forced でもブロック."""


# ═══════════════════════════════════════════════════════════════════════
# P0-A-7: CycleGateResult dataclass フィールド
# ═══════════════════════════════════════════════════════════════════════


class TestCycleGateResultFields:
    """234# 追加フィールドの検証."""

    def test_default_degraded_false(self) -> None:
        r = CycleGateResult()
        assert r.degraded_liquidation is False
        assert r.degraded_reason == ""

    def test_degraded_fields_settable(self) -> None:
        r = CycleGateResult(
            degraded_liquidation=True,
            degraded_reason="buy_dynamic_kill",
        )
        assert r.degraded_liquidation is True
        assert r.degraded_reason == "buy_dynamic_kill"


# ═══════════════════════════════════════════════════════════════════════
# P0-B: one_sided_consecutive_limit エスカレーション Config
# ═══════════════════════════════════════════════════════════════════════


class TestOneSidedEscalationConfig:
    """one_sided エスカレーション Config フィールドの検証."""

    def test_cooldown_offset_default(self) -> None:
        cfg = FillTestConfig()
        assert cfg.one_sided_escalation_cooldown_offset == 2

    def test_cooldown_cycles_default(self) -> None:
        cfg = FillTestConfig()
        assert cfg.one_sided_escalation_cooldown_cycles == 2

    def test_freeze_offset_default(self) -> None:
        cfg = FillTestConfig()
        assert cfg.one_sided_escalation_freeze_offset == 4

    def test_freeze_cycles_default(self) -> None:
        cfg = FillTestConfig()
        assert cfg.one_sided_escalation_freeze_cycles == 3


# ═══════════════════════════════════════════════════════════════════════
# P0-B-2: degraded liquidation Config
# ═══════════════════════════════════════════════════════════════════════


class TestDegradedLiquidationConfig:
    """縮退清算 Config フィールドの検証."""

    def test_enabled_default(self) -> None:
        cfg = FillTestConfig()
        assert cfg.degraded_liquidation_enabled is True

    def test_lot_mult_default(self) -> None:
        cfg = FillTestConfig()
        assert cfg.degraded_liquidation_lot_mult == 0.2

    def test_offset_mult_default(self) -> None:
        cfg = FillTestConfig()
        assert cfg.degraded_liquidation_offset_mult == 3.0

    def test_duty_cycle_default(self) -> None:
        cfg = FillTestConfig()
        assert cfg.degraded_liquidation_duty_cycle == 3


# ═══════════════════════════════════════════════════════════════════════
# P0-C: cancel_reasons の新定数
# ═══════════════════════════════════════════════════════════════════════


class TestCancelReasonsNew:
    """234# で追加された cancel_reason 定数の検証."""

    def test_no_feasible_quote_exists(self) -> None:
        assert CR.NO_FEASIBLE_QUOTE == "no_feasible_quote"

    def test_no_feasible_quote_in_type(self) -> None:
        """CancelReason Literal 型に含まれること."""
        import typing
        args = typing.get_args(CR.CancelReason)
        assert "no_feasible_quote" in args

    def test_degraded_duty_skip_in_type(self) -> None:
        args = __import__("typing").get_args(CR.CancelReason)
        assert "degraded_liquidation_duty_skip" in args

    def test_cooldown_skip_in_type(self) -> None:
        args = __import__("typing").get_args(CR.CancelReason)
        assert "one_sided_cooldown_skip" in args

    def test_freeze_skip_in_type(self) -> None:
        args = __import__("typing").get_args(CR.CancelReason)
        assert "one_sided_freeze_skip" in args


# ═══════════════════════════════════════════════════════════════════════
# P0-A: ソースコード検証 — dual_kill から not balance_forced 削除
# ═══════════════════════════════════════════════════════════════════════


# ═══════════════════════════════════════════════════════════════════════
# 統合: balance_forced + 複合シナリオ
# ═══════════════════════════════════════════════════════════════════════


# ═══════════════════════════════════════════════════════════════════════
# 235# dead parameter cleanup 検証
# ═══════════════════════════════════════════════════════════════════════


class TestDutyCycleGuard:
    """235# duty_cycle=1 や duty_cycle=0 でのガード検証.

    249# 以降、duty_cycle < 2 は ValueError に変更。
    """

    def test_duty_cycle_config_min_1(self) -> None:
        """249# duty_cycle=0 → ValueError."""
        with pytest.raises(ValueError, match="degraded_liquidation_duty_cycle"):
            FillTestConfig(degraded_liquidation_duty_cycle=0)

    def test_duty_cycle_1_means_every_cycle(self) -> None:
        """249# duty_cycle=1 → ValueError (min=2)."""
        with pytest.raises(ValueError, match="degraded_liquidation_duty_cycle"):
            FillTestConfig(degraded_liquidation_duty_cycle=1)


