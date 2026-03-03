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
import textwrap

import pytest

from scripts.v460.lib.fill_config import FillTestConfig
from scripts.v460.lib.cycle_gate_aggregator import (
    CycleGateAggregator,
    CycleGateResult,
)
from scripts.v460.lib import cancel_reasons as CR


# ─── ヘルパー ──────────────────────────────────────────────────────────


def _make_config(**overrides: object) -> FillTestConfig:
    """テスト用の最小 FillTestConfig."""
    defaults: dict[str, object] = {
        "skip_buy_unknown_regime": True,
        "skip_ranging_buy_low_vol": True,
        "low_vol_threshold": 0.75,
        "skip_sell_trending": True,
        "skip_sell_trending_up_only": False,
        "max_consecutive_trending_sell_skip": 30,
        "sell_guard_inv_bypass_threshold": 0.3,
        "buy_dynamic_kill_enabled": True,
        "sell_dynamic_kill_enabled": True,
        "buy_dynamic_kill_threshold_bps": -5.0,
        "sell_dynamic_kill_threshold_bps": -5.0,
        "sell_velocity_skip_enabled": True,
        "sell_velocity_skip_threshold_bps": 8.0,
        "buy_velocity_skip_enabled": True,
        "buy_velocity_skip_threshold_bps": -8.0,
        "skip_sell_unknown_regime": True,
        "degraded_liquidation_enabled": True,
    }
    defaults.update(overrides)
    return FillTestConfig(**defaults)


def _make_gate(**overrides: object) -> CycleGateAggregator:
    return CycleGateAggregator(_make_config(**overrides))


def _default_ctx(**overrides: object) -> dict:
    ctx: dict = {
        "side": "buy",
        "regime": "ranging",
        "vol_ratio": 1.0,
        "balance_forced": False,
        "inv_net_imbalance": 0.0,
        "is_buy_killed": False,
        "is_sell_killed": False,
    }
    ctx.update(overrides)
    return ctx


# ═══════════════════════════════════════════════════════════════════════
# P0-A-1: balance_forced gate bypass 全廃 (AST ベース検証)
# ═══════════════════════════════════════════════════════════════════════


class TestBalanceForcedBypassEradication:
    """balance_forced が Gate 条件で参照されていないことを AST で検証."""

    def test_no_balance_forced_in_gate_check_conditions(self) -> None:
        """全 _check_* メソッドの gate 条件で `not balance_forced` がないこと."""
        src = inspect.getsource(CycleGateAggregator)
        tree = ast.parse(textwrap.dedent(src))

        violations: list[str] = []
        for node in ast.walk(tree):
            if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.Not):
                if isinstance(node.operand, ast.Name) and node.operand.id == "balance_forced":
                    violations.append(f"line {node.lineno}: not balance_forced")

        assert violations == [], (
            f"234# balance_forced bypass が残存: {violations}"
        )


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

    def test_unknown_buy_blocked_with_balance_forced(self) -> None:
        """234#: balance_forced=True でも unknown buy はブロック."""
        gate = _make_gate()
        r = gate.evaluate(**_default_ctx(
            side="buy", regime="unknown", balance_forced=True,
        ))
        assert r.blocked
        assert r.blocking_reason == "unknown_regime_buy_skip"


# ═══════════════════════════════════════════════════════════════════════
# P0-A-3: Gate 2 (ranging buy low vol) — balance_forced でもブロック
# ═══════════════════════════════════════════════════════════════════════


class TestGate2NoBypass:
    """Gate 2: ranging_buy_low_vol は balance_forced でもブロック."""

    def test_ranging_low_vol_blocked_with_balance_forced(self) -> None:
        gate = _make_gate(ranging_buy_low_vol_as_offset=False)
        r = gate.evaluate(**_default_ctx(
            side="buy", regime="ranging", vol_ratio=0.5,
            balance_forced=True,
        ))
        assert r.blocked
        assert r.blocking_reason == "ranging_low_vol_skip"


# ═══════════════════════════════════════════════════════════════════════
# P0-A-4: Gate 3 (trending sell) — balance_forced でもソフトモード適用
# ═══════════════════════════════════════════════════════════════════════


class TestGate3NoBypass:
    """Gate 3: trending_sell は balance_forced でも offset 適用."""

    def test_trending_sell_soft_applies_with_balance_forced(self) -> None:
        """soft mode enabled → balance_forced でも offset 乗数が返る."""
        gate = _make_gate(
            trending_sell_as_offset_enabled=True,
            trending_sell_offset_boost_factor=2.5,
        )
        r = gate.evaluate(**_default_ctx(
            side="sell", regime="trending_up", balance_forced=True,
        ))
        assert not r.blocked
        assert r.trending_offset_mult == 2.5

    def test_trending_sell_hard_blocks_with_balance_forced(self) -> None:
        """hard mode → balance_forced でもブロック."""
        gate = _make_gate(
            trending_sell_as_offset_enabled=False,
            max_consecutive_trending_sell_skip=999,
        )
        r = gate.evaluate(**_default_ctx(
            side="sell", regime="trending_up", balance_forced=True,
        ))
        assert r.blocked
        assert r.blocking_reason == "trending_sell_skip"


# ═══════════════════════════════════════════════════════════════════════
# P0-A-5: Gate 4/5 (dynamic kill) — 縮退清算モード
# ═══════════════════════════════════════════════════════════════════════


class TestDegradedLiquidationGateLevel:
    """Kill Gate blocked + balance_forced → 縮退清算モード."""

    def test_buy_kill_balance_forced_degraded(self) -> None:
        """buy kill + balance_forced → degraded_liquidation=True."""
        gate = _make_gate()
        r = gate.evaluate(**_default_ctx(
            side="buy", is_buy_killed=True, balance_forced=True,
        ))
        assert not r.blocked
        assert r.degraded_liquidation is True
        assert r.degraded_reason == "buy_dynamic_kill"

    def test_sell_kill_balance_forced_degraded(self) -> None:
        """sell kill + balance_forced → degraded_liquidation=True."""
        gate = _make_gate()
        r = gate.evaluate(**_default_ctx(
            side="sell", is_sell_killed=True, balance_forced=True,
        ))
        assert not r.blocked
        assert r.degraded_liquidation is True
        assert r.degraded_reason == "sell_dynamic_kill"

    def test_kill_without_balance_forced_hard_block(self) -> None:
        """balance_forced=False + kill → 完全ブロック."""
        gate = _make_gate()
        r = gate.evaluate(**_default_ctx(
            side="buy", is_buy_killed=True, balance_forced=False,
        ))
        assert r.blocked
        assert r.blocking_reason == "buy_dynamic_kill"
        assert r.degraded_liquidation is False

    def test_degraded_disabled_hard_block(self) -> None:
        """degraded_liquidation_enabled=False → balance_forced でも完全ブロック."""
        gate = _make_gate(degraded_liquidation_enabled=False)
        r = gate.evaluate(**_default_ctx(
            side="buy", is_buy_killed=True, balance_forced=True,
        ))
        assert r.blocked
        assert r.blocking_reason == "buy_dynamic_kill"
        assert r.degraded_liquidation is False

    def test_dual_kill_overrides_degraded(self) -> None:
        """dual kill → dual_kill_bypass で通過。degraded は不要."""
        gate = _make_gate()
        r = gate.evaluate(**_default_ctx(
            side="buy", is_buy_killed=True, is_sell_killed=True,
            balance_forced=True,
        ))
        assert not r.blocked
        assert r.dual_kill_bypassed is True
        # dual_kill_bypass で Gate 4/5 不発 → degraded_liquidation=False
        assert r.degraded_liquidation is False


# ═══════════════════════════════════════════════════════════════════════
# P0-A-6: Gate 7 (unknown regime sell) — balance_forced でもブロック
# ═══════════════════════════════════════════════════════════════════════


class TestGate7NoBypass:
    """Gate 7: unknown_regime_sell は balance_forced でもブロック."""

    def test_unknown_sell_blocked_with_balance_forced(self) -> None:
        gate = _make_gate()
        r = gate.evaluate(**_default_ctx(
            side="sell", regime="unknown", balance_forced=True,
        ))
        assert r.blocked
        assert r.blocking_reason == "rule_skip_unknown_sell"


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


class TestDualKillConditionFix:
    """dual_kill 判定から `not balance_forced` が削除されたことを検証."""

    def test_dual_kill_now_detected_with_balance_forced(self) -> None:
        """234# 以前: balance_forced=True → _dual_kill=False.
        234# 以降: balance_forced=True → _dual_kill=True."""
        gate = _make_gate()
        r = gate.evaluate(**_default_ctx(
            side="buy", is_buy_killed=True, is_sell_killed=True,
            balance_forced=True,
        ))
        assert r.dual_kill_bypassed is True

    def test_dual_kill_source_no_balance_forced(self) -> None:
        """evaluate() のソースに `not balance_forced` が _dual_kill 式にないこと."""
        src = inspect.getsource(CycleGateAggregator.evaluate)
        # dual_kill 計算行を取得
        for line in src.splitlines():
            if "_dual_kill" in line and "is_buy_killed" in line:
                assert "not balance_forced" not in line, (
                    f"dual_kill condition still references balance_forced: {line}"
                )


# ═══════════════════════════════════════════════════════════════════════
# 統合: balance_forced + 複合シナリオ
# ═══════════════════════════════════════════════════════════════════════


class TestBalanceForcedIntegration:
    """balance_forced=True での全ゲート通過パターン検証."""

    def test_normal_regime_balance_forced_passes(self) -> None:
        """ranging + vol > threshold + balance_forced → 通過."""
        gate = _make_gate()
        r = gate.evaluate(**_default_ctx(
            side="buy", regime="ranging", vol_ratio=1.0,
            balance_forced=True,
        ))
        assert not r.blocked
        assert r.degraded_liquidation is False

    def test_balance_forced_buy_kill_degraded_sell_passes(self) -> None:
        """buy kill + balance_forced (buy 側) → degraded.
        sell 側は kill なし → 通常通過."""
        gate = _make_gate()
        r_buy = gate.evaluate(**_default_ctx(
            side="buy", is_buy_killed=True, balance_forced=True,
        ))
        assert not r_buy.blocked
        assert r_buy.degraded_liquidation is True

        r_sell = gate.evaluate(**_default_ctx(
            side="sell", is_buy_killed=True, balance_forced=True,
        ))
        assert not r_sell.blocked
        assert r_sell.degraded_liquidation is False

    def test_consecutive_unknown_bypass_still_works_with_balance_forced(self) -> None:
        """MAX_CONSECUTIVE 到達 → balance_forced に関係なく unknown バイパス."""
        gate = _make_gate()
        for _ in range(CycleGateAggregator.UNKNOWN_REGIME_MAX_CONSECUTIVE):
            gate.evaluate(**_default_ctx(side="buy", regime="unknown"))

        r = gate.evaluate(**_default_ctx(
            side="buy", regime="unknown", balance_forced=True,
        ))
        # バイパス + balance_forced (gate bypass なし) → バイパスで通過
        assert not r.blocked


# ═══════════════════════════════════════════════════════════════════════
# 235# dead parameter cleanup 検証
# ═══════════════════════════════════════════════════════════════════════


class TestDeadParameterCleanup:
    """235# balance_forced が _check_* のシグネチャから削除されたことを検証."""

    @pytest.mark.parametrize("method_name", [
        "_check_unknown_regime_buy",
        "_check_ranging_buy_low_vol",
        "_check_trending_sell",
        "_check_buy_dynamic_kill",
        "_check_sell_dynamic_kill",
        "_check_unknown_regime_sell",
    ])
    def test_no_balance_forced_parameter(self, method_name: str) -> None:
        """各 _check_* メソッドのシグネチャに balance_forced がないこと."""
        method = getattr(CycleGateAggregator, method_name)
        sig = inspect.signature(method)
        assert "balance_forced" not in sig.parameters, (
            f"235# dead parameter: {method_name} still has balance_forced"
        )


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


class TestDeadConfigDeprecation:
    """253# balance_forced_apply_trending_offset 完全削除済."""

    def test_field_removed_253(self) -> None:
        """253# フィールド削除済 (234# dead config → 253# 完全削除)."""
        cfg = FillTestConfig()
        assert not hasattr(cfg, "balance_forced_apply_trending_offset")

    def test_field_not_used_in_gate_aggregator(self) -> None:
        """Gate aggregator のソースで直接参照されていないこと."""
        src = inspect.getsource(CycleGateAggregator)
        assert "balance_forced_apply_trending_offset" not in src
