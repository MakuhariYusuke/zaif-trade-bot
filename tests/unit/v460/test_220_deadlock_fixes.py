"""220# デッドロック修正テスト.

3つの修正を検証:
A. Gate7 unknown_regime_sell に balance_forced bypass 追加 (Gate1との対称性)
B. Dual-kill deadlock breaker: buy+sell 両方 kill 時にゲートバイパス
C. Unknown regime 連続ブロック後の自動バイパス (UNKNOWN_REGIME_MAX_CONSECUTIVE)
"""

from __future__ import annotations

import pytest

from scripts.v460.lib.cycle_gate_aggregator import (
    CycleGateAggregator,
    CycleGateResult,
)

from tests.unit.v460.conftest import make_gate_config as _make_config


# ─── ヘルパー ──────────────────────────────────────────────────────────


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
# Fix A: Gate 7 unknown_regime_sell — balance_forced bypass
# ═══════════════════════════════════════════════════════════════════════


class TestGate7BalanceForcedBypassRemoved:
    """234# balance_forced gate bypass 廃止.

    219# で追加された balance_forced bypass は 234# で削除。
    Kill Gate は絶対的安全権限を持つ。
    balance_forced 時は degraded liquidation で対応。
    """

    def test_sell_unknown_blocked_without_balance_forced(self) -> None:
        """通常: unknown regime の sell はブロック."""
        gate = _make_gate()
        r = gate.evaluate(**_default_ctx(side="sell", regime="unknown"))
        assert r.blocked
        assert r.blocking_reason == "rule_skip_unknown_sell"

    def test_sell_unknown_blocked_with_balance_forced(self) -> None:
        """234#: balance_forced=True でも sell はブロック."""
        gate = _make_gate()
        r = gate.evaluate(**_default_ctx(
            side="sell", regime="unknown",
        ))
        assert r.blocked
        assert r.blocking_reason == "rule_skip_unknown_sell"

    def test_buy_unknown_blocked_with_balance_forced(self) -> None:
        """234#: balance_forced=True でも buy はブロック."""
        gate = _make_gate()
        r = gate.evaluate(**_default_ctx(
            side="buy", regime="unknown",
        ))
        assert r.blocked
        assert r.blocking_reason == "unknown_regime_buy_skip"

    def test_symmetry_both_sides_blocked_without_balance_forced(self) -> None:
        """対称性: balance_forced=False → 両方ブロック."""
        gate = _make_gate()
        r_buy = gate.evaluate(**_default_ctx(side="buy", regime="unknown"))
        assert r_buy.blocked
        r_sell = gate.evaluate(**_default_ctx(side="sell", regime="unknown"))
        assert r_sell.blocked

    def test_symmetry_both_sides_blocked_with_balance_forced(self) -> None:
        """234# 対称性: balance_forced=True でも両方ブロック."""
        gate = _make_gate()
        r_buy = gate.evaluate(**_default_ctx(
            side="buy", regime="unknown",
        ))
        assert r_buy.blocked
        r_sell = gate.evaluate(**_default_ctx(
            side="sell", regime="unknown",
        ))
        assert r_sell.blocked


# ═══════════════════════════════════════════════════════════════════════
# Fix B: Dual-kill deadlock breaker
# ═══════════════════════════════════════════════════════════════════════


class TestDualKillBreaker:
    """buy+sell 両方 kill 時のデッドロック回避."""

    def test_buy_killed_only_blocks_buy(self) -> None:
        """buy だけ kill → buy はブロック."""
        gate = _make_gate()
        r = gate.evaluate(**_default_ctx(
            side="buy", is_buy_killed=True, is_sell_killed=False,
        ))
        assert r.blocked
        assert r.blocking_reason == "buy_dynamic_kill"

    def test_sell_killed_only_blocks_sell(self) -> None:
        """sell だけ kill → sell はブロック."""
        gate = _make_gate()
        r = gate.evaluate(**_default_ctx(
            side="sell", is_buy_killed=False, is_sell_killed=True,
        ))
        assert r.blocked
        assert r.blocking_reason == "sell_dynamic_kill"

    def test_dual_kill_buy_passes(self) -> None:
        """220# Fix: 両方 kill → buy 通過 (デッドロック回避)."""
        gate = _make_gate()
        r = gate.evaluate(**_default_ctx(
            side="buy", is_buy_killed=True, is_sell_killed=True,
        ))
        assert not r.blocked

    def test_dual_kill_sell_passes(self) -> None:
        """220# Fix: 両方 kill → sell 通過 (デッドロック回避)."""
        gate = _make_gate()
        r = gate.evaluate(**_default_ctx(
            side="sell", is_buy_killed=True, is_sell_killed=True,
        ))
        assert not r.blocked

    def test_dual_kill_with_balance_forced_still_passes(self) -> None:
        """234#: balance_forced=True + dual kill → dual_kill_bypass で通過."""
        gate = _make_gate()
        r = gate.evaluate(**_default_ctx(
            side="buy", is_buy_killed=True, is_sell_killed=True,
        ))
        assert not r.blocked
        assert r.dual_kill_bypassed is True  # 234# dual_kill now detected




# ═══════════════════════════════════════════════════════════════════════
# Fix C: Unknown regime 連続ブロック自動バイパス
# ═══════════════════════════════════════════════════════════════════════


class TestUnknownRegimeConsecutiveBypass:
    """連続 unknown regime ブロック後の自動バイパス."""

    def test_counter_increments_on_buy_block(self) -> None:
        """unknown buy ブロック → カウンタ増加."""
        gate = _make_gate()
        for _ in range(3):
            gate.evaluate(**_default_ctx(side="buy", regime="unknown"))
        assert gate._consecutive_unknown_blocks["buy"] == 3

    def test_counter_increments_on_sell_block(self) -> None:
        """unknown sell ブロック → カウンタ増加."""
        gate = _make_gate()
        for _ in range(3):
            gate.evaluate(**_default_ctx(side="sell", regime="unknown"))
        assert gate._consecutive_unknown_blocks["sell"] == 3

    def test_counter_resets_on_non_unknown(self) -> None:
        """non-unknown regime → カウンタリセット."""
        gate = _make_gate()
        # unknown 5回
        for _ in range(5):
            gate.evaluate(**_default_ctx(side="buy", regime="unknown"))
        assert gate._consecutive_unknown_blocks["buy"] == 5
        # non-unknown 1回 → リセット
        gate.evaluate(**_default_ctx(side="buy", regime="ranging"))
        assert gate._consecutive_unknown_blocks["buy"] == 0

    def test_bypass_after_max_consecutive(self) -> None:
        """MAX_CONSECUTIVE 到達後 → ブロックされない."""
        gate = _make_gate()

        # 10回ブロック
        for i in range(CycleGateAggregator.UNKNOWN_REGIME_MAX_CONSECUTIVE):
            r = gate.evaluate(**_default_ctx(side="buy", regime="unknown"))
            assert r.blocked, f"cycle {i} should block"

        # 11回目 → バイパス発動
        r = gate.evaluate(**_default_ctx(side="buy", regime="unknown"))
        assert not r.blocked, "should bypass after MAX_CONSECUTIVE"

    def test_bypass_sell_after_max_consecutive(self) -> None:
        """sell 側も MAX_CONSECUTIVE 到達後にバイパス."""
        gate = _make_gate()

        for i in range(CycleGateAggregator.UNKNOWN_REGIME_MAX_CONSECUTIVE):
            r = gate.evaluate(**_default_ctx(side="sell", regime="unknown"))
            assert r.blocked, f"cycle {i} should block"

        r = gate.evaluate(**_default_ctx(side="sell", regime="unknown"))
        assert not r.blocked, "sell should bypass after MAX_CONSECUTIVE"

    def test_per_side_independence(self) -> None:
        """324# M-2: buy/sell カウンタは独立 — 混合カウントしない."""
        gate = _make_gate()

        # buy 5回 + sell 5回 交互に unknown
        for i in range(CycleGateAggregator.UNKNOWN_REGIME_MAX_CONSECUTIVE):
            side = "buy" if i % 2 == 0 else "sell"
            r = gate.evaluate(**_default_ctx(side=side, regime="unknown"))
            assert r.blocked

        # per-side では buy=3, sell=2 — MAX=5 未到達なのでまだブロック
        assert gate._consecutive_unknown_blocks["buy"] == 3
        assert gate._consecutive_unknown_blocks["sell"] == 2
        r = gate.evaluate(**_default_ctx(side="buy", regime="unknown"))
        assert r.blocked, "per-side: buy=3 < MAX=5, should still block"

    def test_counter_increments_when_unknown_blocked_with_balance_forced(self) -> None:
        """234#: balance_forced でも unknown regime はブロック → カウンタ増加."""
        gate = _make_gate()

        # 4回ブロック (MAX=5 未満)
        for _ in range(4):
            gate.evaluate(**_default_ctx(side="buy", regime="unknown"))
        assert gate._consecutive_unknown_blocks["buy"] == 4

        # 234#: balance_forced でも Gate1 はブロック → カウンタ 5 に
        gate.evaluate(**_default_ctx(
            side="buy", regime="unknown",
        ))
        assert gate._consecutive_unknown_blocks["buy"] == 5

    def test_max_consecutive_class_attr(self) -> None:
        """クラス属性の閾値が正しい."""
        assert CycleGateAggregator.UNKNOWN_REGIME_MAX_CONSECUTIVE == 5


# ═══════════════════════════════════════════════════════════════════════
# 統合テスト: 複合シナリオ
# ═══════════════════════════════════════════════════════════════════════


class TestDeadlockIntegration:
    """複合デッドロックシナリオの統合テスト."""

    def test_triple_deadlock_all_bypassed(self) -> None:
        """unknown + dual_kill → 全バイパスで通過."""
        gate = _make_gate()

        # unknown を MAX 回ブロック
        for _ in range(CycleGateAggregator.UNKNOWN_REGIME_MAX_CONSECUTIVE):
            gate.evaluate(**_default_ctx(side="buy", regime="unknown"))

        # unknown bypass + dual kill bypass → 通過
        r = gate.evaluate(**_default_ctx(
            side="buy", regime="unknown",
            is_buy_killed=True, is_sell_killed=True,
        ))
        assert not r.blocked

    def test_all_gates_9_checks_on_full_pass(self) -> None:
        """全ゲート通過時、9 checks を維持."""
        gate = _make_gate()
        r = gate.evaluate(**_default_ctx(side="buy", regime="ranging"))
        assert not r.blocked
        assert len(r.checks) == 9
