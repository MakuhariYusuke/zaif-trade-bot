"""220# デッドロック修正テスト.

3つの修正を検証:
A. Gate7 unknown_regime_sell に balance_forced bypass 追加 (Gate1との対称性)
B. Dual-kill deadlock breaker: buy+sell 両方 kill 時にゲートバイパス
C. Unknown regime 連続ブロック後の自動バイパス (UNKNOWN_REGIME_MAX_CONSECUTIVE)
"""

from __future__ import annotations

import pytest

from scripts.v460.lib.fill_config import FillTestConfig
from scripts.v460.lib.cycle_gate_aggregator import (
    CycleGateAggregator,
    CycleGateResult,
)


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
# Fix A: Gate 7 unknown_regime_sell — balance_forced bypass
# ═══════════════════════════════════════════════════════════════════════


class TestGate7BalanceForcedBypass:
    """Gate7 に balance_forced bypass を追加 (Gate1 との対称性修正)."""

    def test_sell_unknown_blocked_without_balance_forced(self) -> None:
        """通常: unknown regime の sell はブロック."""
        gate = _make_gate()
        r = gate.evaluate(**_default_ctx(side="sell", regime="unknown"))
        assert r.blocked
        assert r.blocking_reason == "rule_skip_unknown_sell"

    def test_sell_unknown_bypassed_with_balance_forced(self) -> None:
        """219# Fix: balance_forced=True なら sell も通過."""
        gate = _make_gate()
        r = gate.evaluate(**_default_ctx(
            side="sell", regime="unknown", balance_forced=True,
        ))
        assert not r.blocked

    def test_buy_unknown_bypassed_with_balance_forced(self) -> None:
        """Gate1 の既存動作: balance_forced=True なら buy も通過."""
        gate = _make_gate()
        r = gate.evaluate(**_default_ctx(
            side="buy", regime="unknown", balance_forced=True,
        ))
        assert not r.blocked

    def test_symmetry_both_sides_blocked_without_balance_forced(self) -> None:
        """対称性: balance_forced=False → 両方ブロック."""
        gate = _make_gate()
        r_buy = gate.evaluate(**_default_ctx(side="buy", regime="unknown"))
        assert r_buy.blocked
        r_sell = gate.evaluate(**_default_ctx(side="sell", regime="unknown"))
        assert r_sell.blocked

    def test_symmetry_both_sides_pass_with_balance_forced(self) -> None:
        """対称性: balance_forced=True → 両方通過."""
        gate = _make_gate()
        r_buy = gate.evaluate(**_default_ctx(
            side="buy", regime="unknown", balance_forced=True,
        ))
        assert not r_buy.blocked
        r_sell = gate.evaluate(**_default_ctx(
            side="sell", regime="unknown", balance_forced=True,
        ))
        assert not r_sell.blocked


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
        """balance_forced=True + dual kill → 通過 (balance_forced 単独でも通過)."""
        gate = _make_gate()
        r = gate.evaluate(**_default_ctx(
            side="buy", is_buy_killed=True, is_sell_killed=True,
            balance_forced=True,
        ))
        assert not r.blocked

    def test_single_kill_balance_forced_passes(self) -> None:
        """片方 kill + balance_forced → 通過 (既存動作)."""
        gate = _make_gate()
        r = gate.evaluate(**_default_ctx(
            side="buy", is_buy_killed=True, is_sell_killed=False,
            balance_forced=True,
        ))
        assert not r.blocked

    def test_dual_kill_does_not_bypass_with_balance_forced(self) -> None:
        """balance_forced=True は dual_kill_bypass を無効化する (独立制御)."""
        gate = _make_gate()
        # balance_forced=True の場合、_dual_kill conditionでは
        # `not balance_forced` が False なので _dual_kill=False。
        # ただし balance_forced 単独で Gate4/5 をバイパスするので通過。
        r = gate.evaluate(**_default_ctx(
            side="sell", is_buy_killed=True, is_sell_killed=True,
            balance_forced=True,
        ))
        assert not r.blocked  # balance_forced で通過


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
        assert gate._consecutive_unknown_blocks == 3

    def test_counter_increments_on_sell_block(self) -> None:
        """unknown sell ブロック → カウンタ増加."""
        gate = _make_gate()
        for _ in range(3):
            gate.evaluate(**_default_ctx(side="sell", regime="unknown"))
        assert gate._consecutive_unknown_blocks == 3

    def test_counter_resets_on_non_unknown(self) -> None:
        """non-unknown regime → カウンタリセット."""
        gate = _make_gate()
        # unknown 5回
        for _ in range(5):
            gate.evaluate(**_default_ctx(side="buy", regime="unknown"))
        assert gate._consecutive_unknown_blocks == 5
        # non-unknown 1回 → リセット
        gate.evaluate(**_default_ctx(side="buy", regime="ranging"))
        assert gate._consecutive_unknown_blocks == 0

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

    def test_mixed_buy_sell_unknown_counts_together(self) -> None:
        """buy/sell 交互に unknown → 合算カウント."""
        gate = _make_gate()

        for i in range(CycleGateAggregator.UNKNOWN_REGIME_MAX_CONSECUTIVE):
            side = "buy" if i % 2 == 0 else "sell"
            r = gate.evaluate(**_default_ctx(side=side, regime="unknown"))
            assert r.blocked

        # 次はバイパス
        r = gate.evaluate(**_default_ctx(side="buy", regime="unknown"))
        assert not r.blocked

    def test_counter_not_reset_when_unknown_passes_via_balance_forced(self) -> None:
        """balance_forced で unknown が通過しても、regime がまだ unknown ならリセットしない."""
        gate = _make_gate()

        # 5回ブロック
        for _ in range(5):
            gate.evaluate(**_default_ctx(side="buy", regime="unknown"))
        assert gate._consecutive_unknown_blocks == 5

        # balance_forced で通過 → unknown のまま → リセットしない
        gate.evaluate(**_default_ctx(
            side="buy", regime="unknown", balance_forced=True,
        ))
        # balance_forced bypass で Gate1 通過、Gate7 は side=buy なのでスルー
        # regime=unknown だが blocked ではない → リセットしない (pass ブロック)
        assert gate._consecutive_unknown_blocks == 5

    def test_max_consecutive_class_attr(self) -> None:
        """クラス属性の閾値が正しい."""
        assert CycleGateAggregator.UNKNOWN_REGIME_MAX_CONSECUTIVE == 10


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
