"""194# CycleGateAggregator 単体テスト.

192# §3 の「判断箇所一元化」に対応する CycleGateAggregator の
全ゲートを個別・統合でテスト。
"""

from __future__ import annotations

import pytest

from scripts.v460.lib.cycle_gate_aggregator import (
    CycleGateAggregator,
    CycleGateResult,
    GateCheckResult,
)

from tests.unit.v460.conftest import make_gate_config as _make_config


# ─── ヘルパー ──────────────────────────────────────────────────────────


def _make_gate(**overrides: object) -> CycleGateAggregator:
    """テスト用 CycleGateAggregator."""
    return CycleGateAggregator(_make_config(**overrides))


def _default_ctx(**overrides: object) -> dict:
    """evaluate() のデフォルト引数."""
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


# ─── 1. Gate 1: unknown_regime_buy ────────────────────────────────────


class TestUnknownRegimeBuy:
    """A10: unknown regime での buy スキップ."""

    def test_blocks_buy_in_unknown(self) -> None:
        gate = _make_gate()
        r = gate.evaluate(**_default_ctx(side="buy", regime="unknown"))
        assert r.blocked
        assert r.blocking_reason == "unknown_regime_buy_skip"

    def test_allows_buy_in_ranging(self) -> None:
        gate = _make_gate()
        r = gate.evaluate(**_default_ctx(side="buy", regime="ranging"))
        assert not r.blocked

    def test_allows_sell_in_unknown(self) -> None:
        """buy だけブロック、sell は Gate 7 で別途判定."""
        gate = _make_gate(skip_sell_unknown_regime=False)
        r = gate.evaluate(**_default_ctx(side="sell", regime="unknown"))
        assert not r.blocked

    def test_disabled_config(self) -> None:
        gate = _make_gate(skip_buy_unknown_regime=False)
        r = gate.evaluate(**_default_ctx(side="buy", regime="unknown"))
        assert not r.blocked


# ─── 2. Gate 2: ranging_buy_low_vol ──────────────────────────────────


class TestRangingBuyLowVol:
    """A11: B1' ranging buy at low vol."""

    def test_blocks_low_vol(self) -> None:
        gate = _make_gate()
        r = gate.evaluate(**_default_ctx(side="buy", regime="ranging", vol_ratio=0.5))
        assert r.blocked
        assert r.blocking_reason == "ranging_low_vol_skip"

    def test_allows_high_vol(self) -> None:
        gate = _make_gate()
        r = gate.evaluate(**_default_ctx(side="buy", regime="ranging", vol_ratio=0.9))
        assert not r.blocked

    def test_allows_trending(self) -> None:
        gate = _make_gate()
        r = gate.evaluate(**_default_ctx(side="buy", regime="trending", vol_ratio=0.5))
        assert not r.blocked


# ─── 3. Gate 3: trending_sell ────────────────────────────────────────


class TestTrendingSell:
    """A12: trending regime での sell 抑制."""

    def test_blocks_sell_in_trending_up(self) -> None:
        gate = _make_gate()
        r = gate.evaluate(**_default_ctx(side="sell", regime="trending_up"))
        assert r.blocked
        assert r.blocking_reason == "trending_sell_skip"

    def test_blocks_sell_in_trending(self) -> None:
        gate = _make_gate()
        r = gate.evaluate(**_default_ctx(side="sell", regime="trending"))
        assert r.blocked

    def test_allows_buy_in_trending(self) -> None:
        gate = _make_gate()
        r = gate.evaluate(**_default_ctx(side="buy", regime="trending"))
        assert not r.blocked

    def test_trending_up_only_allows_trending_down(self) -> None:
        """176# A: trending_up_only モード."""
        gate = _make_gate(skip_sell_trending_up_only=True)
        r = gate.evaluate(**_default_ctx(side="sell", regime="trending_down"))
        assert not r.blocked

    def test_trending_up_only_blocks_trending_up(self) -> None:
        gate = _make_gate(skip_sell_trending_up_only=True)
        r = gate.evaluate(**_default_ctx(side="sell", regime="trending_up"))
        assert r.blocked

    def test_inv_bypass_allows_sell(self) -> None:
        """171# Guard Paradox: inv >= threshold → sell 許可."""
        gate = _make_gate()
        r = gate.evaluate(**_default_ctx(
            side="sell", regime="trending_up", inv_net_imbalance=0.5,
        ))
        assert not r.blocked

    def test_consecutive_safety_valve(self) -> None:
        """158# §20-B: 連続 skip >= max → 強制許可."""
        gate = _make_gate(max_consecutive_trending_sell_skip=5)
        r = gate.evaluate(**_default_ctx(
            side="sell", regime="trending_up",
            trending_sell_skip_count=5,
        ))
        assert not r.blocked

    def test_consecutive_below_limit_blocks(self) -> None:
        gate = _make_gate(max_consecutive_trending_sell_skip=5)
        r = gate.evaluate(**_default_ctx(
            side="sell", regime="trending_up",
            trending_sell_skip_count=4,
        ))
        assert r.blocked

    def test_hf4_buy_insufficient_allows(self) -> None:
        """166# HF4: buy 側残高不足 → sell 許可."""
        gate = _make_gate()
        r = gate.evaluate(**_default_ctx(
            side="sell", regime="trending_up",
            buy_side_insufficient=True,
        ))
        assert not r.blocked


# ─── 4. Gate 4: buy_dynamic_kill ─────────────────────────────────────


class TestBuyDynamicKill:
    """A13: buy 動的 kill."""

    def test_blocks_when_killed(self) -> None:
        gate = _make_gate()
        r = gate.evaluate(**_default_ctx(side="buy", is_buy_killed=True))
        assert r.blocked
        assert r.blocking_reason == "buy_dynamic_kill"

    def test_allows_when_not_killed(self) -> None:
        gate = _make_gate()
        r = gate.evaluate(**_default_ctx(side="buy", is_buy_killed=False))
        assert not r.blocked



# ─── 5. Gate 5: sell_dynamic_kill ────────────────────────────────────


class TestSellDynamicKill:
    """A14: sell 動的 kill."""

    def test_blocks_when_killed(self) -> None:
        gate = _make_gate()
        r = gate.evaluate(**_default_ctx(side="sell", regime="ranging", is_sell_killed=True))
        assert r.blocked
        assert r.blocking_reason == "sell_dynamic_kill"

    def test_inv_bypass_allows(self) -> None:
        """171# inv bypass: buy 偏重時は sell kill を解除."""
        gate = _make_gate()
        r = gate.evaluate(**_default_ctx(
            side="sell", regime="ranging",
            is_sell_killed=True, inv_net_imbalance=0.5,
        ))
        assert not r.blocked



# ─── 6. Gate 6: velocity_skip ────────────────────────────────────────


class TestVelocitySkip:
    """C4-C5: velocity-based skip."""

    def test_sell_velocity_skip(self) -> None:
        gate = _make_gate()
        r = gate.evaluate(**_default_ctx(
            side="sell", regime="ranging", price_velocity_bps=10.0,
        ))
        assert r.blocked
        assert r.blocking_reason == "rule_velocity_sell_skip"

    def test_buy_velocity_skip(self) -> None:
        gate = _make_gate()
        r = gate.evaluate(**_default_ctx(
            side="buy", regime="ranging", price_velocity_bps=-10.0,
        ))
        assert r.blocked
        assert r.blocking_reason == "rule_velocity_buy_skip"

    def test_no_velocity_data_passes(self) -> None:
        gate = _make_gate()
        r = gate.evaluate(**_default_ctx(
            side="sell", regime="ranging",
        ))
        assert not r.blocked

    def test_within_threshold_passes(self) -> None:
        gate = _make_gate()
        r = gate.evaluate(**_default_ctx(
            side="sell", regime="ranging", price_velocity_bps=5.0,
        ))
        assert not r.blocked


# ─── 7. Gate 7: unknown_regime_sell ──────────────────────────────────


class TestUnknownRegimeSell:
    """C2: unknown regime での sell skip."""

    def test_blocks_sell_in_unknown(self) -> None:
        gate = _make_gate()
        r = gate.evaluate(**_default_ctx(side="sell", regime="unknown"))
        # Gate 1 only blocks buy, Gate 7 blocks sell
        assert r.blocked
        assert r.blocking_reason == "rule_skip_unknown_sell"

    def test_allows_sell_in_trending(self) -> None:
        gate = _make_gate()
        r = gate.evaluate(**_default_ctx(side="sell", regime="ranging"))
        assert not r.blocked


# ─── 8. 統合テスト ───────────────────────────────────────────────────


class TestIntegration:
    """全ゲートの統合評価テスト."""

    def test_all_pass_no_block(self) -> None:
        """全条件クリア → blocked=False."""
        gate = _make_gate()
        r = gate.evaluate(**_default_ctx(side="buy", regime="ranging", vol_ratio=1.0))
        assert not r.blocked
        assert len(r.checks) == 9  # 197#: 7→9 ゲートに拡張
        assert all(not c.blocked for c in r.checks)

    def test_first_gate_blocks_early_exit(self) -> None:
        """最初のゲートでブロック → 後続ゲート未評価."""
        gate = _make_gate()
        r = gate.evaluate(**_default_ctx(side="buy", regime="unknown"))
        assert r.blocked
        assert len(r.checks) == 1  # Gate 1 のみ
        assert r.checks[0].gate_name == "unknown_regime_buy"

    def test_audit_summary(self) -> None:
        gate = _make_gate()
        r = gate.evaluate(**_default_ctx(side="buy", regime="ranging", vol_ratio=1.0))
        summary = r.audit_summary
        assert "✓" in summary
        assert "unknown_regime_buy" in summary

    def test_cancel_reason_mapping(self) -> None:
        gate = _make_gate()
        r = gate.evaluate(**_default_ctx(side="buy", regime="unknown"))
        assert r.cancel_reason == "unknown_regime_buy_skip"

    def test_cancel_reason_velocity(self) -> None:
        gate = _make_gate()
        r = gate.evaluate(**_default_ctx(
            side="sell", regime="ranging", price_velocity_bps=10.0,
        ))
        assert r.cancel_reason == "skip_gate_rule_velocity_sell"

    def test_cancel_reason_unknown_sell(self) -> None:
        gate = _make_gate()
        r = gate.evaluate(**_default_ctx(side="sell", regime="unknown"))
        assert r.cancel_reason == "unknown_regime_sell_skip"

    def test_priority_unknown_regime_buy_over_ranging_low_vol(self) -> None:
        """unknown regime と ranging_low_vol が同時に該当 → unknown が先."""
        # unknown ≠ ranging なので実際に同時にはならないが、設計確認として
        gate = _make_gate()
        r = gate.evaluate(**_default_ctx(side="buy", regime="unknown", vol_ratio=0.3))
        assert r.blocking_reason == "unknown_regime_buy_skip"

    def test_disabled_all_gates(self) -> None:
        """全ゲート無効 → 何も blocked しない."""
        gate = _make_gate(
            skip_buy_unknown_regime=False,
            skip_ranging_buy_low_vol=False,
            skip_sell_trending=False,
            buy_dynamic_kill_enabled=False,
            sell_dynamic_kill_enabled=False,
            sell_velocity_skip_enabled=False,
            buy_velocity_skip_enabled=False,
            skip_sell_unknown_regime=False,
        )
        r = gate.evaluate(**_default_ctx(side="buy", regime="unknown", is_buy_killed=True))
        assert not r.blocked

    def test_sell_in_trending_blocked_then_safety_valve_releases(self) -> None:
        """trending sell ブロック → consecutive safety valve で解放."""
        gate = _make_gate(max_consecutive_trending_sell_skip=3)

        # 2回目 → まだブロック
        r = gate.evaluate(**_default_ctx(
            side="sell", regime="trending_up", trending_sell_skip_count=2,
        ))
        assert r.blocked

        # 3回目 → safety valve 発動
        r = gate.evaluate(**_default_ctx(
            side="sell", regime="trending_up", trending_sell_skip_count=3,
        ))
        assert not r.blocked


class TestCompoundSuppression:
    """451# P1-2: ranging_low_vol + buy_dynamic_kill compound suppression."""

    def test_speculative_buy_kill_recorded(self) -> None:
        """Gate 2 block 時に Gate 4 も投機的チェックされ、speculative_checks に記録."""
        gate = _make_gate(ranging_buy_low_vol_as_offset=False)
        r = gate.evaluate(**_default_ctx(
            side="buy", regime="ranging",
            vol_ratio=0.3,   # low vol → Gate 2 block
            is_buy_killed=True,  # Gate 4 would also block
        ))
        assert r.blocked
        assert r.blocking_reason == "ranging_low_vol_skip"
        assert len(r.speculative_checks) == 1
        assert r.speculative_checks[0].gate_name == "buy_dynamic_kill"
        assert r.speculative_checks[0].blocked
        assert "(✗buy_dynamic_kill)" in r.audit_summary

    def test_no_speculative_when_not_killed(self) -> None:
        """buy_dynamic_kill 非発動時は speculative_checks が空."""
        gate = _make_gate(ranging_buy_low_vol_as_offset=False)
        r = gate.evaluate(**_default_ctx(
            side="buy", regime="ranging",
            vol_ratio=0.3,
            is_buy_killed=False,
        ))
        assert r.blocked
        assert r.blocking_reason == "ranging_low_vol_skip"
        assert len(r.speculative_checks) == 0

    def test_no_speculative_for_sell(self) -> None:
        """sell 側は compound suppression 診断対象外."""
        gate = _make_gate(ranging_buy_low_vol_as_offset=False)
        r = gate.evaluate(**_default_ctx(
            side="sell", regime="ranging",
            vol_ratio=0.3,
            is_buy_killed=True,
        ))
        # ranging_low_vol_skip は buy 限定なので sell はブロックされない
        assert not r.blocked or r.blocking_reason != "ranging_low_vol_skip"
        assert len(r.speculative_checks) == 0
